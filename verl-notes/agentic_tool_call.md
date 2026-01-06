# Tool Calling of Agentic RL in Verl 

在agentic rl的训练过程中，tool call的处理是核心。定义好的tool是怎么传给模型的，模型的tool call请求又是怎么被处理的？这篇笔记结合sglang rollout源码进行浅谈

## 工具传入

工具定义：在verl/verl/tools路径下有很多定义好的工具，根据自己的需求定义一个新工具即可，注意要提供create,execute接口供系统调用。我定义了一个计算器工具，字典形式如下：
```
{
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "A mathematical calculator tool for evaluating complex expressions. It supports basic arithmetic operations (+, -, *, /) and functions from the 'math' library (e.g., math.sin(), math.pi).",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "The mathematical expression to be calculated, such as '10 * (5 + 2) / 3' or 'math.sqrt(9) + math.log(10)'."
                }
            },
            "required": ["expression"]
        }
    }
}
```

在定义好工具后，再仿照examples/sglang_multiturn/config下的配置文件为自己的工具定义一个config。在训练脚本中通过actor_rollout_ref.rollout.multi_turn.tool_config_path指定此config。
同时，在训练数据集文件train.parquet中，也要加上tools_kwargs一项，按字典形式将工具写入数据集。

接下来介绍工具字段在训练过程中的传递。在RayPPOTrainer初始化时，会初始化RLHFDataset类。在RLHFDataset类内部的get_item方法中，数据集的tools_kwargs中的内容会被取出作为工具定义。

进入主训练函数RayPPOTrainer.fit()，其中的rollout语句为：
```angular2html
gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
```
调用了ActorRolloutRefWorker类的generate_sequences函数，这个函数中又调用了SGLangRollout(ActorRolloutRefWorker的rollout属性的值就是SGLangRollout)中的generate_sequences。
```angular2html
with simple_timer("generate_sequences", timing_generate):
    output = self.rollout.generate_sequences(prompts=prompts)
```
SGLangRollout的chat_completion函数中将prompt里的tools信息传给了req。
```angular2html
 async def chat_completion(self, json_request):
        req = AsyncRolloutRequest(
            ...
            tool_schemas=_tool_schemas,
            tools_kwargs=_tools_kwargs,
            ...
        )
```

SGLangRollout的generate_sequences方法中，在多轮条件下调用了_req_level_generate_sequences。
```angular2html
def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
    if self.config.multi_turn.enable:
        return self._req_level_generate_sequences(prompts, **kwargs)
    return self._batch_level_generate_sequences(prompts, **kwargs)
```
_req_level_generate_sequences是指在batch中按逐请求(req)的方式，通过调用_async_rollout_a_request进行rollout生成答案。
```angular2html
def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        output_req_list = loop.run_until_complete(
            asyncio.gather(
                *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
            )
        )
```

_async_rollout_a_request是管理工具调用过程的核心函数，维护了一个工具调用的状态机（参考[Awesome-ML-SYS-Tutorial](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/rlhf/verl/multi-turn/code-walk-through/readme.md)）。
刚进入时，默认为PENDING状态，调用初始化函数进行工具的初始化时，转入到RUNNING状态。此时调用_handle_engine_call函数进行实际的rollout过程。
```angular2html
output = await self._handle_engine_call(_req, request_sampling_params, image_data=image_data)
```
_handle_engine_call将req中的内容转为tokenizer对应的id，然后调用_handle_engine_generate进行生成。
```angular2html
    async def _handle_engine_call(
        self, _req: AsyncRolloutRequest, sampling_params: dict, image_data: Optional[list[Any]] = None
    ) -> dict:
        generation_prompt_ids = _req.get_generation_prompt_ids(self.processing_class)
        return await self._handle_engine_generate(generation_prompt_ids, sampling_params, image_data)
```

到这里出现了问题。之前的工具信息在req中，现在却消失了。答案就在get_generation_prompt_ids函数中。
```angular2html
def get_generation_prompt_ids(
    self, processing_class: PreTrainedTokenizer | PreTrainedTokenizerFast | ProcessorMixin
) -> list[int]:
        ...
        tools = [tool.model_dump() for tool in self.tool_schemas] if self.tool_schemas else None
        generation_prompt_ids = self._handle_apply_chat_template(
            processing_class,
            messages,
            multi_modal_data=self.multi_modal_data,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )
        ...
```
其中，tool.model_dump()将工具转化为字典形式，便于后续处理。_handle_apply_chat_template()将工具以文本形式注入了聊天模板。
具体而言：tools字段会被传递给jinja渲染模块，接着被转成字符串并加上某些引导词，放在System Prompt的末尾。以我使用的Qwen3-4B模型为例，其tokenizer_config.json中定义的处理工具的方式为（部份）：
```
{%- if tools %}
    {{- '<|im_start|>system\n' }}
    {%- if messages[0].role == 'system' %}
        {{- messages[0].content + '\n\n' }}
    {%- endif %}
    {{- "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
   ...
   {%- if message.tool_calls %}
    {%- for tool_call in message.tool_calls %}
        <tool_call>
        {"name": "{{ tool_call.name }}", "arguments": {{ tool_call.arguments | tojson }}}
        </tool_call>
    {%- endfor %}
{%- endif %}
```
这里用prompt的形式告诉模型：有哪些工具可以调用，以及工具调用必须按<tool_call> </tool_call>的模板进行。

至此得到一个结论：**Agentic RL的工具调用过程其实没什么特殊，就是封装在比较底层的一个prompt实现**

## 工具输出

那么tool_call返回结果后，又是怎么处理的呢？在SGLangRollout的_async_rollout_a_request的RUNNING处理模块中，有这样的一段：
```angular2html
if self._function_call_parser and self._function_call_parser.has_tool_call(content):
    _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
```
其中，content就是模型返回的文本内容。self._function_call_parser.has_tool_call()函数，它在SGLangRollout的initialize()中被定义，经过层层剥离，最终指向所用模型Qwen3-4B的function_call_parser，即Qwen25Detector
```angular2html
    self.bot_token = "<tool_call>"
    self.eot_token = "</tool_call>"

def has_tool_call(self, text: str) -> bool:
    """Check if the text contains a Qwen 2.5 format tool call."""
    return self.bot_token in text

```
实现逻辑很简单：就找返回文本中的<tool_call> </tool_call>字段，检测到就发起工具调用。发起后，状态机进入TOOL_CALLING状态：
```angular2html
while current_turns < self.config.multi_turn.max_assistant_turns:
    ...
    elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
        if _req.messages[-1].tool_calls is not None:
            parsed_tool_calls = _req.messages[-1].tool_calls
            tool_call_results = await asyncio.gather(
                *[
                    self._tool_map[tool_call.function.name].execute(
                        _req.request_id,
                        tool_call.function.arguments,
                        **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
                    )
                    for tool_call in parsed_tool_calls
                ]
            )
            _req.add_tool_response_messages(self.processing_class, [resp for resp, _, _ in tool_call_results])
            # print('tool_call_results', tool_call_results)
            for tool_call, (resp, reward, metrics) in zip(parsed_tool_calls, tool_call_results, strict=True):
                _req.update_metrics(metrics, tool_call.function.name)
            if len(_req.input_ids) >= self.config.max_model_len:
                finish_reason_type = FinishReasonTypeEnum.STOP
                break
            _req.state = AsyncRolloutRequestStateEnum.RUNNING
```
进入TOOL_CALLING后，根据tool_call请求调用相应工具函数的execute()接口，进行实际工具调用。调用的结果存在历史消息中，便于在下一轮被模型看到，决定要不要继续调用。直到模型不再发起调用或者到达最大轮次。