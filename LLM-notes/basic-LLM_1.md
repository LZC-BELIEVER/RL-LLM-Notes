# LLM基本框架理解

对LLM的基本框架对理解是进行强化微调等任务的基础。本笔记将记录我对LLM基础架构的一些学习过程。

作者：LZC from CUHKSZ

参考：Standford CS-336

## Tokenization
Tokenization指将输入的序列划分为tokens，并分别将token映射为对应id的模块。

课程中提到的Tokenization有以下四种：
- 字符型，按单个字符（包括英文，汉字，特殊符号等，因此不等同于一个ascii码）进行编码，每个字符为一个token。压缩比低，即对于同样长度的文本，token更多。
- 字节型，按输入字符串的字节编码，每个字节为一个token。同样压缩比低。
- 词型，按词分割，每个词为一个token。词汇量大，如supercalifragilisticexpialidocious。且对于新词只能用未知代替。
- BPE(Byte-Pair Encoding)字节对，主流方法，是以上方法的兼顾，迭代地将训练数据中最常出现的相邻token对合并成一个新token来构建词汇表。

BPE字节对编码的实现：
```
def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    #Start with the list of bytes of string.
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    for i in range(num_merges):
        #Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  # @inspect counts
        #Find the most common pair.
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair
        #Merge that pair.
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices
    return BPETokenizerParams(vocab=vocab, merges=merges)
```
BPE字节对方法首先对输入字符串进行字符型编码，然后遍历所有对token对，找出最常见的一对创建一个新的token索引，并将原序列中的所有此token对替换为新的token。
经过一定次数的迭代，就构成了一个BPE字节对Tokenization。

BPE字节对这一方法让我想起了RL中，针对奖励稀疏问题提出的Skill Extraction via Byte-Pair Encoding方法[1]，这篇论文将RL的动作空间(文中例子是机器人走迷宫，动作空间是朝某方向走一步)
作为初始词表，然后进行基于BPE的特征提取，即借鉴字节对编码思想，每次找到最频繁的相邻动作对，加入词表，并同时将轨迹中所有该动作对合成一个新子词（subword），反复迭代直至目标词表大小。
这样，就压缩了原有的轨迹长度，缓解了奖励稀疏问题。

同样，合适的BPE Tokenization也能为RL微调LLM的奖励稀疏问题有贡献，即平衡：token不能太长，免得词汇表泛化程度太低；token也不能太短，免得奖励稀疏，且浪费计算资源。







[1] Yunis, D., Jung, J., Dai, F., & Walter, M. (2024). Subwords as Skills: Tokenization for Sparse-Reward Reinforcement Learning. Advances in Neural Information Processing Systems, 37, 67663-67688.

