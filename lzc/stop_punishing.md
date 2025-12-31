# Stop punishing maybe better

Author: Zichang Lan

### RL Modeling of LLM Fine-Tuning (Background)

Proximal Policy Optimization (PPO) and Group Relatively Policy Optimization (GRPO) are two of the most popular reinforcement learning algorithms used in fine-tuning large language models (LLMs). The key difference between them, from my perspective, is the reward model they use. Omitting the difference of the reward model, the goal function of both of them could be written as:

$$L^{\text{CLIP}}(\theta)=\frac{1}{n}\sum_{t=1}^{n}[\min(r_t(\theta)\hat{A}_t,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)]
\quad(1)$$

where $clip$ means to limit the value of $r_t(\theta)$ within the range of $[1-\epsilon,1+\epsilon]$, $\hat{A}_t$ is the advantage function at time step $t$, defined as:

$$\hat{A}_i = r_i - \frac{1}{K} \sum_{j=1}^{K} r_j
\quad(2)$$

where $r$ is the reward given by the reward model. And $r_t(\theta)$ is the probability ratio defined as:$$r_t(\theta) = \frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}  
\quad(3)$$

But here comes the question: How is equation(1) formulated? Note that the general estimate of total reward should be:$$E_{s, a \sim \pi_{\theta}}[G_t] = \sum_{t} \pi_{\theta}(a_t|s_t) \cdot G_t
\quad(4)$$

The total reward should be the accumulation of the policy multiples the corresponding reward at every step. Why is there a "ratio" between the $\pi_\theta$ and $\pi_{old}$, instead of simply using $\pi_\theta$?

The answer lies in the importance sampling for off-policy policy evaluation talked in the lecture. Actually, the formulation equation (4) should be written as:

$$E_{s, a \sim \pi_{\theta}}[G_t] \approx E_{s, a \sim \pi_{\theta_{\text{old}}}} \left[ {\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}} \cdot G_t \right]
\quad(5)$$

That is because the training process is actually off-policy, although ideally it should be on-policy. The update of the parameter of model is processed by minibatch so the policy the model rollouts is different from the policy the model updates.

Now we are using the state-action pairs sampled from the old policy ($\pi_{\theta_{\text{old}}}$) to estimate the expected reward of the new policy ($\pi_{\theta}$), according to the importance sampling as below:

$$V^{\pi_{1}}(s)\approx\frac{1}{n}\sum_{j=1}^{n}\frac{\mathbb{P}(h_{j}|\pi_{1},s)}{\mathbb{P}(h_{j}|\pi_{2},s)}G(h_{j}) \\ =\frac{1}{n}\sum_{j=1}^{n}\frac{\prod_{t=1}^{L_{j}-1}\pi_{1}(a_{j,t}|s_{j,t})\mathbb{P}(r_{j,t}|s_{j,t},a_{j,t})\mathbb{P}(s_{j,t+1}|s_{j,t},a_{j,t})}{\prod_{t=1}^{L_{j}-1}\pi_{2}(a_{j,t}|s_{j,t})\mathbb{P}(r_{j,t}|s_{j,t},a_{j,t})\mathbb{P}(s_{j,t+1}|s_{j,t},a_{j,t})} \\ =\frac{1}{n}\sum_{j=1}^{n}G(h_{j})\prod_{t=1}^{L_{j}-1}\frac{\pi_{1}(a_{j,t}|s_{j,t})}{\pi_{2}(a_{j,t}|s_{j,t})}.
\quad(6)$$

Compare equation (5) with the above equation (6), we could see that there is a remarkable difference: in equation (5), 
there is just $\frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$, but in the equation (6) there is $\prod_{t=1}^{L_{j}-1}\frac{\pi_{1}(a_{j,t}|s_{j,t})}{\pi_{2}(a_{j,t}|s_{j,t})}$. How could we say they are the same?

The reason is derived from the modeling of the LLM fine-tuning process:

As we know, the rollout process of an LLM is actually generating a sequence of tokens step by step, until reaching the end token, and it can be represented by this:

$$P(Y \mid X) = \prod_{j=1}^{m} P(y_j \mid X, Y_{1:j-1})
\quad(7)$$

which means the probability of generating a new token is affected by the prompt and all the previously generated tokens, so it is not a Markov Decision Process (MDP).

In this process, it is wrong to write the state transition probability $\mathbb{P}(s_{t+1}|s_t,a_t)$, because the next state is affected by all the previous states and actions, not only the current state and action, so all the reasoning in (6) doesn't stand.

In this case, we need to model this process differently. Though when generating each token it is not an MDP, we could take the whole rollout trajectory corresponding to a prompt as an "action", take the prompt as the initial "state". Then the process could be modeled as a one-step MDP.

Of course, in this MDP there is only one policy in the process, so $\frac{\pi_{\theta}(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ and $\prod_{t=1}^{L_{j}-1}\frac{\pi_{1}(a_{j,t}|s_{j,t})}{\pi_{2}(a_{j,t}|s_{j,t})}$ are obviously equivalent.

One last thing is the difference between the reward $G$ in equation (4) and the advantage $A$ in equation (2). The reward in PPO or GRPO is represented as advantage rather than the reward, because in this way the reward is normalized across different samples, which could stabilize the training process. 

### Essence of SFT and GRPO 

Using the one-step MDP above, let's talk about SFT and GRPO. The loss of SFT is:

$$\mathcal{L}_{SFT}(\theta) = -\sum_{t=1}^{n} P_{\theta}(y_{i} \mid x_{i})
\quad(8)$$

where n represents the number of training data, $x_{i}$ denotes the i-th prompt and $y_{i}$ denotes the labeled "right" trajectory of $x_{i}$.

Simplify the loss of GRPO, we have:

$$\mathcal{L}(\theta)=-\frac{1}{n}\sum_{t=1}^{n}(\pi_{\theta}(a_{t} \mid s_{t})\hat{A}_t)
\quad(9)$$

here, since the action $a_{t}$ is defined as the trajectory and the state s is defined as the prompt, we can know that 
$\pi_{\theta}(a_{t} \mid s_{t})$ is just $P_{\theta}(y_{i} \mid x_{i})$ in equation (8). So the loss function of GRPO can be written as:

$$\mathcal{L}_{GRPO}(\theta)=-\frac{1}{n}\sum_{t=1}^{n}P_{\theta}(y_{i} \mid x_{i})\hat{A}_t
\quad(10)$$

Compare equation (8) and equation (10) we can see that the loss of GRPO is the weighted average of $P_{\theta}(y_{i} \mid x_{i})$ (the weight is $\hat{A}_{t}$) while that of SFT is just the average. Actually in GRPO, the loss of the samples with $\hat{A}_t>0$ would decrease while the loss of whose $\hat{A}_t<0$ would increase.

**So personally speaking, I think SFT is "rewarding" (maximize the possibility) all the (prompt,labeled trajectory) samples while GRPO is "rewarding" the "good" (whose $\hat{A}_t>0$) samples and "punishing" the "bad" ($\hat{A}_t<0$) samples.**

## Stop punishing maybe better (What I really what to say)

Take the math reasoning as an example: Suppose there is a math problem needs 5 steps to complete, and now the model has rollout two trajectories. One is totally right in all 5 steps and has the right answer, the other is right in the beginning 4 steps but wrong in the last step so its answer is wrong.

According to GRPO, the first trajectory will be rewarded and the second trajectory will be punished. **But here is the problem: GRPO is punishing the whole trajectory of the second one so its beginning 4 right steps would also be punished, which is actually harmful to the model!**

***So, can we stop punishing and only "reward" the "good" trajectories when using GRPO?***

The KEY idea (GRPO without punishment, GRPOWP) is:

$$ \hat{A}_{t} = max(0, r_i - \frac{1}{K} \sum_{j=1}^{K} r_j)
$$

Could it work?

## Experiment

I trained Qwen2.5-1.5B-Instruct model with GSM8k dataset, before which I randomly selected 100 pieces of data as test set.
Compare the test accuracy between base model, GRPO trained model and GRPOWP (without punishment) trained model, the results are in the table below:

<table>
  <thead>
    <tr>
      <th></th>
      <th>5 epoch</th>
      <th>30 epoch</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Base (Qwen2.5-1.5B-Instruct)</td>
      <td>6.33%</td>
      <td>6.33%</td>
    </tr>
    <tr>
      <td>GRPO</td>
      <td>71.7%</td>
      <td>62.3%</td>
    </tr>
    <tr>
      <td>GRPOWP</td>
      <td>45.3%</td>
      <td>55.1%</td>
    </tr>
  </tbody>
</table>

<style>
table {
  border-collapse: collapse;
  width: 100%;
  margin: 1em 0;
}
table thead th {
  border-top: 2px solid #000;
  border-bottom: 1px solid #000;
  padding: 8px;
  text-align: center;
}
table tbody td {
  padding: 8px;
  text-align: center;
}
table tbody tr:last-child td {
  border-bottom: 2px solid #000;
}
</style>

Unfortunately, from the experiment I can not prove the GRPOWP is superior to GRPO. Maybe I should design another experiment to prove how "punishment" would affect the training process, or maybe there is something wrong in my understanding about RL in LLM. I would try to figure it out later.










