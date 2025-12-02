# RL算法一些思考


## 为什么是新/旧的比值
$$L^{\text{CLIP}}(\theta)=E_t[\min(r_t(\theta)\hat{A}_t,\text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t)]$$
在GRPO的Loss函数中，矫正比例$r_t(\theta)$表示为新策略$\pi_\theta$和旧策略$\pi_{old}$的比值，那么为什么非要旧策略呢？
$$E_{s, a \sim \pi_{\theta}}[G_t] = \sum_{t} \pi_{\theta}(a_t|s_t) \cdot G_t$$
在一般的强化学习问题中， 转移概率乘以优势函数足以刻画当前步的奖励了，为什么要除以旧策略？
$$E_{s, a \sim \pi_{\theta}}[G_t] \approx E_{s, a \sim \pi_{\theta_{\text{old}}}} \left[ \underbrace{\frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}}_{r_t(\theta)} \cdot G_t \right]$$

矫正比例起了这样的作用：
- 当$\pi_\theta$>$\pi_{old}$时，即在同一个s下，新策略作出a动作的概率更大时，$r_t(\theta)$ > $1$，此动作的优势函数被“放大”。
- 反之，此动作的优势函数被“缩小”。

在verl框架的训练过程中，用于采样的行为策略和更新的目标策略一般是不一致的。即，对于某个prompt
采样出多个样本，构成多个minibatch，当根据第一个minibatch完成对模型参数的更新后，对于剩下的采样，生成它们的策略就和按照它们进行更新的策略不一致了。
$$\nabla J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left[ \nabla \log \pi_{\theta}(a_i|s_i) \cdot Q^{\pi_{\theta}}(s_i, a_i) \right], \quad \text{其中 } (s_i, a_i) \sim \pi_{\text{other}}$$
也就是说，在上面的公式中，$a$,$s$都是从其他策略采样的结果，其他策略可能产生当前策略$\pi$极低概率遇到的状态、极低概率生成的动作，因此根据此$\nabla J(\theta)$进行梯度下降更新训练的必要条件是on-policy。

所以从本质上，矫正比例的作用是：
- 当$\pi_\theta$>$\pi_{old}$时，即在同一个s下，新策略比旧策略作出a动作的概率更大时，$r_t(\theta)$ > $1$，即赋予此动作的优势函数更高的“权重”，让结果近似于是由更新后的，on-policy的新策略生成的。
- 反之亦然。

还有一点个人的思考：

你原来的总奖励不是策略乘以相应动作的奖励吗？现在策略除没了，那这个公式还和原来的意义一致吗？

我个人认为，现在的所谓“策略”体现在求期望$E$上，就像刚才说的，当矫正比例大于1，该动作的权重被放大，这个“权重”除以所有“权重”的和就是另一种意义的“策略”，即在这种意义下的策略乘以相应的奖励，求和得到RL过程的总奖励。