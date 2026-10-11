# VAE — 从数学理论到算法实现

**目标**：$\Pr(x|\phi) \rightarrow$ 在参数为 $\phi$ 的神经网络下得到数据 $x$ 的概率。
我们想要最大化这个概率，采取取对数的预处理：

$$
\begin{aligned}
\log [\Pr(x|\phi)] &= \log \left[ \int \Pr(x, z | \phi) dz \right] \\
&= \log \left[ \int q(z) \frac{\Pr(x, z | \phi)}{q(z)} dz \right]
\end{aligned}
$$

### 引入 Jensen 不等式（左侧笔记内容）

由于 $g(x) = \log(x)$ 为凹函数，根据 Jensen 不等式：

$$
\begin{aligned}
g[E(y)] &\ge E[g(y)] \\
\log[E(y)] &\ge E[\log(y)] \\
\because \log \left[ \int \Pr(y) \cdot y dy \right] &\ge \int \Pr(y) \log(y) dy \\
\therefore \log \left[ \int \Pr(y) \cdot h(y) dy \right] &\ge \int \Pr(y) \log[h(y)] dy
\end{aligned}
$$

### 应用 Jensen 不等式展开（推导主过程）

$$
\begin{aligned}
&\ge \int q(z) \cdot \log \left[ \frac{\Pr(x, z | \phi)}{q(z)} \right] dz \\
&= \int q(z|\theta) \log \left[ \frac{\Pr(x, z | \phi)}{q(z|\theta)} \right] dz \\
&= \int q(z|\theta) \cdot \log \left[ \frac{\Pr(x | z, \phi) \Pr(z)}{q(z|\theta)} \right] dz \\
&= \int q(z|\theta) \cdot \log [\Pr(x | z, \phi)] dz + \int q(z|\theta) \cdot \log \left[ \frac{\Pr(z)}{q(z|\theta)} \right] dz \\
&= \int q(z|\theta) \cdot \log [\Pr(x | z, \phi)] dz - D_{KL}[q(z|\theta) \| \Pr(z)]
\end{aligned}
$$

---

### 右侧笔记：ELBO 的实际计算与转化

其中，ELBO 的第一项 $\int q(z|\theta) \cdot \log [\Pr(x | z, \phi)] dz$ 是一个不可解的积分，但这是一个**期望**。
我们可以用**蒙特卡洛估计**，更粗略地，**一次采样**作为近似：

$$ \text{第一项} \approx \log [\Pr(x | z^*, \phi)] $$
其中 $z^*$ 从近似后验 $q(z|x, \theta)$ 中采样。

**为了闭式计算**，我们将 $q$ 分布设为 Gaussian 分布，并试图找到最接近后验分布的一个。
**所选 $q(z|\theta, x)$ 是一个均值为 $\mu$，对角协方差为 $\Sigma$ 的多元正态分布。**

对于第一项（重构损失），代入高斯分布公式：

$$
\begin{aligned}
\log \frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{(x-\mu)^2}{2\sigma^2}} 
&= \log \frac{1}{\sqrt{2\pi}\sigma} + \frac{-1}{2\sigma^2} (x-\mu)^2 \\
&\iff (x-\mu)^2 \quad \text{(MSE)}
\end{aligned}
$$
*(注：二值数据对应 BCE 同理)*

### 右侧笔记：KL 散度闭式解

两个正态分布的情况下才能有闭式形式计算：

$$ D_{KL} = \frac{1}{2} \left( \text{Tr}[\Sigma] + \mu^T\mu - D_z - \log |\Sigma| \right) $$

假设 $q(z|x, \theta)$ 为对角协方差矩阵 $\Sigma$，则：

$$
\text{Tr}[\Sigma] = \sum \sigma_i^2, \quad \mu^T\mu = \sum \mu_i^2
$$

最终推导出 KL 散度的代码形式：

$$ \therefore D_{KL} = \frac{1}{2} \sum_{i=1}^{D_z} \left( \sigma_i^2 + \mu_i^2 - 1 - \log \sigma_i^2 \right) $$

---

### 左侧底部结论

**最后，最大化变最小化，调整解码器 $\phi$ 和编码器 $\theta$ 来调优。**