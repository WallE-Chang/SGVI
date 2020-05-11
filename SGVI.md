## SGVI(Stochastic Gradient Variational Inference)

> https://www.bilibili.com/video/av32047507?p=5

### Black Box Variational Inference

> https://arxiv.org/pdf/1401.0118.pdf

$\text{Let }\mathcal{L}(\phi) \triangleq \text{ELBO}=\mathbb{E}_{q(z| \phi)}[\log p(x, z)-\log q(z| \phi)] $

**NOTE:** 这个属于EM 算法的E步, 固定$\ p(x,z), \ argmax \ q(z| \phi)$

对 $\mathcal{L}$ 的参数 $\phi$ 求导得：
$$
\begin{align*} 
\nabla_{\phi} \mathcal{L} 
&=\nabla_{\phi} \mathbb{E}_{q(z| \phi)}[\log p(x, z)-\log q(z| \phi)] \tag{0}\\
&=\nabla_{\phi} \int(\log p(x, z)-\log q(z | \phi)) q(z | \phi) d z \tag{1} \\ 
&=\int \nabla_{\phi}[(\log p(x, z)-\log q(z | \phi)) q(z | \phi)] d z \tag{2}\\ 
&=\int \nabla_{\phi}[\log p(x, z)-\log q(z | \phi)] q(z | \phi) d z  +\int \nabla_{\phi} q(z | \phi)(\log p(x, z)-\log q(z | \phi)) d z \tag{3}\\
&=-\mathbb{E}_{q(z| \phi)}[\log q(z | \phi)]+\int \nabla_{\phi} q(z | \phi)(\log p(x, z)-\log q(z | \phi)) d z \tag{4}
\end{align*}
$$

- $\mathrm{(1)}$ 将期望展开

- $\mathrm{(2)}$ 交换了积分和求导的位置

- $\mathrm{(3)}$ 用了分步求导：前导后不导 + 前不导后导

- $\mathrm{(4)}$ 注意到 $\nabla_{\phi}\log p(x, z)=0$

$\begin{aligned} 
& \text{Let }\mathrm{a}=-\mathbb{E}_{q(z| \phi)}[\log q(z | \phi)], \quad
\mathrm{b}=\int \nabla_{\phi} q(z | \phi)(\log p(x, z)-\log q(z | \phi)) d z\\
&\text{So }\nabla_{\phi} \mathcal{L}=\mathrm{a}+\mathrm{b}
\end{aligned}$

首先求 $\mathrm{a}$
$$
\begin{aligned} 
&=-\mathbb{E}_{q(z| \phi)}\left[\nabla_{\phi} \log q(z | \phi)\right]\\ 
&=-\mathbb{E}_{q(z| \phi)}\left[\frac{\nabla_{\phi} q(z | \phi)}{q(z | \phi)}\right]\\
&=-\int\frac{\nabla_{\phi} q(z | \phi)}{q(z | \phi)}q(z | \phi) d z\\
&=-\int \nabla_{\phi} q(z | \phi) d z \\ 
&=-\nabla_{\phi} \int q(z | \phi) d z
=-\nabla_{\phi} 1=0 \end{aligned}
$$


接着求 $\mathrm{b}$
$$
\begin{align*} 
&=\int \nabla_{\phi}[q(z | \phi)](\log p(x, z)-\log q(z | \phi)) d z \\ 
&=\int [\nabla_{\phi} [\log q(z | \phi)](\log p(x, z)-\log q(z | \phi))]\ q(z | \phi)  d z \tag{5}\\ 
&=\mathbb{E}_{q(z| \phi)}\left[\nabla_{\phi}[\log q(z | \phi)](\log p(x, z)-\log q(z | \phi))\right]
\end{align*}
$$

- $\mathrm{(5)}$ 用到了 $\nabla_{\phi}[q(z | \phi)]=\nabla_{\phi}[\log q(z | \phi)]q(z | \phi) $

$$
\text{So,} \ \ \ \ \nabla_{\phi} \mathcal{L}=\mathrm{a}+\mathrm{b}=\mathbb{E}_{q(z| \phi)}\left[\nabla_{\phi} \log q(z | \phi)(\log p(x, z)-\log q(z | \phi))\right] \tag{6}
$$

所以我们可以使用蒙特卡洛方法，Draw $S$ samples from $q(z | \phi)$
$$
\nabla_{\phi} \mathcal{L} \approx  \frac{1}{S} \sum_{s=1}^{S} \nabla_{\phi} \log q(z[s] | \phi)(\log p(x, z[s])-\log q(z[s] | \phi))
$$
然后梯度上升更新参数
$$
\phi=\phi+\rho \nabla_{\phi}\mathcal{L}
$$
但是这里存在一个问题，定性的分析是 $\nabla_{\phi} [\log q(z[s] | \phi)]$ 在 $ q(z[s] | \phi)$ 趋近于0附近变化很剧烈，这就造成使用蒙特卡洛方法得到的 $\nabla_{\phi}$ 方差会很大，从而使得采样数量要求非常巨大，所以我们要想办法减小方差。



### The reparameterization trick

> https://arxiv.org/pdf/1312.6114.pdf

1. random variable ${\mathbf{Z}} \sim q_{\phi}(\mathbf{z} | \mathbf{x})$
2. noise variable  $\boldsymbol{\epsilon} \sim p(\boldsymbol{\epsilon})$ , 分布 $p(\boldsymbol{\epsilon})$ 是我们挑选的，所以已知
3. using a differentiable transformation ${\mathbf{Z}}=g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x})$
4. $\mathbb{E}_{q_{\phi}\left(\mathbf{z} | \mathbf{x}^{(i)}\right)}[f(\mathbf{z})]=\mathbb{E}_{p(\epsilon)}\left[f\left(g_{\phi}\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)\right] \simeq \frac{1}{L} \sum_{l=1}^{L} f\left(g_{\phi}\left(\epsilon^{(l)}, \mathbf{x}^{(i)}\right)\right) \quad$ where $\quad \epsilon^{(l)} \sim p(\epsilon)$

$$
\begin{align*} 
\mathbb{E}_{q_{\phi}\left(\mathbf{z} | \mathbf{x}^{(i)}\right)}[f(\mathbf{z})]
&= \int q_{\phi}\left(\mathbf{z} | \mathbf{x}^{(i)}\right)f(\mathbf{z}) d z \tag{7}\\
&= \int q_{\phi}\left(g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})\right)f\left(g_{\phi}\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)d (g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}))\tag{8}\\
&= \int q_{\phi}\left(g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})\right)f\left(g_{\phi}\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right) \nabla_{\epsilon}[g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})] d\boldsymbol{\epsilon} 
\tag{9}\\
&=\int p(\epsilon)\left[f\left(g_{\phi}\left(\boldsymbol{\epsilon},\mathbf{x}^{(i)}\right)\right)\right]d\epsilon
\tag{10}\\
&=\mathbb{E}_{p(\epsilon)}\left[f\left(g_{\phi}\left(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}\right)\right)\right]
\tag{11}\\
&=\mathbb{E}_{p(\epsilon)}[f({\mathbf{z}})]
\tag{12}\\
\end{align*}
$$

- $\mathrm{(7)}$ 将期望展开成积分形式

- $\mathrm{(8)}$ 采用定积分里面的换元法 ${\mathbf{Z}}=g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x})$

- $\mathrm{(9)}$ 对换元后的结果进一步计算，对 $\epsilon$ 求梯度，即
  $$
  \frac{d (g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}))}{d\boldsymbol{\epsilon}}=\nabla_{\epsilon}[g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})]\\
  d (g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}))=\nabla_{\epsilon}[g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})]d\boldsymbol{\epsilon}
  $$
  

- $\mathrm{(10)}$ 用到了 
  $$
  \begin{align*} 
  \int p(\boldsymbol{\epsilon})d\boldsymbol{\epsilon}
  &=1\\
  &=\int q_{\phi}(\mathbf{z} | \mathbf{x})dz\\
  &= \int q_{\phi}\left(g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})\right)d (g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)}))\\
  &=\int q_{\phi}\left(g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})\right)\nabla_{\epsilon}[g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})]
  d\boldsymbol{\epsilon}\\
  \end{align*}
  $$
  $\text{So, } p(\boldsymbol{\epsilon})=q_{\phi}\left(g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})\right)\nabla_{\phi}[g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x}^{(i)})]$

- $\mathrm{(11)}$ 将定积分转为期望形式

- $\mathrm{(12)}$ 使用 ${\mathbf{Z}}=g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x})$

所以我们可以得出 
$$
\mathbb{E}_{q_{\phi}\left(\mathbf{z} | \mathbf{x}^{(i)}\right)}[f(\mathbf{z})]=\mathbb{E}_{p(\epsilon)}[f({\mathbf{z}})] \tag{13}
$$
将 $\mathrm{(13)}$ 应用到 $\mathrm{(0)}$ 中
$$
\begin{align*} 
\nabla_{\phi} \mathcal{L}
&=\nabla_{\phi} \mathbb{E}_{q(z| \phi)}[\log p(x^{(i)}, z)-\log q(z| \phi)] \\
&=\nabla_{\phi} \mathbb{E}_{p(\epsilon)}[\log p(x^{(i)}, z)-\log q(z| \phi)] \tag{14}\\
&=\mathbb{E}_{p(\epsilon)}[\nabla_{\phi} (\log p(x^{(i)}, z)-\log q(z| \phi))] \tag{15}\\
&=\mathbb{E}_{p(\epsilon)}[\frac{d}{d z}(\log p(x^{(i)}, z)-\log q(z| \phi)) \frac{d z}{d \phi}] \tag{16}\\
&= \mathbb{E}_{p(\epsilon)}[(\frac{d}{d z}(\log p(x^{(i)}, z)-\log q(z| \phi)))( \frac{d}{d \phi}g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x^{(i)}}))] \tag{17}\\
\end{align*}
$$

- $\mathrm{(14)}$ 相当于令 $\mathrm{(13)}$ 中的 $f(\mathbf{z}) =\log p(x^{(i)}, z)-\log q(z| \phi) $
- $\mathrm{(15)}$ 交换梯度与期望的顺序，因为 $p(\epsilon)$ 与 $\phi$ 无关
- $\mathrm{(16)}$ 采用了链式求导 $\frac{d x}{d \phi}=\frac{d x}{d z}\frac{d z}{d \phi}$
- $\mathrm{(17)} \ \ {\mathbf{Z}}=g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x})$

这样就可以使用蒙特卡洛方法计算 $\mathrm{(17)}$

采样 $\boldsymbol{\epsilon}^{(l)} \sim p(\boldsymbol{\epsilon})$， $l=1,2,...,L$
$$
\nabla_{\phi} \mathcal{L}\approx \frac{1}{L} \sum_{l=1}^{L}[(\frac{d}{d z}(\log p(x^{(i)}, z)-\log q(z| \phi)))( \frac{d}{d \phi}g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x^{(i)}}))]\\
\text{where }\ \  {\mathbf{Z}}=g_{\phi}(\boldsymbol{\epsilon}, \mathbf{x})
$$
然后梯度上升更新参数：
$$
\phi=\phi+\rho \nabla_{\phi}\mathcal{L}
$$
