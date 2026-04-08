# Introduction/Proposal

## Preferential Attachment 

:::{prf:definition} Preferential Attachment Model
**Preferential attachment model** is a class of random graph models that describes the growth of a network. The models start with an initial graph $G_0$ and gradually add nodes to it, each time forming $m$ edges to the exsiting node with probability weighted by a preference function $p$. Formally, these models are characterized by $(m, p)$ where $m \in \mathbb{N}$ is the paramter that descirbes the number of edges formed by the new node with existing nodes and $$p:G_t \to [0,1]^{n_t}$$[^nt] is the preference function that, given the graph at $G_t$, gives the probability of each $v\in V_t$ being attached by the new node.
:::

:::{figure} image/preferential_attachment.png
Preferential Attachment Model. The adjacency matrix $A_t$ is a hermitian $\{0,1\}^{n_t\times n_t}$
:::

Classic preferential models include the [*Barabási–Albert model*](https://en.wikipedia.org/wiki/Barab%C3%A1si%E2%80%93Albert_model) which uses degree centrality as a preference function. This model is quite well-studied. A key property is that: 

:::{prf:proposition} Degree of BA model follows power-law
The degree distribution $P(k)$ of a model is defined as the fraction of nodes to have degree $k$:
$$
P(k) = \frac{n_k}{k}
$$
the BA model is *scale-free* and follows the power law:
$$
P_{\mathrm{BA}}(k) = k^{-3}
$$
:::

*Adami et. al.* proposed a new preferential attachment model that aims to grow the network by a global centrality rather than a local one like *degree centrality*. This is relevant in the sense when the new incoming node adjoins, the gain in "popularity" of the immediate node(s) can propagate further. 

:::{prf:definition} Eigenvector Preferential Attachment Model
Eigenvector Preferential Attachment Models(**EPAM**), in particular, are preferential attachment models with 
preference function defined by:
$$p(G_t)=\begin{pmatrix}
\frac{v_1}{\sum_{i=1}^{n_t} v_i}\\
\frac{v_2}{\sum_{i=1}^{n_t} v_i}\\
\vdots \\
\frac{v_{n_t}}{\sum_{i=1}^{n_t} v_i}
\end{pmatrix}$$
where $\mathbf{v}$ is the **perron-frobenius** eigenvector, which is the eigenvector corresponding to the greatest eigenvalue. The positivity of probability is guaranteed by the *Perron-Frobenius Theorem*. 
:::

## Problem Statement
The goal of the attention network $f_{\phi}$ is to  deterministically parametrize the mapping $$f :\mathbb{G}\times \mathbb{N}\to [0,1]^{n_N\times n_N}$$
which, given a inital graph $G_0$ and a desired number of steps, gives the expectation of the graph $f(G_0, N)$. The expectation is a matrix $(a_{ij})$ where $$a_{ij}=\mathbb{P}\{(i,j)\in E_N\}$$

:::{figure} image/model_overview.png

:::

:::{note} A Possible Connection To CV?
The structure of this problem, after being drawn out like above, sounds similar to a traditional CV task of given a part of a image, then complete the image based on the semantics of the given fragment. However, since we have rigours mathematical constraints that we want to impose it doesn't seem too relevant to do transfer learning leveraging existing ViTs.
:::


## Motivation: Why Attention? 
:::{note} Not GAN?
In the first meeting, Farouk suggested to stick with simpler models like GAN first before trying something like attention. However after pondering with the idea a little bit I think attention might bring more interpretability hence allows us to draw more mathematically interesting observation(see [motivation 2](motiv2)).
:::
### Motivation 1. Proposing Better Degree Distribution By Data Replication
We know quite little about EPAM, in particular, the exact form of the degree distribution is unkown. Although it not hard to approximate the degree distribution numerically by averaging trajectories at large time-steps, for instance, in a previous work by *Arha Gatram, Srujan Roplekar, Haoyu Tang, Shuwei Zhang*, the authors obtained a numerical approximation:
:::{figure} image/degree_dist_m1.png
:label: raw_deg_dist
Numerically Approximated Degree Distribution For $m=1$
:::

Which is good enough in general, however, it can be seen that at a few place the numerical approximation does not converge. This is worsen when $m$ becomes larger. If we have $\mathbb{E}[G_t]$, we can approximate degree distributions using two elementary approaches:

#### Approximating $\deg(v_i)$ as Possion Bionmial Random Variable
With the mapping $f_{phi}$ described above, note if we let:
$$
X_{ij} = \begin{cases}
1 & \text{if $(i,j)\in E_N$}\\
0 & \text{else}
\end{cases}
$$
Then $\deg(v)=\sum_{j\in \{1,...,n_t\}} X_{ij}$. We make an assumption:

:::{attention} Independence Assumption
Let $\{X_{ij}\}_{j=1}^{n_t}$ be **independent**, not identical bernoulli distributions. Note in reality they are negative-correlated, however, we can assume the correlation to be weak and see what the result looks like.
:::

With this assumption, $\deg(v)$ becomes a possion binomial random variable, when $n$ is large this can be approximated by a simple poisson random distribution with $\lambda = \sum_{j\in \{1,...,n_t\}}\mathbb{E}{[X_{ij}]}$ which is just the $i^{th}$ row sum of $\mathbb{E}[G_t]$. The resulting degree distribution is hence just:
$$
P(k)=\frac{1}{n_t}\sum_{i=1}^{n_t} \mathrm{Poisson}(\mathrm{rowsum}(\mathbb{E}[G_t],\mathrm{row=i}))
$$

which is a continuous, smooth approximation of the true degree distribution. 

:::{warning} Poisson Variance

The above approach seems to be problemetic in the sense that a for $X\sim \mathrm{Poisson}(\lambda), \mathrm{var}(X)=\lambda$. As a result, a peak towards the right is necessarily short and fat. However, in the [naive histogram](raw_deg_dist) we can see a sharp, tall peak exists towards the larger degree end. This strongly suggest a linear combination of Poisson variable might not capature degree distribution of EPAM correctly. Likely due to the failure of the independence assumption.
:::

#### Continuous Degree Distribution 
Another method is to sort the nodes by degrees to obtain:
$$
k_1\leq k_2\leq...\leq k_{n_t}
$$
We can view this as a discretization $(k_i, i)_i$ of a mapping from degree to rank $1,...,n_t$, hence we can interpolate to obtain a function $f: k_i \to [1,...,n_t]$. 
Now:
$$
P(k) &= \frac{d}{dk}P\{K\leq k\} \\
&=\frac{d}{dk}P\{i \geq f(k)\} \\
&= \frac{d}{dk}\left(1- \frac{f(k)}{n_t}\right)
$$
where the last equality follows from that the rank is uniformly distributed. 

---

#### Computational Complexity of Numerically Approximating $\mathbb{E}[G_t]$ and Biasedness of Monte Carlo 
At this point, attention is irrelevant since one can argue averaging different trajectories also gets $\mathbb{E}[G_t]$ too(This is infact how [](raw_deg_dist) in computed).  

However, this might be very biased as the whole space of possible graphs is of size $(T+1)!$, however, we are taking only $T$ trajectories meaning that all $T$ trajectories might be compressed near a single point in the true distribution.(**It would be nice if I can argue this mathematically, need to figure out**)

Another approach is to brute-force enumerate the whole space at stage $T$. However, this is prohibitively expensive as the number of eigenvalue problems we need to solve is $\mathcal{O}((T+1)!)$. This approach allows us to get the ground truth of $\mathbb{E}[G_t]$ as well as the degree distribution by weighting over different trajectories. For instance, for $T=8$:

:::{figure} image/network_evolution.mp4

:::

Using the two approximations I mentioned before, we can infer the the degree distribution solely from $\mathbb{E}[G_t]$:

:::{figure} image/degree_distribution_approximation
:::

It can be seen that the approximation isn't exactly accurate, however this might be due to that $N$
is small. As $N\to \infty$, the independence between different edges would increase, for instance. 


#### Data Replication
We might be able to get more information out of simulation by using the following technique:

:::{warning}
Farouk suggested this would make the model become biased, however, it's also possible I didn't make myself clear since I wanted to describe the intuition that, the model learns how we evovle from $G_0$, hence what exactly does $G_0$ look like does not matter if the model is behaving as expected. 
:::
:::{note} Data Replication

For each trajectory, we produce data samples by partitioning the trajecotry into **disjoint sections** with random length 

```{code-cell}
start = 0
While(start < n_t)
    len = randomInt(1,n_t)
    new_sample = sample(trajectory[start, start+len], len)
    samples.add(new_sample)
    start = start+len
EndWhile
```

:::

:::{figure} image/data_replication.png
Data Replication
:::

(motiv2)=
### Motivation 2. Using Attention Matrix To Interpret Model Dynamics 
Note the perron-frobenius eigenvector $\mathbf{v}$ trivially satisfy:
$$
(A-\lambda I)v=0
$$
Expand the matrix-vector multiplication:
$$
 v_i = \frac{1}{\lambda}\sum_{j\in K(i)}v_j
$$
where $K(i) \subseteq \mathbb{N}_{\leq n_t}$ denotes the index of neighbors of $v_i$. This shows

## Model Architecture
This is still not exactly clear to me. A reasonable formulation is standard **encoder-decoder** attention where each row/column vector(including the 0 vectors that represent the future nodes to be added to the network) is a token. Some positional embedding is required(RoPE?). 

The decoder would be responsible of producing a expectation $[0,1]^{n_t\times n_t}$. A simple squared loss like:
$$
\mathcal L = \sum_{k=1}^{N}\sum_{i,j} (p_{ij}-A_{ij}^{(k)})^2
$$
where $N$ is number of samples can be used? Need to sort out

[^nt]: $n_t$ is the number of nodes at time step $t$
