# Eigenvector Preferential Attachment Simulator 

## Eigen Solvers
### Power Iteration 
Recall in a bipartite graph, we have the following proposition: 

:::{prf:definition}
A graph is *bipartite* iff its spectrum is symmetric around $0$. In particular the eigenvalue with largest modulus comes in pairs $\lambda_{max}, -\lambda_{max}$
:::

As a result, power iteration would oscillate between the two. Two counter this, we add a shift:
$$
A\gets A + sI 
$$
where $s>0$. Now the eigenvalues differ in modulus: the positive one has modulus $\lambda_{max}+s$ while the negative one has $\lmabda_{max}-s$. 
Note if $s\to 0$, then their ratio goes to 1 hence convergence rate wouldbe slow. However, if we take $s\to infty$, then their ratio also goes to $1$ jeoperdizing the convergence rate too. 