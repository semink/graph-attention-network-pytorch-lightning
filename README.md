# Graph Attention Network (GAT)
This is a pytorch-lightning implementation of Graph Attention Network (Velickovic2018).

## Note

### Equation (3) in the original paper

Let’s define ${\bf{a}}=\begin{bmatrix}{\bf{a}}_1\\{\bf{a}}_2\end{bmatrix}$, then ${\bf{a}}^T\begin{bmatrix}{\bf{W}}h_i||{\bf{W}}h_j\end{bmatrix}=\begin{bmatrix}{\bf{a}}_1^T&{\bf{a}}_2^T\end{bmatrix}\begin{bmatrix}{\bf{W}}h_i\\{\bf{W}}h_j\end{bmatrix}={\bf{a}}_1^T{\bf{W}}h_i+{\bf{a}}_2^T{\bf{W}}h_j$. 

We can define ${\bf{b}}_1^T={\bf{a}}_1^T{\bf{W}}$ and ${\bf{b}}_2^T={\bf{a}}_2^T{\bf{W}}$, then ${\bf{a}}^T\begin{bmatrix}{\bf{W}}h_i||{\bf{W}}h_j\end{bmatrix}={\bf{b}}_1^Th_i+{\bf{b}}_2^Th_j$. Therefore, in this implementation, we define the parameter ${\bf{b}}$ instead of multiplying $\bf{a}$ and $\bf{W}$.