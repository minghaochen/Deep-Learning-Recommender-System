FM 的形式
$$
y(x)=w_0+\sum_{i=1}^{n}w_ix_i+\sum_{i=1}^n \sum_{j=i+1}^n<v_i,v_j>x_ix_j
$$
其中前两项为基本的线性回归，后一项为特征的交互项。$x\in R^n$ 为 n 为特征，$w$ 为线性回归参数，$V\in R^{n\times k}$ 为交互矩阵，$k$ 为超参数，相对于对输入 $x$ 的每一维定义了一个隐向量，$<v_i,v_j>$ 表示向量的内积。

可以化简交互项为
$$
\sum_{i=1}^n \sum_{j=i+1}^n<v_i,v_j>x_ix_j
\\=\frac{1}{2}\sum_{f=1}^k ((\sum_{i=1}^n v_{i,f}x_i)^2-\sum_{i=1}^nv_{i,f}^2x_i^2)
\\=\frac{1}{2}\sum [(x\cdot V)^2-(x^2\cdot V^2)]
$$
 中括号中就是矩阵乘积的平方减去矩阵平方的乘积