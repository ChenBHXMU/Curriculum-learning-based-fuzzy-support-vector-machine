# Curriculum-learning-based-fuzzy-support-vector-machine
Chen B, Gao Y, Liu J, et al. Curriculum learning-based fuzzy support vector machine[J]. IEEE Transactions on Fuzzy Systems, 2023.

**Parameter settings:**
item =20; 
isOneVone = 1;
isCluster = 1;
Set the type to 1, 2, 3, and 4, the fuzzy membership corresponds to $\frac{1}{1+\xi_i}$, $1-\frac{\xi_{i}}{\max \left(\mathbf{\xi}\right)+\Delta}$, $\frac{2}{1+e^{\beta \xi_i}}$, and $e^{-\frac{\left\|\xi_i-\mu\right\|^{2}}{2 \sigma^{2}}}$.

When item is set to 1, isCluster is set to 0, and type is set to 1, 2, 3, and 4, the fuzzy membership corresponds to $\frac{1}{1+|u_i|}$, $1-\frac{u_{i}}{\max \left(|\mathbf{u}|\right)+\Delta}$, $\frac{2}{1+e^{\beta |u_{i}|}}$, and $e^{-\frac{\left\|u_{i}-\mu\right\|^{2}}{2 \sigma^{2}}}$.

The relevant papers are as follows:

1.Wu Y, Liu Y. Adaptively weighted large margin classifiers[J]. Journal of Computational and Graphical Statistics, 2013, 22(2): 416-432.


2.Batuwita R, Palade V. FSVM-CIL: fuzzy support vector machines for class imbalance learning[J]. IEEE Transactions on Fuzzy Systems, 2010, 18(3): 558-571.

3.Liu J. Fuzzy support vector machine for imbalanced data with borderline noise[J]. Fuzzy Sets and Systems, 2021, 413: 64-73.
