# iridia
A Universal Model ofr Predicing IR Absorbance Spectra

**I**nfra**R**ed **I**nteratomic **DI**pole **A**nalysis or **iridia** is an implementation of the proposed "Universal Model for Predicting IR Absorbance Spectra" described in [Lennon et al](https://github.com/lenpcmc/iridia). The implementation relies on [Python ASE](https://wiki.fysik.dtu.dk/ase/) and functions with any interatomic potential and charge equilibrium model appropriately.

At the center of **iridia** are two core concepts: **V**ibrational **D**ensity **o**f **S**tates (**VDoS**) and Charqe Equilibrium (**QEq**), described below. We use these methods to predict the IR absorbance cross section (eq. 1) of any given system of atoms.

$$
\begin{equation}
  \sigma_k (\omega) = \dfrac{\pi}{3 \epsilon_0 c} \left( \dfrac{\partial \mu}{\partial Q_k} \right)^2 \dfrac{\gamma_k}{\left( \omega_k - \omega \right)^2 - {\gamma_k}^2}
\end{equation}
$$
