# Vibrations


## vibrations.hess

### Hessian Matrix

The **hessian** matrix stores the partials of force between atom $i$ and all other atoms $j$.
Said more simply, the hessian stores spring constants between atoms.
With the hessian, we then apply a simple transformation to get the **dynamical**, and from that, the **Vibrational Density of States** (VDoS).

```math
	\boldsymbol{H}_{i,j} = \dfrac{\partial^2 E}{\partial r_i \partial s_j}
```

\# Note that the hessian is a 3N $\times$ 3N matrix, where the partials with respect to every atom have to be computed for $x$, $y$, and $z$. We denote this in the definition above with $r_i$ and $s_j$.

---

We've implemented two ways to find the hessian, the **numeric** approach and the **autograd** approach.
The numeric approach is the most general; it's simply an approximate derivative found by shifting the position of atom by a small factor and measuring the change in the atomic forces.
We've implemented this with forward differences: `f(x+h)`, backward differences `f(x-h)`, and central differences `avg[ f(x+h), f(x-h) ]`.

The function `iridia.vibrations.hess.hessian` implements this approach.
It takes one parameter, `atoms: Atoms`, and has three keyword parameters `h: float = 1e-5`, `method: str = "central"`, and `verbose: str | None = "Computing Hessian Matrix"`;
where `h` is the finite difference, `method` is the type of difference, and `verbose` is the message displayed along with the progress bar. 
When `verbose == None`, no message of progress bar are displayed.
The function `iridia.vibrations.hess.hessRow` is used internally to compute the force partial experienced every atom with respect to the finite displacement of a single atom.

```
def hessian(
        atoms: Atoms,
        verbose: str = "Solving Hessian Matrix",
        **kwargs,
    ) -> np.ndarray:
    """ Get the Hessian (dE/dn,dm) for a given """
    """ set of atoms. """

    # Ensure Calculator
    if (atoms.calc is None):
        atoms.calc = CHGNetCalculator()

    # Allocate and Fill
    H: np.ndarray = np.array([
        hessRow(atoms, i, **kwargs) for i in trange( 3*len(atoms), desc = f"{verbose}" )
    ])

    # Ensure Symmetry over Numeric Precision
    return -1. * (H + H.T) / 2.


def hessRow(atoms: Atoms, i: int, method: str = "central", h: float = 1e-5) -> np.ndarray:
    """ Find the ith row of the Hessian """
    """ for a given set of atoms. """

    # Shorthand
    apos: np.ndarray = atoms.positions
    
    # Finite Forward Difference
    if (method == "foward"):
        fn: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] += h

        fp: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] -= h
    
        D: np.ndarray = (fp - fn) / h

    elif (method == "backward"):
        fp: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] -= h
        
        fn: np.ndarray = atoms.get_forces().flatten()
        apos[i // 3, i % 3] += h
    
        D: np.ndarray = (fp - fn) / h
    
    else:
        apos[i // 3, i % 3] += h
```


## vibrations.autohess

### Autohessian

Our implementation remains open to any **interatomic potential** using **Python ASE**'s calculator module.
That being said, when a user potential isn't provided we deafult to using the **CHGNet** potential.
CHGNet is a machine learning interatomic potential (MLIP) trained on **ab initio**, **DFT** simulations of atomic relaxation.
CHGNet has been shown to ***Asesome Stuff here*** 

CHGNet's underlying architecture us built upon the **PyTorch** machine learning library.
Our implementation makes use of PyTorch's **Autograd** module which capable of performant, ***multivariable differentiation...***.
This allows us to **analytically** solve for the hessians of large systems within minutes.

---

For more information on our autohessian implementation, refer to https://github.com/lenpcmc/iridia/docs/autohess.md


## vibrations.vdos

### Dynamical Matrix

The **dynamical** matrix is fourier transform of our **hessian** matrix, given by:

```math
	\boldsymbol{D}_{i,j} = \dfrac{1}{\sqrt{m_i m_j}} \boldsymbol{H}_{i,j}
```

Where $m_i$ and $m_j$ denote the masses of atoms $i$ and $j$ respectively.

---

We implement this with the function `iridia.vibrations.vdos.dynamical`, which takes in a set of `atoms` along with a hessian function and its keyword arguments.
The mass coefficients of are computed by taking the outer product of atomic masses with itself $\boldsymbol{m}_i \@ {\boldsymbol{m}_j}^T$.

```
def dynamical(
        atoms: Atoms,
        verbose: str = "Solving Dynamical Matrix",
        hfunc: Callable[[Atoms], np.ndarray] = hessian,
        **kwargs,
    ) -> np.ndarray:
    """ Get the Dynamical (Hessian * 1/sqrt(m1*m2)) """
    """ for a given set of atoms. """

    mtensor: np.ndarray = np.array([ (m, m, m) for m in atoms.get_masses() ]).reshape((atoms.positions.size, 1))
    mmask: np.ndarray = 1. / np.sqrt(mtensor @ mtensor.T)
    H: np.ndarray = hfunc(atoms, **kwargs)

    return H * mmask
```


### Vibrational Density of States

The **eigenvalues** and **eigenvectors** of the dynamical matrix encode the frequencies and phonon pathways of interatomic vibrations for a set of atoms.
