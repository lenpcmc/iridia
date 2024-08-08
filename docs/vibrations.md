# Vibrations


## vibrations.hess

### Hessian Matrix

The "hessian" matrix stores the partials of force between atom $i$ and all other atoms $j$.
Said more simply, the hessian stores spring constants between atoms.
With the hessian, we then apply a simple transformation to get the "dynamical", and from that, the "Vibrational Density of States (VDoS)".

$$\mathbold{H} = \begin{bmatrix}
	
	\dfrac{\partial^2 E}{\partial x_1 \partial x_1} & 
	\dfrac{\partial^2 E}{\partial x_1 \partial y_1} & 
	\dfrac{\partial^2 E}{\partial x_1 \partial z_1} & 
	\dfrac{\partial^2 E}{\partial x_1 \partial x_2} & 
	\cdots
	\dfrac{\partial^2 E}{\partial x_1 \partial z_n} \\
	
	\dfrac{\partial^2 E}{\partial y_1 \partial x_1} & 
	\dfrac{\partial^2 E}{\partial y_1 \partial y_1} & 
	\dfrac{\partial^2 E}{\partial y_1 \partial z_1} & 
	\dfrac{\partial^2 E}{\partial y_1 \partial x_2} & 
	\cdots
	\dfrac{\partial^2 E}{\partial y_1 \partial z_n} \\
	
	\dfrac{\partial^2 E}{\partial z_1 \partial x_1} & 
	\dfrac{\partial^2 E}{\partial z_1 \partial y_1} & 
	\dfrac{\partial^2 E}{\partial z_1 \partial z_1} & 
	\dfrac{\partial^2 E}{\partial z_1 \partial x_2} & 
	\cdots
	\dfrac{\partial^2 E}{\partial z_1 \partial z_n} \\

	\dfrac{\partial^2 E}{\partial x_2 \partial x_1} & 
	\dfrac{\partial^2 E}{\partial x_2 \partial y_1} & 
	\dfrac{\partial^2 E}{\partial x_2 \partial z_1} & 
	\dfrac{\partial^2 E}{\partial x_2 \partial x_2} & 
	\cdots
	\dfrac{\partial^2 E}{\partial x_2 \partial z_n} \\

	\vdots
	\vdots
	\vdots
	\vdots
	\ddots
	\vdots \\

	\dfrac{\partial^2 E}{\partial z_n \partial x_1} & 
	\dfrac{\partial^2 E}{\partial z_n \partial y_1} & 
	\dfrac{\partial^2 E}{\partial z_n \partial z_1} & 
	\dfrac{\partial^2 E}{\partial z_n \partial x_2} & 
	\cdots
	\dfrac{\partial^2 E}{\partial z_n \partial z_n} \\
	
\end{bmatrix}$$


We've implemented two ways to find the hessian, the "numeric" approach and the "autograd" approach.
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


## vibrations.vdos

### Dynamical Matrix

The dynamical matrix is a transformed hessian that includes the masses of each atom. 

