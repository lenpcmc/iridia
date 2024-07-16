from .main import *

pi: float = np.pi
ep0: float = 8.854e-12
c: float = 299_792_458

def absorbance(w: float, k: float, ddm: float, y: float = 0.25) -> float:
    # Init
    w: np.ndarray = extend(w, 3).swapaxes(0,1)
    k: np.ndarray = extend(k, 3)
    ddm: np.ndarray = extend(ddm, 3)
    y: np.ndarray = extend(y, 3)

    ddm2: np.ndarray = np.sum(ddm**2, axis = 1, keepdims = True)

    # Calc
    absCoeff: float = pi / ( 3. * ep0 * c )
    broadening: np.ndarray = y / ((k - w)**2 + y**2)

    #return absCoeff * ddm2 * broadening
    return np.squeeze(np.sum( absCoeff * ddm2 * broadening, axis = 0 ))

