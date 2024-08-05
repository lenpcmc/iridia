from .main import *

from numpy.typing import ArrayLike
from collections.abc import Callable


def spectNorm(spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.array(spectrum)
    return spectrum / np.sum(spectrum)


def vspect(
        x: np.ndarray,
        spectrum: np.ndarray,
        *label: str,
        invert: bool = True,
        conv: float = 33.36,
        **kwargs,
    ) -> None:
    fig,ax = spectrumPlot(invert, **kwargs)

    spectrum = np.atleast_2d(spectrum)
    for i,s in enumerate(spectrum):
        ax.plot(x * conv, s, label = label[i] if i < len(label) else None, **kwargs)

    ax.legend() if len(label) > 0 else None
    plt.show()
    return


def spectrumPlot(
        invert: bool = True,
        xlabel = r"Wavenumbers [cm$^{-1}$]",
        ylabel = r"Absorbance",
        **kwargs,
    ) -> (plt.Figure, plt.Axes):
    
    fig,ax = plt.subplots(**kwargs)

    # X-Axis
    ax.set_xlabel(xlabel)
    ax.invert_xaxis() if invert else False
    ax.tick_params('x', which = "minor", top = False)
    
    # Y-Axis
    ax.set_ylabel(ylabel)
    ax.set_yticklabels(list())
    ax.tick_params('y', which = "minor", left = False, right = False)

    return fig,ax

