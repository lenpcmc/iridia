from .main import *

def vspect(
        x: np.ndarray,
        spectrum: np.ndarray,
        *label: str,
        invert: bool = True,
        conv: float = 33.36,
        **kwargs,
    ) -> None:
    fig,ax = plt.subplots()
    spectrumPlot(ax, x, spectrum, label, invert = invert, conv = conv, **kwargs)
    plt.legend()
    plt.show()
    return


def spectrumPlot(
        ax: plt.Axes,
        x: np.ndarray,
        spectrum: np.ndarray,
        *label: str,
        invert: bool = False,
        conv: float = 33.36,
        xlabel = r"Wavenumbers [cm$^{-1}$]",
        ylabel = r"Abs",
        **kwargs,
    ) -> None:
    
    # Init
    x: np.ndarray = np.squeeze( conv * np.array(x) )
    S: np.ndarray = np.squeeze( spectNorm(spectrum).T )
    S = np.atleast_2d(S)

    # Plot
    for i,s in enumerate(S):
        lab = label[i] if i < len(label) else ""
        ax.plot(x, s, label = lab, linewidth = 0.5)
    
    # X-Axis
    ax.set_xlabel(xlabel)
    ax.invert_xaxis() if invert else False
    ax.tick_params('x', which = "minor", top = False)
    
    # Y-Axis
    ax.set_ylabel(ylabel)
    ax.set_yticklabels(list())
    ax.tick_params('y', which = "minor", left = False, right = False)

    return


def spectNorm(spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.array(spectrum)
    return spectrum / np.sum(spectrum, keepdims = True)
