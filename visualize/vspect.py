from .main import *

def vspect(x: np.ndarray, spectrum: np.ndarray, *label: tuple[str], invert: bool = True, conv: float = 33.36) -> None:
    fig,ax = plt.subplots()
    spectrumPlot(ax, x, spectrum, label, invert = invert, conv = conv)
    plt.legend()
    plt.show()
    return


def spectrumPlot(ax: plt.Axes, x: np.ndarray, spectrum: np.ndarray, *label: tuple[str], invert: bool = False, conv: float = 33.36, xlabel = r"Wavenumbers [cm$^{-1}$]", ylabel = r"Abs") -> None:
    # Init
    x: np.ndarray = conv * np.array(x)
    S: np.ndarray = spectNorm(spectrum).T

    # Plot
    for i,s in enumerate(S):
        lab = label[i] if i < len(label) else ""
        ax.plot(x, s, label = lab)
    #ax.plot(x, [s for s in S], label = label)
    
    ax.set_xlabel(xlabel)
    ax.invert_xaxis() if invert else False
    ax.tick_params('x', which = "minor", top = False)
    
    ax.set_ylabel(ylabel)
    ax.set_yticklabels(list())
    ax.tick_params('y', which = "minor", left = False, right = False)

    return ax


def spectNorm(spectrum: np.ndarray) -> np.ndarray:
    spectrum = np.array(spectrum)
    return spectrum / np.sum(spectrum, axis = 0, keepdims = True)
