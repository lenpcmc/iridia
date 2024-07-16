from .main import *

def vplot(
        freqk: np.ndarray,
        width: int = 100,
        label: str = None,
        save: str = False,
        **kwargs,
    ) -> None:
    """ Plot the Vibrational Density of States """
    """ given a set of frequencies. """

    # Init
    fig,ax = plt.subplots()
    vdosPlot(ax, freqk, width, label, **kwargs)
    
    # Directive
    plt.savefig(save) if bool(save) else plt.show()

    return


def vdosPlot(
        ax: plt.Axes,
        freqk: np.ndarray,
        width: int = 100,
        label: str = None,
        title: str = "",
        xlabel: str = r"$\nu$ [THz]",
        ylabel: str = r"VDOS [A.U.]",
        **kwargs,
    ) -> None:

    """ Plot a histogram of the Vibrational Density """
    """ of States given a set of frequenices. """
    
    ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.xaxis.set_tick_params(which = "minor", bottom = False)

    ax.set_ylabel(ylabel)
    ax.set_yticklabels([])
    ax.yaxis.set_tick_params(which = "minor", bottom = False)

    x,y = vdosDist(freqk, width)
    y *= len(freqk)

    ax.plot(x[1:], y[1:], label = label)
    ax.legend() if bool(label) else None

    return


def vdosDist(freqk: np.ndarray, width: int = 100) -> (np.ndarray, np.ndarray):
    """ Probability Distribution of VDOS """
    x: np.ndarray = np.linspace(0, np.max(freqk), width)
    y: np.ndarray = np.histogram(freqk, bins = np.linspace(0, np.max(freqk), width + 1))[0] / len(freqk)
    return x,y

