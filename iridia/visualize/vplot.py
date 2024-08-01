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
    fig,ax = vdosPlot(**kwargs)
    x,y = vdosDist(freqk, width)
    ax.plot(x, y, label = label)
    ax.legend()
    
    # Directive
    plt.savefig(save) if bool(save) else plt.show()

    return


def vdosPlot(
        title: str = "",
        xlabel: str = r"$\nu$ [THz]",
        ylabel: str = r"VDoS [A.U.]",
        **kwargs,
    ) -> (plt.Figure, plt.Axes):

    """ Plot a histogram of the Vibrational Density """
    """ of States given a set of frequenices. """

    fig,ax = plt.subplots(**kwargs)
    
    ax.set_title(title)

    ax.set_xlabel(xlabel)
    ax.xaxis.set_tick_params(which = "minor", top = False)

    ax.set_ylabel(ylabel)
    ax.set_yticklabels([])
    ax.yaxis.set_tick_params(which = "minor", left = False, right = False)

    return fig,ax


def vdosDist(freqk: np.ndarray, width: int = 100) -> (np.ndarray, np.ndarray):
    """ Probability Distribution of VDOS """

    x: np.ndarray = np.linspace(0, np.max(freqk), width)
    dx: float = (np.max(x) - np.min(x)) / len(x)

    y: np.ndarray = np.histogram(freqk, bins = np.linspace(0, np.max(freqk), width + 1))[0]
    y = y / np.sum(y*dx)

    return x[1:], y[1:]

