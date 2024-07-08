from .main import *


def vplot(freqk: np.ndarray, width: int = 100, title: str = "", save: str = False) -> None:
    """ Plot the Vibrational Density of States """
    """ given a set of frequencies. """

    # Init
    fig,ax = plt.subplots()
    vdosPlot(ax, freqk, width)
    
    # Format Plot
    ax.set_title(title)

    ax.set_xlabel(r"$\nu$ [THz]")
    ax.xaxis.set_tick_params(which = "minor", bottom = False)

    ax.set_ylabel(r"VDOS [A.U.]")
    ax.set_yticklabels([])
    ax.yaxis.set_tick_params(which = "minor", bottom = False)

    # Directive
    if bool(save):
        plt.savefig(save)
    else:
        plt.show()

    return


def vdosPlot(ax: plt.Axes, freqk: np.ndarray, width: int = 100) -> None:
    """ Plot a histogram of the Vibrational Density """
    """ of States given a set of frequenices. """
    x,y = vdosDist(freqk, width)
    y *= len(freqk)
    ax.plot(x,y)
    return


def vdosDist(freqk: np.ndarray, width: int = 100) -> (np.ndarray, np.ndarray):
    """ Probability Distribution of VDOS """
    x: np.ndarray = np.linspace(0, np.max(freqk), width)
    y: np.ndarray = np.histogram(freqk, bins = np.linspace(0, np.max(freqk), width + 1))[0] / len(freqk)
    return x,y
