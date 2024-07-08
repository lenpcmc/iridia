from main import *
#from .main import *


def main():
    return


def vplot(freqk: np.ndarray, width: int = 100, title: str = "") -> None:
    """ Plot the Vibrational Density of States """
    fig,ax = plt.subplots()
    vdosPlot(ax, freqk, width)
    ax.set_title(title)

    ax.set_xlabel(r"$\nu$ [THz]")
    ax.xaxis.set_tick_params(which = "minor", bottom = False)
    ax.set_ylabel(r"VDOS [A.U.]")
    ax.set_yticklabels([])
    ax.yaxis.set_tick_params(which = "minor", bottom = False)

    plt.savefig(f"{im_root}/{title}.png")
    #plt.show()
    #plt.close()
    return


def vdosPlot(ax: plt.Axes, freqk: np.ndarray, width: int = 100) -> plt.Axes:
    x = np.linspace(0, np.max(freqk), width)
    y = np.histogram(d, bins = np.linspace(0, np.max(freqk), width + 1))[0]

    ax.plot(x, y)
    return ax



if __name__ == "__main__":
    main()
