from .main import *
from matplotlib.animation import FuncAnimation as animate_plot

from ..vibrations import *
from .actoms import *
from .vis3d import *

def main():
    atoms = ase_read(f"{ir_root}/resources/atoms/wollastonite.cif")
    dyn = np.load(f"{ir_root}/resources/arrays/wollastonite.npy")
    freqk, vibrations = vdosDyn(dyn)
    print(f"{vibrations.shape = }")
    vibration = vibrations[50]
    animateVibration(atoms, vibration)
    return


def animateAtoms(atoms: Atoms, path: np.ndarray, fps: float = 30., frames: int = 60,):
    
    # Get Bounds
    bounds: np.ndarray = np.array([ np.min(atoms.cell, axis = 1), np.max(atoms.cell, axis = 1) ])
    pad: np.ndarray = 0.2 * (bounds[1] - bounds[0])
    pbounds: np.ndarray = np.array([bounds[0] - pad, bounds[1] + pad]).T

    # Matplotlib Setup
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    plt.axis( pbounds.flatten() )
    
    colors = atomColors(atoms)
    sizes = atomSizes(atoms)

    def animate(frame) -> plt.Axes:
        while bool(ax.collections):
            ax.collections[0].remove()

        ppos: np.ndarray = extend( atoms.positions, 3 ) + path[...,:frame + 1]

        # Atoms
        sphereScatter(ax, ppos[...,frame], r = sizes, color = colors )
        plt.show()
        exit()

        # Atom Trails
        for i,a in enumerate(atoms):
            print(sizes[i])
            ax.plot( *ppos[i], color = colors[i] )

        return ax

    anim = animate_plot(fig, animate, frames = frames)
    plt.show()
    return anim


def animateVibration(
        atoms: Atoms,
        vibration: np.ndarray,
        fps: float = 30.,
        frames: int = 61,
        scale: float = 10.,
    ) -> None:

    path: np.ndarray = extend( vibration * scale, 3 ) * np.sin(2. * np.pi * np.arange(frames) / fps)
    print(f"{path.shape = }")
    animateAtoms(atoms, path, fps, frames)
    return


main()
