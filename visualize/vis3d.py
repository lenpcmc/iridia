from .main import *

def sphereScatter(ax, *xyz, r = 1., color = 'b'):
    xyz: np.ndarray = extend(xyz, 3)
    pts: np.ndarray = extend(sphere(r), 3)
    sps: np.ndarray = xyz + pts

    if len(np.array(color)) <= 1:
        colors = [ color for i in range(len(xyz)) ]
    else:
        colors = color
    
    for i,s in enumerate(sps):
        ax.plot_surface(*s, color = colors[i])
    ax.set_aspect("equal")
    
    return


def sphere(r, tmax = 32, pmax = 16):
    r = extend(r, 3).T
    theta = np.linspace(0., 2. * np.pi, tmax)
    phi = np.linspace(0., 1 * np.pi, pmax)
    xs = r * np.outer( np.cos(theta), np.sin(phi) ).reshape(tmax, pmax, 1)
    ys = r * np.outer( np.sin(theta), np.sin(phi) ).reshape(tmax, pmax, 1)
    zs = r * np.outer( np.ones(theta.size), np.cos(phi) ).reshape(tmax, pmax, 1)
    return np.squeeze((xs, ys, zs))

