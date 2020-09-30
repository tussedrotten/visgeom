import numpy as np


def generate_ellipsoid(n=20, **kwargs):
    """Generate points on an ellipsoid in 3D.

    Keyword Arguments
        * *pose* -- Pose (R, t) of the ellipsoid in the global frame, where R is a 3x3 rotation matrix and t is a 3D column vector, default (np.identity(3), np.zeros((3, 1)))
        * *scale* -- 3D vector with scale factors along each dimension, default np.ones((3, 1))

    :param n: The number of samples on the ellipsoid for each polar coordinate, default 20
    :param kwargs: See above

    :return: A tuple (x, y, z) which can be used to plot the surface.
    """
    pose = kwargs.get('pose', (np.identity(3), np.zeros((3, 1))))
    scale = kwargs.get('scale', np.ones((3, 1)))

    u = np.linspace(0, 2 * np.pi, n + 1)
    v = np.linspace(0, np.pi, n + 1)
    x = -np.outer(np.cos(u), np.sin(v)).T
    y = -np.outer(np.sin(u), np.sin(v)).T
    z = -np.outer(np.ones_like(u), np.cos(v)).T

    R, t = pose
    sphere_points = (R * scale) @ np.vstack((x.flatten(), y.flatten(), z.flatten())) + t

    return np.reshape(sphere_points[0, :], x.shape), \
           np.reshape(sphere_points[1, :], y.shape), \
           np.reshape(sphere_points[2, :], z.shape)


def generate_box(**kwargs):
    """Generate the 8 corners of a box in 3D.

    Keyword Arguments
        * *pose* -- Pose (R, t) of the box in the global frame, where R is a 3x3 rotation matrix and t is a 3D column vector, default (np.identity(3), np.zeros((3, 1)))
        * *scale* -- Scale factor, default 1.0

    :param kwargs: See above

    :return: A 3x8 matrix, with each column representing a box corner.
    """
    pose = kwargs.get('pose', (np.identity(3), np.zeros((3, 1))))
    scale = kwargs.get('scale', 1)
    box_points = np.array([[-1, 1, 1, -1, -1, 1, 1, -1],
                           [-1, -1, 1, 1, -1, -1, 1, 1],
                           [-1, -1, -1, -1, 1, 1, 1, 1]])

    R, t = pose
    S = scale * np.identity(3)
    return R @ S @ box_points + t


def plot_as_box(ax, points, **kwargs):
    """Plots a 3x8 matrix as a connected box, where each column is a corner.

    Keyword Arguments
        * *alpha* -- Alpha value (transparency), default 1
        * *edgecolor* -- Color of box edges, default 'k''
        * *pointcolor* -- Color of box corner points, default 'k'

    :param ax: Current axes
    :param points: Corner points in box (3x8 matrix)
    :param kwargs: See above

    :return: List of artists.
    """
    alpha = kwargs.get('alpha', 1)
    edgecolor = kwargs.get('edgecolor', 'k')
    pointcolor = kwargs.get('pointcolor', 'k')

    artists = ax.plot(points[0, :], points[1, :], points[2, :], pointcolor + '.', alpha=alpha)

    for start in (0, 4):
        inds = np.concatenate((np.arange(start, start + 4), [start]))
        artists.extend(ax.plot(points[0, inds], points[1, inds], points[2, inds], edgecolor + ':', alpha=alpha))

    for side in range(0, 4):
        inds = (side, side + 4)
        artists.extend(ax.plot(points[0, inds], points[1, inds], points[2, inds], edgecolor + ':', alpha=alpha))

    return artists
