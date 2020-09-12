import numpy as np


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
        * *edgecolor* -- Color of box edges, default 'k''
        * *pointcolor* -- Color of box corner points, default 'k'

    :param ax: Current axes
    :param points: Corner points in box (3x8 matrix)
    :param kwargs: See above

    :return: List of artists.
    """
    edgecolor = kwargs.get('edgecolor', 'k')
    pointcolor = kwargs.get('pointcolor', 'k')

    artists = [ax.plot(points[0, :], points[1, :], points[2, :], pointcolor + '.')]

    for start in (0, 4):
        inds = np.concatenate((np.arange(start, start + 4), [start]))
        artists.append(ax.plot(points[0, inds], points[1, inds], points[2, inds], edgecolor + ':'))

    for side in range(0, 4):
        inds = (side, side + 4)
        artists.append(ax.plot(points[0, inds], points[1, inds], points[2, inds], edgecolor + ':'))
