import visgeom.utils
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np


def axis_equal(ax):
    """Emulate ax.axis('equal') for 3D axes, which is currently not supported in matplotlib.

    :param ax: Current axes
    """
    ax.set_box_aspect([1, 1, 1])

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def plot_pose(ax, pose, **kwargs):
    """Plot the pose (R, t) in the global frame.

    Keyword Arguments
        * *alpha* -- Alpha value (transparency), default 1
        * *axis_colors* -- List of colors for each axis, default ('r', 'g', 'b')
        * *scale* -- Scale factor, default 1.0
        * *text* -- Text description plotted at pose origin, default ''

    :param ax: Current axes
    :param pose: The pose (R, t) of the local frame relative to the global frame,
        where R is a 3x3 rotation matrix and t is a 3D column vector.
    :param kwargs: See above

    :return: List of artists.
    """
    R, t = pose
    alpha = kwargs.get('alpha', 1)
    axis_colors = kwargs.get('axis_colors', ('r', 'g', 'b'))
    scale = kwargs.get('scale', 1)
    text = kwargs.get('text', '')

    artists = []

    # If R is a valid rotation matrix, the columns are the local orthonormal basis vectors in the global frame.
    for i in range(0, 3):
        axis_line = np.column_stack((t, t + R[:, i, np.newaxis] * scale))
        artists.append(ax.plot(axis_line[0, :], axis_line[1, :], axis_line[2, :], axis_colors[i] + '-', alpha=alpha))

    if text:
        artists.append(ax.text(t[0, 0], t[1, 0], t[2, 0], text))

    return artists


def plot_camera_frustum(ax, K, pose_w_c, **kwargs):
    """Plot a camera frustum in the global "world" frame.

    Keyword Arguments
        * *alpha* -- Alpha value (transparency), default 1
        * *edgecolor* -- Frustum color, default 'k'
        * *img_size* -- Size of image in pixels, default (2*K[0, 2], 2*K[1, 2])
        * *scale* -- Scale factor, default 1.0
        * *text* -- Text description plotted at camera origin, default ''

    :param ax: Current axes
    :param K: Camera calibration matrix (3x3 upper triangular matrix)
    :param pose_w_c: The pose (R, t) of the camera frame relative to the world frame,
        where R is a 3x3 rotation matrix and t is a 3D column vector.
    :param kwargs: See above

    :return: List of artists.
    """
    R_w_c, t_w_c = pose_w_c
    alpha = kwargs.get('alpha', 1)
    edgecolor = kwargs.get('edgecolor', 'k')
    img_size = kwargs.get('img_size', (2 * K[0, 2], 2 * K[1, 2]))
    scale = kwargs.get('scale', 1)
    text = kwargs.get('text', '')

    # Homogeneous coordinates (normalised) for the corner pixels.
    img_corners_uh = np.array([[0, img_size[0], img_size[0], 0],
                               [0, 0, img_size[1], img_size[1]],
                               np.ones(4)])

    # Corners transformed to the normalised image plane.
    img_corners_xn = np.linalg.inv(K) @ img_corners_uh

    # Frustum points in the camera coordinate system.
    frustum_x_c = np.hstack((np.zeros((3, 1)), img_corners_xn))

    # Frustum points in the world coordinate system.
    S = scale * np.identity(3)
    frustum_x_w = R_w_c @ S @ frustum_x_c + t_w_c

    # Plot outline.
    inds = (0, 4, 3, 0, 1, 4, 0, 3, 2, 0, 2, 1, 0)
    artists = [ax.plot(frustum_x_w[0, inds], frustum_x_w[1, inds], frustum_x_w[2, inds], edgecolor + '-', alpha=alpha)]

    if text:
        artists.append(ax.text(t_w_c[0, 0], t_w_c[1, 0], t_w_c[2, 0], text))

    return artists


def plot_camera_image_plane(ax, K, pose_w_c, **kwargs):
    """Plot a camera image plane in the global "world" frame.

    Keyword Arguments
        * *alpha* -- Alpha value (transparency), default 0.25
        * *edgecolor* -- Color of edge around image plane, default 'k'
        * *facecolor* -- Image plane color, default 'b'
        * *img_size* -- Size of image in pixels, default (2*K[0, 2], 2*K[1, 2])
        * *scale* -- Scale factor, default 1.0

    :param ax: Current axes
    :param K: Camera calibration matrix (3x3 upper triangular matrix)
    :param pose_w_c: The pose (R, t) of the camera frame relative to the world frame,
        where R is a 3x3 rotation matrix and t is a 3D column vector.
    :param kwargs: See above

    :return: List of artists.
    """
    R_w_c, t_w_c = pose_w_c
    alpha = kwargs.get('alpha', 0.25)
    edgecolor = kwargs.get('edgecolor', 'k')
    facecolor = kwargs.get('facecolor', 'b')
    img_size = kwargs.get('img_size', (2 * K[0, 2], 2 * K[1, 2]))
    scale = kwargs.get('scale', 1)

    # Homogeneous coordinates (normalised) for the corner pixels.
    img_corners_uh = np.array([[0, img_size[0], img_size[0], 0],
                               [0, 0, img_size[1], img_size[1]],
                               np.ones(4)])

    # Corners transformed to the normalised image plane.
    img_corners_xn = np.linalg.inv(K) @ img_corners_uh

    # Image plane points in the world coordinate system.
    S = scale * np.identity(3)
    plane_x_w = R_w_c @ S @ img_corners_xn + t_w_c

    # Plot plane
    poly = Poly3DCollection([plane_x_w.T], alpha=alpha, facecolor=facecolor, edgecolor=edgecolor)
    artists = [ax.add_collection(poly)]

    return artists


def plot_covariance_ellipsoid(ax, mean, covariance, chi2_val=11.345, **kwargs):
    """Plot a 3D covariance ellipsoid.

    Keyword Arguments
        * *alpha* -- Alpha value (transparency), default 0.2
        * *color* -- Ellipsoid surface color, default 'r'
        * *n* -- Granularity of the ellipsoid, default 20

    :param ax: Current axes
    :param mean: The mean, a 3D column vector.
    :param covariance: The covariance, a 3x3 matrix.
    :param chi2_val: The chi-square distribution value for the ellipsoid scale. Default 11.345 corresponds to 99%
    :param kwargs: See above

    :return: List of artists.
    """
    alpha = kwargs.get('alpha', 0.2)
    color = kwargs.get('color', 'r')
    n = kwargs.get('n', 20)

    u, s, _ = np.linalg.svd(covariance)
    scale = np.sqrt(chi2_val * s)

    x, y, z = visgeom.utils.generate_ellipsoid(n, pose=(u, mean), scale=scale)

    return [ax.plot_surface(x, y, z, alpha=alpha, color=color)]
