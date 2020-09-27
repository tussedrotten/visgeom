import matplotlib.pyplot as plt
import matplotlib
import visgeom as vg
import numpy as np

# Define distribution in 3D.
mean = np.zeros((3, 1))
covariance = np.diag(np.array([1, 2, 1]) ** 2)

# Draw points from distribution.
num_draws = 1000
random_points = np.random.multivariate_normal(mean.flatten(), covariance, num_draws).T

# Plot result.
# Use Qt 5 backend in visualisation.
matplotlib.use('qt5agg')

# Create figure and axis.
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# Plot simulated points.
ax.plot(random_points[0, :], random_points[1, :], random_points[2, :], 'k.', alpha=0.1)

# Plot the estimated mean pose.
vg.plot_covariance_ellipsoid(ax, mean, covariance)

# Show figure.
vg.plot.axis_equal(ax)
plt.show()
