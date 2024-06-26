import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import quaternion


def parametric_square(u, v):
    x = u
    y = v
    return x, y


def spherical_coordinates(u, v):
    r = 1
    theta = u * 2 * np.pi
    phi = v * np.pi
    return theta, phi, r


def quaternion_rotation(qx, qy):
    q = qx * qy
    return q


# Create a grid of points on the square
u_values = np.linspace(0, 1, 10)
v_values = np.linspace(0, 1, 10)
u_grid, v_grid = np.meshgrid(u_values, v_values)

# Map the square onto the sphere using spherical coordinates
theta, phi, r = spherical_coordinates(u_grid, v_grid)

# Create a quaternion for rotation (30° around x and 45° around y)
qx = quaternion.from_float_array(
    [np.cos(30 * np.pi / 180), np.sin(30 * np.pi / 180), 0, 0]
)
qy = quaternion.from_float_array(
    [np.cos(45 * np.pi / 180), 0, np.sin(45 * np.pi / 180), 0]
)

# Rotate the points on the sphere

rotated_points = []

for u in range(len(u_values)):
    for v in range(len(v_values)):
        q = quaternion.quaternion(*spherical_coordinates(u, v)) * qx * qy
        rotated_point = np.array(
            [q[1], q[2], q[3]]
        )  # Assuming the real part is at index
        rotated_points.append((rotated_point[0], rotated_point[1], rotated_point[2]))

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*zip(*rotated_points), c="b", marker="o")
plt.show()
