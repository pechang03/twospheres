import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import quaternion
from numpy import *


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
    [np.cos(30 * np.pi / 180), 0, 0, np.sin(30 * np.pi / 180)]
)

qy = quaternion.from_float_array(
    [np.cos(45 * np.pi / 180), 0, np.sin(45 * np.pi / 180), 0]
)
q_rot = qx * qy

# Rotate the mapped shape on the sphere's surface using quaternions
rotated_points = []

for theta, phi in zip(theta.ravel(), phi.ravel()):
    # Convert spherical coordinates to Cartesian coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Rotate the point using the quaternion
    q_0 = quaternion.quaternion(0.0, 0.0, 0.0, 0.0)

    q1 = quaternion.quaternion(x, y, z, 0.0)
    rotated_point = q_rot * q1 * q_rot.conj()
    rotated_points.append((rotated_point.x, rotated_point.y, rotated_point.z))

# Plot the resulting points on the sphere's surface

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(*zip(*rotated_points), c="b", marker="o")

plt.show()
