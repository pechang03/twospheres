## This code generates a set of points that form the surface of a tube around the trefoil knot curve. The tube is constructed by taking each point on the interpolated curve, computing a local Frenet frame (tangent, normal, and binormal vectors), and then applying this frame as a rotation matrix to a circle of points in the plane normal to the curve at that point. The radius of the circle determines the thickness of the tube.

# The resulting `surf` array contains the coordinates of the tube surface, which can be plotted using Matplotlib's 3D scatter plot. This gives you a visual representation of a vortex ring with a trefoil knot structure. Keep in mind that this is a static geometric model and does not represent the dynamics of fluid flow within the vortex tube. For dynamic simulation, more advanced techniques from computational fluid dynamics would be required.

# Generated with the help of Wizardlm2

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import splprep, splev

# Define parameters of trefoil knot
a = 5.0  # Scale factor for the x-coordinate
b = 1.0  # Scale factor for the z-coordinate
c = 0.5  # Scale factor for the y-coordinate
n_turns = 3  # Number of turns in the knot (3 for a trefoil)
radius = 0.2  # Radius of the vortex tube
ntheta = 100  # Number of points for circle in parameter space

# Create a 1D grid for the parameter t
t = np.linspace(0, 4 * np.pi, 1000)

# Define trefoil knot shape
x = a * np.sin(n_turns * t)
y = c * (np.cos(t) + 2 * np.cos(2 * t))
z = b * (np.sin(t) - 2 * np.sin(2 * t))

# Interpolate the curve to get a smooth representation
tck, u = splprep([x, y, z], s=0)
xi, yi, zi = splev(u, tck)

# Create points on a circle in parameter space
theta = np.linspace(0, 2 * np.pi, ntheta)
circle_points = np.c_[np.cos(theta), np.sin(theta)]

# Generate the tube surface around the trefoil knot curve
surf = []
for i in range(len(xi)):
    # Rotation matrix for the Frenet frame of the curve at point i
    tangent = np.array(
        [xi[i + 1] - xi[i - 1], yi[i + 1] - yi[i - 1], zi[i + 1] - zi[i - 1]]
    )
    normal = splprep([xi, yi, zi], u, s=0, der=1)[1][0]
    binormal = np.cross(tangent, normal)
    R = np.c_[normal, binormal, tangent].T

    # Apply the rotation to each point on the circle and scale by tube radius
    points = R @ (circle_points * radius).T + np.array([xi[i], yi[i], zi[i]])
    surf.append(points)

# Convert list of arrays into a single array for plotting
surf = np.array(surf).reshape((-1, 3))

# Plot the tube surface
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(surf[:, 0], surf[:, 1], surf[:, 2], c=xi, cmap="viridis")

# Set plot limits and labels
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
# ax.set_zlim(-10, 10)
ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.set_zlabel('Z')
ax.set_title("Vortex Ring with Trefoil Knot Structure")

plt.show()
