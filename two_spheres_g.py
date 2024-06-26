import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the sphere parameters
radius = 1.0
center1 = [0, radius, 0]
center2 = [0, -radius, 0]

# Create a meshgrid for the plot
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi, 100)
U, V = np.meshgrid(u, v)

# Calculate the coordinates of each point on the spheres
X1 = center1[0] + radius * np.sin(V) * np.cos(U)
Y1 = center1[1] + radius * np.sin(V) * np.sin(U)
Z1 = center1[2] + radius * np.cos(V)
X2 = center2[0] + radius * np.sin(V) * np.cos(U)
Y2 = center2[1] + radius * np.sin(V) * np.sin(U)
Z2 = center2[2] + radius * np.cos(V)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Plot the spheres using plot_surface
ax.plot_surface(X1, Y1, Z1, color="b", alpha=0.5)
ax.plot_surface(X2, Y2, Z2, color="r", alpha=0.5)

# Set the axis limits and labels
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
# ax.set_zlim(-3, 3)
ax.set_xlabel("X")
ax.set_ylabel("Y")
# ax.set_zlabel("Z")

# Show the plot
plt.show()
