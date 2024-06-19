import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

t = np.linspace(0, 2 * np.pi, 1000)
 x = np.sin(t) + 2 * np.sin(2 * t)
 y = np.cos(t) - 2 * np.cos(2 * t)
 z = -np.sin(3 * t)

 fig = plt.figure()
 ax = fig.add_subplot(111, projection="3d")
 ax.plot(x, y, z, color="b", alpha=0.5)
 ax.set_xlim(-4, 4)
 ax.set_ylim(-4, 4)
 ax.set_zlim(-4, 4)
 ax.set_xlabel("X")
 ax.set_ylabel("Y")
 ax.set_zlabel("Z")

 plt.show()

