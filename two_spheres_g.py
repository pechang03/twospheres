import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the sphere parameters
radius = 1.0
center1 = [0, radius, 0]
center2 = [0, -radius, 0]

def plot_spheres(ax):
    # Plot the first sphere
    u = np.linspace(0, 2 * np.pi, 100)
