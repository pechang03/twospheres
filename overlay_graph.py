import matplotlib.pyplot as plt
import networkx as nx
from two_spheres_g import plot_spheres  # Assuming the function to plot spheres is in this file

# Create the geometric graph with seed for reproducibility
G = nx.random_geometric_graph(200, 0.125, seed=896803)
# Get positions of nodes
pos = nx.get_node_attributes(G, "pos")

# Find node near the center (0.5, 0.5)
dmin = 1
ncenter = 0
for n in pos:
    x, y = pos[n]
    d = (x - 0.5) ** 2 + (y - 0.5) ** 2
    if d < dmin:
        ncenter = n
        dmin = d

# Color nodes by path length from node near the center
p = dict(nx.single_source_shortest_path_length(G, ncenter))

# Plot spheres from two_spheres_g.py
fig, ax = plt.subplots()
plot_spheres(ax)  # Assuming this function takes an axis as argument

# Overlay the geometric graph nodes
nx.draw_networkx_nodes(
    G,
    pos,
    nodelist=list(p.keys()),
    node_size=80,
    node_color=list(p.values()),
    cmap=plt.cm.Reds_r,
    ax=ax,  # Draw on the same axis as spheres
)

# Draw edges of the geometric graph
nx.draw_networkx_edges(G, pos, alpha=0.4, ax=ax)

# Set plot limits and remove axes
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis("off")

plt.show()
