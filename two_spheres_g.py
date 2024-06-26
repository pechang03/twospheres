import matplotlib.pyplot as plt
import networkx as nx

# Create a simple graph
G = nx.Graph()
G.add_node(1, pos=(0, 0))
G.add_node(2, pos=(1, 0))
G.add_edge(1, 2)

# Draw the graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=True, node_size=3000, node_color="skyblue", font_size=10, font_weight='bold', edge_color='gray')

# Set plot limits and remove axes
plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 0.5)
plt.axis('off')

# Save the figure before showing it to avoid displaying an empty plot
plt.savefig("graph_plot.png")
