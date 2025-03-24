import matplotlib.pyplot as plt
import networkx as nx

# Define the flowchart structure
G = nx.DiGraph()

# Define nodes with descriptive labels
nodes = {
    "input": "User Input\n(Web Form)",
    "flask": "Flask App\n(app.py)",
    "tokenizer": "Tokenizer\n(HF Transformers)",
    "inference": "Model Inference\n(StarCoder + LoRA)",
    "post": "Post-processing\n(Regex, Cleanup)",
    "result": "Result Display\n(Web Response)"
}

# Add nodes to graph
for key, label in nodes.items():
    G.add_node(key, label=label)

# Define edges between nodes
edges = [
    ("input", "flask"),
    ("flask", "tokenizer"),
    ("flask", "inference"),
    ("tokenizer", "post"),
    ("inference", "post"),
    ("post", "result")
]

G.add_edges_from(edges)

# Layout for clearer spacing
pos = {
    "input": (-2, 0),
    "flask": (0, 0),
    "tokenizer": (2, 2),
    "inference": (2, -2),
    "post": (4, 0),
    "result": (6, 0)
}

# Draw the diagram
plt.figure(figsize=(12, 6))
nx.draw(
    G,
    pos,
    with_labels=False,
    node_size=10000,  # Increased node size
    node_color="#d0e1f2",
    arrows=True,
    connectionstyle="arc3,rad=0.0"
)

# Draw labels on top of nodes
for key in G.nodes():
    x, y = pos[key]
    plt.text(x, y, nodes[key], fontsize=10, ha="center", va="center")

plt.axis("off")
plt.tight_layout()
plt.savefig("/mnt/data/flask_app_flow_big_nodes.png")
plt.show()
