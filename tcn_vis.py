
import matplotlib.pyplot as plt
import networkx as nx

def draw_detailed_tcn_grid_flipped_causal(num_timepoints=30, kernel_size=2, dilations=[1, 2, 4, 8]):
    G = nx.DiGraph()

    # Generate input nodes (timepoints)
    for t in range(num_timepoints):
        G.add_node(f"t{t}", pos=(t, 0), layer="input")

    # Generate dilated causal convolutional layers
    paths_to_highlight = []
    all_edges = []
    dependency_edges = []
    
    for i, dilation in enumerate(dilations):
        for t in range(num_timepoints):
            if t >= dilation * (kernel_size - 1):
                G.add_node(f"c{i}_t{t}", pos=(t, i + 1), layer="conv")
                for k in range(kernel_size):
                    prev_t = t - k * dilation
                    if prev_t >= 0:
                        edge_start = f"t{prev_t}" if i == 0 else f"c{i - 1}_t{prev_t}"
                        if G.has_node(edge_start):  # Ensure the previous node exists
                            edge = (edge_start, f"c{i}_t{t}")
                            all_edges.append(edge)
                            if i > 0:
                                dependency_edges.append(edge)
                            if t == num_timepoints - 1:  # rightmost point
                                paths_to_highlight.append(edge)

    # Generate output nodes for causal convolutions
    for t in range(num_timepoints):
        if G.has_node(f"c{len(dilations) - 1}_t{t}"):  # Ensure the previous node exists
            G.add_node(f"out_c_t{t}", pos=(t, len(dilations) + 1), layer="output")
            edge = (f"c{len(dilations) - 1}_t{t}", f"out_c_t{t}")
            all_edges.append(edge)
            if t == num_timepoints - 1:  # rightmost point
                paths_to_highlight.append(edge)

    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    labels = {node: node for node in G.nodes()}

    # Color nodes based on layers
    node_colors = []
    for node in G.nodes():
        layer = G.nodes[node]['layer']
        if layer == "input":
            node_colors.append('lightblue')
        elif layer == "conv":
            node_colors.append('lightgreen')
        elif layer == "output":
            node_colors.append('red')

    plt.figure(figsize=(20, 10))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=300, node_color=node_colors, font_size=8, font_weight='bold', edge_color='gray')

    # Draw all edges in gray
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='gray', width=1)

    # Highlight the path for the rightmost point in blue
    nx.draw_networkx_edges(G, pos, edgelist=paths_to_highlight, edge_color='blue', width=2)

    # Highlight the dependency paths in red
    nx.draw_networkx_edges(G, pos, edgelist=dependency_edges, edge_color='red', width=1, style='dashed')

    plt.title("Causal Temporal Convolutional Network (TCN) with Dilations 1, 2, 4, 8")
    plt.show()

draw_detailed_tcn_grid_flipped_causal()


def draw_detailed_tcn_grid_flipped_noncausal(num_timepoints=30, kernel_size=2, layers=4):
    G = nx.DiGraph()

    # Generate input nodes (timepoints)
    for t in range(num_timepoints):
        G.add_node(f"t{t}", pos=(t, 0), layer="input")

    # Generate regular convolutional layers
    all_edges = []
    for i in range(layers):
        for t in range(num_timepoints):
            G.add_node(f"c{i}_t{t}", pos=(t, i + 1), layer="conv")
            for k in range(-(kernel_size // 2), (kernel_size // 2) + 1):
                prev_t = t + k
                if 0 <= prev_t < num_timepoints:
                    edge_start = f"t{prev_t}" if i == 0 else f"c{i - 1}_t{prev_t}"
                    if G.has_node(edge_start):  # Ensure the previous node exists
                        edge = (edge_start, f"c{i}_t{t}")
                        all_edges.append(edge)

    # Generate output nodes for regular convolutions
    for t in range(num_timepoints):
        G.add_node(f"out_c_t{t}", pos=(t, layers + 1), layer="output")
        edge = (f"c{layers - 1}_t{t}", f"out_c_t{t}")
        all_edges.append(edge)

    # Draw the graph
    pos = nx.get_node_attributes(G, 'pos')
    labels = {node: node for node in G.nodes()}

    # Color nodes based on layers
    node_colors = []
    for node in G.nodes():
        layer = G.nodes[node]['layer']
        if layer == "input":
            node_colors.append('lightblue')
        elif layer == "conv":
            node_colors.append('lightgreen')
        elif layer == "output":
            node_colors.append('red')

    plt.figure(figsize=(20, 10))
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=300, node_color=node_colors, font_size=8, font_weight='bold', edge_color='gray')

    # Draw all edges in gray
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='gray', width=1)

    plt.title("Non-Causal Regular Convolutional Network")
    plt.show()

draw_detailed_tcn_grid_flipped_noncausal()

