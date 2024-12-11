def save_bolinas_str(fn, graph, add_names=False):
    bolinas_graph = graph.to_bolinas(add_names=add_names)
    with open(fn, "w") as f:
        f.write(f"{bolinas_graph}\n")


def save_as_dot(fn, graph):
    with open(fn, "w") as f:
        f.write(graph.to_dot())
