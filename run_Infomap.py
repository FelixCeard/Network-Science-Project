from pathlib import Path
import time

import networkx as nx
import pandas as pd
from infomap import Infomap


# Folder where this script is
project_dir = Path(__file__).resolve().parent

misc_dir = project_dir / "output" / "misc"
assert misc_dir.exists(), "Misc directory does not exist"

infomap_dir = project_dir / "output" / "algorithms" / "infomap"
infomap_dir.mkdir(parents=True, exist_ok=True)

# Infomap uses the directed count-weighted graph because it models transaction flow.
edges_file = misc_dir / "edges_directed_count.csv"

# Output files created by this
communities_out = infomap_dir / "infomap_communities.csv"
summary_out = infomap_dir / "infomap_summary.csv"


def load_graph(path: Path) -> nx.DiGraph:
    """
    Load the directed weighted graph from the edge-list CSV.
    The input file must contain:
    - source: transaction sender
    - target: transaction receiver
    - weight: number of transactions from source to target
    Returns a NetworkX directed graph.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing edge file: {path}")
    edges = pd.read_csv(path)

    # Make sure the file has the columns needed to build the graph
    required_columns = {"source", "target", "weight"}
    missing = required_columns - set(edges.columns)

    if missing:
        raise ValueError(f"Missing columns in edge file: {missing}")

    graph = nx.DiGraph()

    # Add every weighted edge to the graph
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.source,
            row.target,
            weight=float(row.weight),
        )

    return graph


def run_infomap(graph: nx.DiGraph) -> tuple[dict, float]:
    """
    Run Infomap community detection on the directed transaction graph.
    Infomap requires integer node IDs, so original account IDs are mapped to
    integers before running and mapped back afterward.
    """
    node_to_id = {node: idx for idx, node in enumerate(graph.nodes())}
    id_to_node = {idx: node for node, idx in node_to_id.items()}

    infomap = Infomap("--directed --two-level --silent --seed 42")

    for source, target, data in graph.edges(data=True):
        infomap.add_link(
            node_to_id[source],
            node_to_id[target],
            float(data.get("weight", 1.0)),
        )

    start = time.time()

    infomap.run()

    runtime = time.time() - start

    partition = {
        id_to_node[node.node_id]: node.module_id
        for node in infomap.tree
        if node.is_leaf
    }

    missing_nodes = set(graph.nodes()) - set(partition)
    if missing_nodes:
        raise ValueError(
            f"Infomap did not return communities for {len(missing_nodes)} nodes"
        )

    return partition, runtime


def save_communities(partition: dict) -> None:
    """
    Save one community assignment per node.
    This output format is shared across all algorithms ideally.
    """
    rows = [
        {
            "node": node,
            "algorithm": "infomap",
            "graph_variant": "directed_count",
            "community": community,
        }
        for node, community in partition.items()
    ]

    communities = pd.DataFrame(rows)
    communities.to_csv(communities_out, index=False)


def save_summary(graph: nx.DiGraph, partition: dict, runtime: float) -> None:
    """
    Save basic information about the Infomap result.
    These values are useful for algorithm comparison in the paper.
    """
    community_sizes = pd.Series(partition).value_counts()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "infomap",
                "graph_variant": "directed_count",
                "num_nodes": graph.number_of_nodes(),
                "num_edges": graph.number_of_edges(),
                "num_communities": int(community_sizes.size),
                "median_community_size": float(community_sizes.median()),
                "largest_community_size": int(community_sizes.max()),
                "runtime_seconds": runtime,
            }
        ]
    )

    summary.to_csv(summary_out, index=False)


def main() -> None:
    """
    Complete Infomap pipeline:
    load the directed graph;
    run Infomap;
    save node-to-community assignments;
    save summary statistics.
    """
    graph = load_graph(edges_file)

    partition, runtime = run_infomap(graph)

    save_communities(partition)
    save_summary(graph, partition, runtime)

    print("Infomap complete.")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Communities: {len(set(partition.values()))}")
    print(f"Runtime seconds: {runtime:.2f}")
    print(f"Wrote communities to: {communities_out}")
    print(f"Wrote summary to: {summary_out}")


if __name__ == "__main__":
    main()