from pathlib import Path
import time

import networkx as nx
import pandas as pd
import community as community_louvain


# Folder where this script is
project_dir = Path(__file__).resolve().parent

misc_dir = project_dir / "output" / "misc"
assert misc_dir.exists(), "Misc directory does not exist"

louvain_dir = project_dir / "output" / "algorithms" / "louvain"
louvain_dir.mkdir(parents=True, exist_ok=True)

# Louvain uses the undirected count-weighted graph
edges_file = misc_dir / "edges_undirected_count.csv"

# Output files created by this
communities_out = louvain_dir / "louvain_communities.csv"
summary_out = louvain_dir / "louvain_summary.csv"


def load_graph(path: Path) -> nx.Graph:
    """
    Load the undirected weighted graph from the edge-list CSV;
    The input file must contain:
    - source: first endpoint
    - target: second endpoint
    - weight: number of transactions between the two addresses
    Returns a NetworkX undirected graph.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing edge file: {path}")
    edges = pd.read_csv(path)

    # Make sure the file has the columns needed to build the graph
    required_columns = {"source", "target", "weight"}
    missing = required_columns - set(edges.columns)

    if missing:
        raise ValueError(f"Missing columns in edge file: {missing}")

    graph = nx.Graph()

    # Add every weighted edge to the graph
    for row in edges.itertuples(index=False):
        graph.add_edge(
            row.source,
            row.target,
            weight=float(row.weight),
        )

    return graph


def run_louvain(graph: nx.Graph) -> tuple[dict, float]:
    """
    Run Louvain community detection.
    Louvain returns a partition dictionary:
    node -> community_id
    The phishing labels are not used here, Louvain only sees the graph structure.
    """
    start = time.time()

    partition = community_louvain.best_partition(
        graph,
        weight="weight",
        random_state=42,
    )

    runtime = time.time() - start

    return partition, runtime


def save_communities(partition: dict) -> None:
    """
    Save one community assignment per node.
    This output format is shared across all algorithms ideally.
    """
    rows = [
        {
            "node": node,
            "algorithm": "louvain",
            "graph_variant": "undirected_count",
            "community": community,
        }
        for node, community in partition.items()
    ]

    communities = pd.DataFrame(rows)
    communities.to_csv(communities_out, index=False)


def save_summary(graph: nx.Graph, partition: dict, runtime: float) -> None:
    """
    Save basic information about the Louvain result.
    These values are useful for algorithm comparison in the paper.
    """
    community_sizes = pd.Series(partition).value_counts()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "louvain",
                "graph_variant": "undirected_count",
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
    complete Louvain pipeline:
    load the undirected graph;
    run Louvain;
    save node-to-community assignments;
    save summary statistics.
    """
    graph = load_graph(edges_file)

    partition, runtime = run_louvain(graph)

    save_communities(partition)
    save_summary(graph, partition, runtime)

    print("Louvain complete.")
    print(f"Nodes: {graph.number_of_nodes()}")
    print(f"Edges: {graph.number_of_edges()}")
    print(f"Communities: {len(set(partition.values()))}")
    print(f"Runtime seconds: {runtime:.2f}")
    print(f"Wrote communities to: {communities_out}")
    print(f"Wrote summary to: {summary_out}")


if __name__ == "__main__":
    main()