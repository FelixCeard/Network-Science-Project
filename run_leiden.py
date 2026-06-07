from __future__ import annotations

import argparse
import time
from pathlib import Path

import igraph as ig
import leidenalg
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

project_dir = Path(__file__).resolve().parent

misc_dir = project_dir / "output" / "misc"
assert misc_dir.exists(), "Misc directory does not exist"

leiden_dir = project_dir / "output" / "algorithms" / "leiden"
leiden_dir.mkdir(parents=True, exist_ok=True)

# Leiden uses the same undirected count-weighted graph as Louvain.
edges_file = misc_dir / "edges_undirected_count.csv"

# Output files created by this script.
communities_out = leiden_dir / "leiden_communities.csv"
summary_out = leiden_dir / "leiden_summary.csv"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Leiden community detection on the undirected Ethereum "
            "transaction graph."
        )
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help=(
            "Resolution parameter.  When omitted, ModularityVertexPartition is "
            "used (no resolution parameter).  When provided, CPMVertexPartition "
            "is used with this value as the resolution."
        ),
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=-1,
        help=(
            "Number of Leiden iterations.  -1 (default) runs until convergence."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------


def load_graph(path: Path) -> tuple[ig.Graph, dict[str, int], list[str]]:
    """
    Load the undirected weighted graph from the edge-list CSV and convert it
    to an igraph.Graph suitable for leidenalg.

    leidenalg operates on igraph objects, so NetworkX is not used here.
    Node identities (Ethereum address strings) are mapped to integer igraph
    vertex IDs and stored for the reverse mapping.

    Returns
    -------
    graph       : igraph.Graph with a ``weight`` edge attribute
    node_to_id  : mapping from address string to igraph vertex ID
    id_to_node  : reverse mapping (list indexed by vertex ID)
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing edge file: {path}")

    edges_df = pd.read_csv(path)

    required_columns = {"source", "target", "weight"}
    missing = required_columns - set(edges_df.columns)
    if missing:
        raise ValueError(f"Missing columns in edge file: {missing}")

    # Collect all unique node names and assign stable integer IDs.
    all_nodes: list[str] = sorted(
        set(edges_df["source"].tolist()) | set(edges_df["target"].tolist())
    )
    node_to_id: dict[str, int] = {node: idx for idx, node in enumerate(all_nodes)}

    # Build igraph edge list and weight list in one pass.
    edge_list = [
        (node_to_id[row.source], node_to_id[row.target])
        for row in edges_df.itertuples(index=False)
    ]
    weights = [float(row.weight) for row in edges_df.itertuples(index=False)]

    graph = ig.Graph(
        n=len(all_nodes),
        edges=edge_list,
        directed=False,
    )
    graph.es["weight"] = weights

    return graph, node_to_id, all_nodes


# ---------------------------------------------------------------------------
# Leiden detection
# ---------------------------------------------------------------------------


def run_leiden(
    graph: ig.Graph,
    resolution: float | None,
    n_iterations: int,
    seed: int,
) -> tuple[dict[int, int], str, float]:
    """
    Run Leiden community detection.

    Quality function selection
    --------------------------
    * ``resolution`` is None  →  ModularityVertexPartition (comparable to
      Louvain; no resolution parameter needed).
    * ``resolution`` is a float  →  CPMVertexPartition with the given
      resolution.  Higher values produce smaller, denser communities.

    Returns
    -------
    vertex_to_community : mapping from igraph vertex ID to community ID
    quality_function    : human-readable name of the quality function used
    runtime             : wall-clock seconds
    """
    if resolution is None:
        partition_type = leidenalg.ModularityVertexPartition
        partition_kwargs: dict = {}
        quality_function = "ModularityVertexPartition"
    else:
        partition_type = leidenalg.CPMVertexPartition
        partition_kwargs = {"resolution_parameter": resolution}
        quality_function = f"CPMVertexPartition(resolution={resolution})"

    start = time.time()

    partition = leidenalg.find_partition(
        graph,
        partition_type,
        weights="weight",
        n_iterations=n_iterations,
        seed=seed,
        **partition_kwargs,
    )

    runtime = time.time() - start

    # leidenalg returns a VertexPartition whose membership list is indexed by
    # igraph vertex ID (0-based integer).
    vertex_to_community = {
        vertex_id: community_id
        for vertex_id, community_id in enumerate(partition.membership)
    }

    return vertex_to_community, quality_function, runtime


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_communities(
    vertex_to_community: dict[int, int],
    id_to_node: list[str],
) -> None:
    """
    Save one community assignment per node.
    Output format is shared across all algorithms.
    """
    rows = [
        {
            "node": id_to_node[vertex_id],
            "algorithm": "leiden",
            "graph_variant": "undirected_count",
            "community": community_id,
        }
        for vertex_id, community_id in vertex_to_community.items()
    ]

    communities = pd.DataFrame(rows)
    communities.to_csv(communities_out, index=False)


def save_summary(
    graph: ig.Graph,
    vertex_to_community: dict[int, int],
    quality_function: str,
    resolution: float | None,
    runtime: float,
) -> None:
    """
    Save basic information about the Leiden result.
    These values are useful for algorithm comparison in the paper.
    """
    community_sizes = pd.Series(list(vertex_to_community.values())).value_counts()

    summary = pd.DataFrame(
        [
            {
                "algorithm": "leiden",
                "graph_variant": "undirected_count",
                "quality_function": quality_function,
                "resolution": resolution if resolution is not None else "N/A",
                "num_nodes": graph.vcount(),
                "num_edges": graph.ecount(),
                "num_communities": int(community_sizes.size),
                "median_community_size": float(community_sizes.median()),
                "largest_community_size": int(community_sizes.max()),
                "runtime_seconds": runtime,
            }
        ]
    )

    summary.to_csv(summary_out, index=False)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Complete Leiden pipeline:
      1. Load the undirected transaction graph.
      2. Run Leiden community detection.
      3. Save node-to-community assignments.
      4. Save summary statistics.
    """
    args = parse_args()

    print("Loading graph...")
    graph, node_to_id, id_to_node = load_graph(edges_file)
    print(f"Loaded graph: {graph.vcount()} nodes, {graph.ecount()} edges.")

    print("Running Leiden...")
    vertex_to_community, quality_function, runtime = run_leiden(
        graph,
        resolution=args.resolution,
        n_iterations=args.n_iterations,
        seed=args.seed,
    )

    save_communities(vertex_to_community, id_to_node)
    save_summary(graph, vertex_to_community, quality_function, args.resolution, runtime)

    num_communities = len(set(vertex_to_community.values()))

    print("Leiden complete.")
    print(f"Nodes:           {graph.vcount()}")
    print(f"Edges:           {graph.ecount()}")
    print(f"Communities:     {num_communities}")
    print(f"Quality function: {quality_function}")
    print(f"Runtime seconds: {runtime:.2f}")
    print(f"Wrote communities to: {communities_out}")
    print(f"Wrote summary to:     {summary_out}")


if __name__ == "__main__":
    main()
