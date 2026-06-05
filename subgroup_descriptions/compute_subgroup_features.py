import argparse
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


project_dir = Path(__file__).resolve().parent.parent
output_dir = Path(__file__).resolve().parent

misc_dir = project_dir / "output" / "misc"
algorithms_dir = project_dir / "output" / "algorithms"

MAX_NODES_FOR_PATHS = 300
MAX_NODES_FOR_CENTRALITY = 500
MAX_NODES_FOR_CONSTRAINT = 200

ALGORITHM_CONFIG = {
    "infomap": {
        "graph_variant": "directed_count",
        "edges_file": "edges_directed_count.csv",
        "directed": True,
    },
    "louvain": {
        "graph_variant": "undirected_count",
        "edges_file": "edges_undirected_count.csv",
        "directed": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute NetworkX structural features for each detected community "
            "(subgroup) and save them to CSV."
        )
    )
    parser.add_argument(
        "--algorithm",
        choices=sorted(ALGORITHM_CONFIG),
        default="infomap",
        help="Community detection algorithm whose partition to describe.",
    )
    return parser.parse_args()


def load_graph(path: Path, directed: bool):
    edges = pd.read_csv(path)
    required_columns = {"source", "target", "weight"}
    missing = required_columns - set(edges.columns)
    if missing:
        raise ValueError(f"Missing columns in edge file: {missing}")

    graph = nx.DiGraph() if directed else nx.Graph()

    for row in edges.itertuples(index=False):
        attrs = {"weight": float(row.weight)}
        if hasattr(row, "phishing_tx_count"):
            attrs["phishing_tx_count"] = float(row.phishing_tx_count)
        if hasattr(row, "total_tx_count"):
            attrs["total_tx_count"] = float(row.total_tx_count)
        graph.add_edge(row.source, row.target, **attrs)

    return graph


def load_communities(algorithm: str) -> pd.DataFrame:
    communities_file = (
        algorithms_dir / algorithm / f"{algorithm}_communities.csv"
    )
    if not communities_file.exists():
        raise FileNotFoundError(f"Missing communities file: {communities_file}")

    communities = pd.read_csv(communities_file)
    required_columns = {"node", "community"}
    missing = required_columns - set(communities.columns)
    if missing:
        raise ValueError(f"Missing columns in communities file: {missing}")

    return communities


def safe_metric(func, default=np.nan):
    try:
        return func()
    except (
        nx.NetworkXError,
        nx.NetworkXPointlessConcept,
        nx.PowerIterationFailedConvergence,
        ZeroDivisionError,
        ValueError,
        FloatingPointError,
        KeyError,
    ):
        return default


def without_self_loops(graph):
    cleaned = graph.copy()
    cleaned.remove_edges_from(nx.selfloop_edges(cleaned))
    return cleaned


def summarize(values, prefix: str) -> dict:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_std": float(arr.std()),
        f"{prefix}_min": float(arr.min()),
        f"{prefix}_max": float(arr.max()),
    }


def degree_stats(graph, directed: bool) -> dict:
    if graph.number_of_nodes() == 0:
        return {
            "avg_degree": np.nan,
            "median_degree": np.nan,
            "std_degree": np.nan,
            "min_degree": np.nan,
            "max_degree": np.nan,
            "avg_weighted_degree": np.nan,
            "avg_in_degree": np.nan,
            "avg_out_degree": np.nan,
            "avg_weighted_in_degree": np.nan,
            "avg_weighted_out_degree": np.nan,
        }

    if directed:
        in_degrees = np.array([d for _, d in graph.in_degree()], dtype=float)
        out_degrees = np.array([d for _, d in graph.out_degree()], dtype=float)
        weighted_in = np.array(
            [d for _, d in graph.in_degree(weight="weight")], dtype=float
        )
        weighted_out = np.array(
            [d for _, d in graph.out_degree(weight="weight")], dtype=float
        )
        total_degrees = in_degrees + out_degrees
        weighted_degrees = weighted_in + weighted_out
        return {
            "avg_degree": float(total_degrees.mean()),
            "median_degree": float(np.median(total_degrees)),
            "std_degree": float(total_degrees.std()),
            "min_degree": float(total_degrees.min()),
            "max_degree": float(total_degrees.max()),
            "avg_weighted_degree": float(weighted_degrees.mean()),
            "avg_in_degree": float(in_degrees.mean()),
            "avg_out_degree": float(out_degrees.mean()),
            "avg_weighted_in_degree": float(weighted_in.mean()),
            "avg_weighted_out_degree": float(weighted_out.mean()),
        }

    degrees = np.array([d for _, d in graph.degree()], dtype=float)
    weighted_degrees = np.array(
        [d for _, d in graph.degree(weight="weight")], dtype=float
    )
    return {
        "avg_degree": float(degrees.mean()),
        "median_degree": float(np.median(degrees)),
        "std_degree": float(degrees.std()),
        "min_degree": float(degrees.min()),
        "max_degree": float(degrees.max()),
        "avg_weighted_degree": float(weighted_degrees.mean()),
        "avg_in_degree": np.nan,
        "avg_out_degree": np.nan,
        "avg_weighted_in_degree": np.nan,
        "avg_weighted_out_degree": np.nan,
    }


def edge_weight_stats(graph) -> dict:
    if graph.number_of_edges() == 0:
        return {
            "total_edge_weight": 0.0,
            "avg_edge_weight": np.nan,
            "median_edge_weight": np.nan,
            "max_edge_weight": np.nan,
            "total_phishing_edge_weight": 0.0,
            "phishing_edge_weight_fraction": np.nan,
        }

    weights = np.array(
        [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)],
        dtype=float,
    )
    phishing_weights = np.array(
        [
            data.get("phishing_tx_count", 0.0)
            for _, _, data in graph.edges(data=True)
        ],
        dtype=float,
    )
    total_weight = float(weights.sum())
    total_phishing_weight = float(phishing_weights.sum())

    return {
        "total_edge_weight": total_weight,
        "avg_edge_weight": float(weights.mean()),
        "median_edge_weight": float(np.median(weights)),
        "max_edge_weight": float(weights.max()),
        "total_phishing_edge_weight": total_phishing_weight,
        "phishing_edge_weight_fraction": (
            total_phishing_weight / total_weight if total_weight else np.nan
        ),
    }


def largest_component_subgraph(graph, directed: bool):
    if graph.number_of_nodes() == 0:
        return graph.copy()

    if directed:
        if graph.number_of_edges() == 0:
            return graph.subgraph(next(iter(graph.nodes()))).copy()
        largest_nodes = max(nx.weakly_connected_components(graph), key=len)
    else:
        if graph.number_of_edges() == 0:
            return graph.subgraph(next(iter(graph.nodes()))).copy()
        largest_nodes = max(nx.connected_components(graph), key=len)

    return graph.subgraph(largest_nodes).copy()


def connectivity_stats(graph, directed: bool) -> dict:
    num_nodes = graph.number_of_nodes()
    if num_nodes == 0:
        return {
            "num_connected_components": 0,
            "largest_component_size": 0,
            "largest_component_fraction": np.nan,
            "is_connected": False,
            "num_weakly_connected_components": np.nan,
            "num_strongly_connected_components": np.nan,
            "largest_weak_component_size": np.nan,
            "largest_weak_component_fraction": np.nan,
            "largest_strong_component_size": np.nan,
            "largest_strong_component_fraction": np.nan,
            "is_weakly_connected": np.nan,
            "is_strongly_connected": np.nan,
        }

    if directed:
        weak_components = list(nx.weakly_connected_components(graph))
        strong_components = list(nx.strongly_connected_components(graph))
        largest_weak = max(weak_components, key=len)
        largest_strong = max(strong_components, key=len)
        return {
            "num_connected_components": np.nan,
            "largest_component_size": np.nan,
            "largest_component_fraction": np.nan,
            "is_connected": np.nan,
            "num_weakly_connected_components": len(weak_components),
            "num_strongly_connected_components": len(strong_components),
            "largest_weak_component_size": len(largest_weak),
            "largest_weak_component_fraction": len(largest_weak) / num_nodes,
            "largest_strong_component_size": len(largest_strong),
            "largest_strong_component_fraction": len(largest_strong) / num_nodes,
            "is_weakly_connected": nx.is_weakly_connected(graph),
            "is_strongly_connected": nx.is_strongly_connected(graph),
        }

    components = list(nx.connected_components(graph))
    largest = max(components, key=len)
    return {
        "num_connected_components": len(components),
        "largest_component_size": len(largest),
        "largest_component_fraction": len(largest) / num_nodes,
        "is_connected": nx.is_connected(graph),
        "num_weakly_connected_components": np.nan,
        "num_strongly_connected_components": np.nan,
        "largest_weak_component_size": np.nan,
        "largest_weak_component_fraction": np.nan,
        "largest_strong_component_size": np.nan,
        "largest_strong_component_fraction": np.nan,
        "is_weakly_connected": np.nan,
        "is_strongly_connected": np.nan,
    }


def path_stats(graph, directed: bool) -> dict:
    empty = {
        "diameter": np.nan,
        "radius": np.nan,
        "avg_shortest_path_length": np.nan,
        "wiener_index": np.nan,
        "center_size": np.nan,
        "periphery_size": np.nan,
    }
    if graph.number_of_nodes() > MAX_NODES_FOR_PATHS:
        return empty

    largest = largest_component_subgraph(graph, directed)
    if largest.number_of_nodes() < 2 or largest.number_of_edges() == 0:
        return empty

    if directed:
        if not nx.is_weakly_connected(largest):
            return empty
        analysis_graph = largest.to_undirected()
    else:
        if not nx.is_connected(largest):
            return empty
        analysis_graph = largest

    return {
        "diameter": safe_metric(lambda: nx.diameter(analysis_graph)),
        "radius": safe_metric(lambda: nx.radius(analysis_graph)),
        "avg_shortest_path_length": safe_metric(
            lambda: nx.average_shortest_path_length(analysis_graph)
        ),
        "wiener_index": safe_metric(lambda: nx.wiener_index(analysis_graph)),
        "center_size": safe_metric(
            lambda: len(nx.center(analysis_graph)), default=np.nan
        ),
        "periphery_size": safe_metric(
            lambda: len(nx.periphery(analysis_graph)), default=np.nan
        ),
    }


def structural_stats(graph, directed: bool) -> dict:
    empty = {
        "transitivity": np.nan,
        "avg_clustering": np.nan,
        "avg_weighted_clustering": np.nan,
        "degree_assortativity": np.nan,
        "degree_assortativity_weighted": np.nan,
        "in_degree_assortativity": np.nan,
        "out_degree_assortativity": np.nan,
        "in_out_degree_assortativity": np.nan,
        "reciprocity": np.nan,
        "num_bridges": np.nan,
        "num_articulation_points": np.nan,
        "algebraic_connectivity": np.nan,
        "spectral_radius": np.nan,
        "degree_pearson_correlation": np.nan,
        "total_triangles": np.nan,
        "avg_triangles": np.nan,
        "avg_square_clustering": np.nan,
        "max_core_number": np.nan,
        "avg_core_number": np.nan,
        "graph_clique_number": np.nan,
        "is_bipartite": np.nan,
        "global_efficiency": np.nan,
        "local_efficiency": np.nan,
        "avg_constraint": np.nan,
        "avg_effective_size": np.nan,
    }
    if graph.number_of_nodes() == 0:
        return empty

    undirected_view = without_self_loops(
        graph.to_undirected() if directed else graph
    )

    bridges = np.nan
    articulation_points = np.nan
    algebraic_connectivity = np.nan
    if not directed and graph.number_of_edges() > 0:
        bridges = safe_metric(lambda: len(list(nx.bridges(graph))))
        articulation_points = safe_metric(
            lambda: len(list(nx.articulation_points(graph)))
        )
        if nx.is_connected(graph):
            algebraic_connectivity = safe_metric(
                lambda: nx.algebraic_connectivity(graph, weight="weight")
            )

    spectral_radius = np.nan
    if graph.number_of_nodes() <= MAX_NODES_FOR_CENTRALITY and graph.number_of_edges() > 0:
        adjacency = nx.to_numpy_array(graph, weight="weight")
        eigenvalues = np.linalg.eigvals(adjacency)
        spectral_radius = float(np.max(np.abs(eigenvalues)))

    triangle_counts = safe_metric(
        lambda: list(nx.triangles(undirected_view).values()),
        default=[],
    )
    core_numbers = safe_metric(
        lambda: list(nx.core_number(undirected_view).values()),
        default=[],
    )

    constraint = np.nan
    effective_size = np.nan
    if graph.number_of_nodes() <= MAX_NODES_FOR_CONSTRAINT and graph.number_of_edges() > 0:
        undirected_for_burt = undirected_view
        constraint = safe_metric(
            lambda: float(np.mean(list(nx.constraint(undirected_for_burt).values())))
        )
        effective_size = safe_metric(
            lambda: float(
                np.mean(list(nx.effective_size(undirected_for_burt).values()))
            )
        )

    clique_number = np.nan
    if graph.number_of_nodes() <= MAX_NODES_FOR_CONSTRAINT:
        clique_number = safe_metric(
            lambda: nx.approximation.large_clique_size(undirected_view)
        )

    return {
        "transitivity": safe_metric(lambda: nx.transitivity(undirected_view)),
        "avg_clustering": safe_metric(lambda: nx.average_clustering(graph)),
        "avg_weighted_clustering": safe_metric(
            lambda: nx.average_clustering(graph, weight="weight")
        ),
        "degree_assortativity": safe_metric(
            lambda: nx.degree_assortativity_coefficient(graph)
        ),
        "degree_assortativity_weighted": safe_metric(
            lambda: nx.degree_assortativity_coefficient(graph, weight="weight")
        ),
        "in_degree_assortativity": (
            safe_metric(
                lambda: nx.degree_assortativity_coefficient(graph, x="in+in")
            )
            if directed
            else np.nan
        ),
        "out_degree_assortativity": (
            safe_metric(
                lambda: nx.degree_assortativity_coefficient(graph, x="out+out")
            )
            if directed
            else np.nan
        ),
        "in_out_degree_assortativity": (
            safe_metric(
                lambda: nx.degree_assortativity_coefficient(graph, x="in+out")
            )
            if directed
            else np.nan
        ),
        "reciprocity": (
            safe_metric(lambda: nx.reciprocity(graph)) if directed else np.nan
        ),
        "num_bridges": bridges,
        "num_articulation_points": articulation_points,
        "algebraic_connectivity": algebraic_connectivity,
        "spectral_radius": spectral_radius,
        "degree_pearson_correlation": safe_metric(
            lambda: nx.degree_pearson_correlation_coefficient(undirected_view)
        ),
        "total_triangles": (
            float(np.sum(triangle_counts)) if len(triangle_counts) else np.nan
        ),
        "avg_triangles": (
            float(np.mean(triangle_counts)) if len(triangle_counts) else np.nan
        ),
        "avg_square_clustering": safe_metric(
            lambda: float(np.mean(list(nx.square_clustering(undirected_view).values())))
            if undirected_view.number_of_nodes()
            else np.nan
        ),
        "max_core_number": (
            float(np.max(core_numbers)) if len(core_numbers) else np.nan
        ),
        "avg_core_number": (
            float(np.mean(core_numbers)) if len(core_numbers) else np.nan
        ),
        "graph_clique_number": clique_number,
        "is_bipartite": safe_metric(
            lambda: float(nx.is_bipartite(undirected_view)), default=np.nan
        ),
        "global_efficiency": (
            safe_metric(lambda: nx.global_efficiency(undirected_view))
            if graph.number_of_nodes() <= MAX_NODES_FOR_PATHS
            else np.nan
        ),
        "local_efficiency": (
            safe_metric(lambda: nx.local_efficiency(undirected_view))
            if graph.number_of_nodes() <= MAX_NODES_FOR_PATHS
            else np.nan
        ),
        "avg_constraint": constraint,
        "avg_effective_size": effective_size,
    }


def centrality_stats(graph, directed: bool) -> dict:
    empty = {
        "avg_degree_centrality": np.nan,
        "avg_closeness_centrality": np.nan,
        "avg_betweenness_centrality": np.nan,
        "avg_pagerank": np.nan,
        "max_pagerank": np.nan,
        "avg_hub_score": np.nan,
        "avg_authority_score": np.nan,
    }
    if graph.number_of_nodes() == 0 or graph.number_of_nodes() > MAX_NODES_FOR_CENTRALITY:
        return empty

    undirected_view = graph.to_undirected() if directed else graph
    degree_centrality = list(nx.degree_centrality(graph).values())
    closeness = safe_metric(
        lambda: list(nx.closeness_centrality(undirected_view).values()),
        default=[],
    )
    betweenness = safe_metric(
        lambda: list(
            nx.betweenness_centrality(undirected_view, weight="weight").values()
        ),
        default=[],
    )
    pagerank = safe_metric(
        lambda: list(nx.pagerank(graph, weight="weight").values()),
        default=[],
    )

    result = {
        **summarize(degree_centrality, "degree_centrality"),
        **summarize(closeness, "closeness_centrality"),
        **summarize(betweenness, "betweenness_centrality"),
        **summarize(pagerank, "pagerank"),
    }

    if directed and graph.number_of_edges() > 0:
        try:
            hubs, authorities = nx.hits(graph, max_iter=200, normalized=True)
            result.update(summarize(list(hubs.values()), "hub_score"))
            result.update(summarize(list(authorities.values()), "authority_score"))
        except nx.PowerIterationFailedConvergence:
            result.update(summarize([], "hub_score"))
            result.update(summarize([], "authority_score"))
    else:
        result.update(
            {
                "hub_score_mean": np.nan,
                "hub_score_median": np.nan,
                "hub_score_std": np.nan,
                "hub_score_min": np.nan,
                "hub_score_max": np.nan,
                "authority_score_mean": np.nan,
                "authority_score_median": np.nan,
                "authority_score_std": np.nan,
                "authority_score_min": np.nan,
                "authority_score_max": np.nan,
            }
        )

    return result


def modularity_analysis_graph(graph):
    if graph.is_directed():
        undirected = nx.Graph()
        for source, target, data in graph.edges(data=True):
            weight = float(data.get("weight", 1.0))
            if undirected.has_edge(source, target):
                undirected[source][target]["weight"] += weight
            else:
                undirected.add_edge(source, target, weight=weight)
        return undirected
    return graph


def compute_partition_modularity_stats(
    graph,
    community_nodes: dict,
) -> tuple[float, float, float, dict]:
    modularity_graph = modularity_analysis_graph(graph)
    partition = [nodes for nodes in community_nodes.values() if nodes]

    graph_modularity = safe_metric(
        lambda: nx.community.modularity(
            modularity_graph, partition, weight="weight"
        ),
        default=np.nan,
    )
    coverage, performance = safe_metric(
        lambda: nx.community.partition_quality(modularity_graph, partition),
        default=(np.nan, np.nan),
    )

    total_weight = modularity_graph.size(weight="weight")
    if total_weight == 0:
        return graph_modularity, coverage, performance, {}

    weighted_degrees = dict(modularity_graph.degree(weight="weight"))
    contributions = {}

    for community, nodes in community_nodes.items():
        internal_weight = modularity_graph.subgraph(nodes).size(weight="weight")

        community_strength = sum(weighted_degrees.get(node, 0.0) for node in nodes)
        e_c = internal_weight / total_weight
        a_c = community_strength / (2 * total_weight)
        contributions[community] = e_c - a_c**2

    return graph_modularity, coverage, performance, contributions


def internal_weighted_strength(graph, node, community_nodes: set, directed: bool) -> float:
    strength = 0.0
    if directed:
        for _, target, data in graph.out_edges(node, data=True):
            if target in community_nodes:
                strength += float(data.get("weight", 1.0))
        for source, _, data in graph.in_edges(node, data=True):
            if source in community_nodes:
                strength += float(data.get("weight", 1.0))
        return strength

    for neighbor, data in graph[node].items():
        if neighbor in community_nodes:
            strength += float(data.get("weight", 1.0))
    return strength


def total_weighted_strength(graph, node, directed: bool) -> float:
    if directed:
        in_strength = sum(
            float(data.get("weight", 1.0))
            for _, _, data in graph.in_edges(node, data=True)
        )
        out_strength = sum(
            float(data.get("weight", 1.0))
            for _, _, data in graph.out_edges(node, data=True)
        )
        return in_strength + out_strength

    return float(graph.degree(node, weight="weight"))


def compute_node_role_stats(
    graph,
    community_nodes: dict,
    directed: bool,
) -> dict:
    node_to_community = {
        node: community
        for community, nodes in community_nodes.items()
        for node in nodes
    }
    community_sets = {
        community: set(nodes) for community, nodes in community_nodes.items()
    }

    participation_by_community = defaultdict(list)
    zscore_by_community = defaultdict(list)
    within_strength_by_community = defaultdict(list)

    for community, nodes in community_nodes.items():
        for node in nodes:
            total_strength = total_weighted_strength(graph, node, directed)
            within_strength = internal_weighted_strength(
                graph, node, community_sets[community], directed
            )
            within_strength_by_community[community].append(within_strength)

            if total_strength > 0:
                participation = 1.0 - (within_strength / total_strength) ** 2
            else:
                participation = 0.0
            participation_by_community[community].append(participation)

    for community, strengths in within_strength_by_community.items():
        arr = np.asarray(strengths, dtype=float)
        if arr.size == 0:
            continue
        mean_strength = arr.mean()
        std_strength = arr.std()
        if std_strength == 0:
            zscores = np.zeros_like(arr)
        else:
            zscores = (arr - mean_strength) / std_strength
        zscore_by_community[community].extend(zscores.tolist())

    role_stats = {}
    for community in community_nodes:
        role_stats[community] = {
            **summarize(
                participation_by_community[community],
                "participation_coefficient",
            ),
            **summarize(
                zscore_by_community[community],
                "within_module_degree_zscore",
            ),
            **summarize(
                within_strength_by_community[community],
                "within_module_strength",
            ),
        }
    return role_stats


def compute_boundary_stats(graph, communities: pd.DataFrame) -> pd.DataFrame:
    node_to_community = dict(
        zip(communities["node"], communities["community"])
    )
    community_nodes = communities.groupby("community")["node"].apply(set).to_dict()

    internal_edges = {community: 0 for community in community_nodes}
    internal_weight = {community: 0.0 for community in community_nodes}
    internal_phishing_weight = {community: 0.0 for community in community_nodes}
    boundary_edges = {community: 0 for community in community_nodes}
    boundary_weight = {community: 0.0 for community in community_nodes}
    boundary_phishing_weight = {community: 0.0 for community in community_nodes}

    for source, target, data in graph.edges(data=True):
        source_community = node_to_community.get(source)
        target_community = node_to_community.get(target)
        if source_community is None or target_community is None:
            continue

        weight = float(data.get("weight", 1.0))
        phishing_weight = float(data.get("phishing_tx_count", 0.0))

        if source_community == target_community:
            internal_edges[source_community] += 1
            internal_weight[source_community] += weight
            internal_phishing_weight[source_community] += phishing_weight
        else:
            boundary_edges[source_community] += 1
            boundary_edges[target_community] += 1
            boundary_weight[source_community] += weight
            boundary_weight[target_community] += weight
            boundary_phishing_weight[source_community] += phishing_weight
            boundary_phishing_weight[target_community] += phishing_weight

    rows = []
    for community, nodes in community_nodes.items():
        internal = internal_edges[community]
        boundary = boundary_edges[community]
        total_incident = internal + boundary
        rows.append(
            {
                "community": community,
                "internal_edge_count": internal,
                "boundary_edge_count": boundary,
                "internal_edge_fraction": (
                    internal / total_incident if total_incident else np.nan
                ),
                "boundary_edge_fraction": (
                    boundary / total_incident if total_incident else np.nan
                ),
                "expansion": boundary / len(nodes) if nodes else np.nan,
                "internal_edge_weight": internal_weight[community],
                "boundary_edge_weight": boundary_weight[community],
                "internal_phishing_edge_weight": internal_phishing_weight[community],
                "boundary_phishing_edge_weight": boundary_phishing_weight[community],
                "conductance": (
                    boundary_weight[community]
                    / (2 * internal_weight[community] + boundary_weight[community])
                    if (2 * internal_weight[community] + boundary_weight[community])
                    else np.nan
                ),
                "normalized_cut": (
                    boundary_weight[community]
                    / (internal_weight[community] + boundary_weight[community])
                    if (internal_weight[community] + boundary_weight[community])
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_subgroup_features(
    graph,
    communities: pd.DataFrame,
    algorithm: str,
    graph_variant: str,
    directed: bool,
) -> pd.DataFrame:
    community_nodes = communities.groupby("community")["node"].apply(list).to_dict()
    boundary_stats = compute_boundary_stats(graph, communities).set_index("community")

    graph_modularity, partition_coverage, partition_performance, modularity_contribs = (
        compute_partition_modularity_stats(graph, community_nodes)
    )
    node_role_stats = compute_node_role_stats(graph, community_nodes, directed)

    rows = []
    for community, nodes in sorted(community_nodes.items()):
        subgraph = graph.subgraph(nodes).copy()
        num_nodes = subgraph.number_of_nodes()
        num_edges = subgraph.number_of_edges()
        num_isolates = len(list(nx.isolates(subgraph)))

        row = {
            "community": community,
            "algorithm": algorithm,
            "graph_variant": graph_variant,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "density": safe_metric(lambda: nx.density(subgraph), default=0.0),
            "num_isolates": num_isolates,
            "isolate_fraction": num_isolates / num_nodes if num_nodes else np.nan,
            "graph_modularity": graph_modularity,
            "partition_coverage": partition_coverage,
            "partition_performance": partition_performance,
            "modularity_contribution": modularity_contribs.get(community, np.nan),
        }
        row.update(degree_stats(subgraph, directed))
        row.update(edge_weight_stats(subgraph))
        row.update(connectivity_stats(subgraph, directed))
        row.update(path_stats(subgraph, directed))
        row.update(structural_stats(subgraph, directed))
        row.update(centrality_stats(subgraph, directed))
        row.update(node_role_stats.get(community, {}))
        row.update(boundary_stats.loc[community].to_dict())
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    config = ALGORITHM_CONFIG[args.algorithm]

    graph = load_graph(misc_dir / config["edges_file"], directed=config["directed"])
    communities = load_communities(args.algorithm)

    features = compute_subgroup_features(
        graph=graph,
        communities=communities,
        algorithm=args.algorithm,
        graph_variant=config["graph_variant"],
        directed=config["directed"],
    )
    
    print(features.columns)
    
    features_to_keep = [
        'community',
        'algorithm',
        'graph_variant',
        # 
        'num_nodes',
        'num_edges',
        'density',
        'num_isolates',
        'isolate_fraction',
        'graph_modularity',
        'avg_degree',
        "median_degree",
        "std_degree",   
        "min_degree",
        "max_degree",
        "avg_weighted_degree",
        "avg_in_degree",
        "avg_out_degree",
        "avg_weighted_in_degree",
        "avg_weighted_out_degree",
        "total_edge_weight",
        "avg_edge_weight",
        "median_edge_weight",
        "max_edge_weight",
        "num_weakly_connected_components",
        "num_strongly_connected_components",
        "largest_weak_component_size",
        "largest_weak_component_fraction",
        "largest_strong_component_size",
        "largest_strong_component_fraction",
        "diameter",
        "avg_shortest_path_length",
        "avg_clustering",
        "avg_weighted_clustering",
        "degree_assortativity",
        "degree_assortativity_weighted",
        "spectral_radius",
        "total_triangles",
        "avg_triangles",
        "degree_centrality_mean",
        "degree_centrality_median",
        "degree_centrality_std",
        "degree_centrality_min",
        "degree_centrality_max",
        "closeness_centrality_mean",
        "closeness_centrality_median",
        "closeness_centrality_std",
        "closeness_centrality_min",
        "closeness_centrality_max",
        "betweenness_centrality_mean",
        "betweenness_centrality_median",
        "betweenness_centrality_std",
        "betweenness_centrality_max",
        "pagerank_mean",
        "pagerank_median",
        "pagerank_std",
        "pagerank_min",
        "pagerank_max",
    ]
    
    
    features = features[features_to_keep]
    # exit()

    features_out = output_dir / f"{args.algorithm}_subgroup_features.csv"
    features.to_csv(features_out, index=False)

    print(f"{args.algorithm.title()} subgroup features complete.")
    print(f"Communities: {len(features)}")
    print(f"Features: {len(features.columns)}")
    print(f"Wrote features to: {features_out}")


if __name__ == "__main__":
    main()
