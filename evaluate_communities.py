import argparse
from pathlib import Path

import pandas as pd


project_dir = Path(__file__).resolve().parent

misc_dir = project_dir / "output" / "misc"
assert misc_dir.exists(), "Misc directory does not exist"

algorithms_dir = project_dir / "output" / "algorithms"
assert algorithms_dir.exists(), "Algorithms directory does not exist"

transactions_file = misc_dir / "transactions_clean.csv"
node_labels_file = misc_dir / "node_labels.csv"

DEFAULT_GRAPH_VARIANTS = {
    "louvain": "undirected_count",
    "infomap": "directed_count",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate how labelled suspicious transactions and accounts are "
            "distributed over graph communities."
        )
    )
    parser.add_argument(
        "--algorithm",
        default="infomap",
        help="Algorithm directory name in output/algorithms/.",
    )
    parser.add_argument(
        "--graph-variant",
        default=None,
        help="Graph variant label to write in the evaluation output.",
    )
    parser.add_argument(
        "--top-limit",
        type=int,
        default=20,
        help="Number of phishing-enriched communities to save.",
    )
    
    parser.add_argument(
        "--min-community-size",
        type=int,
        default=50,
        help="Minimum community size to include in the evaluation.",
    )

    return parser.parse_args()


def get_algorithm_paths(algorithm: str) -> tuple[Path, Path, Path]:
    algorithm_dir = algorithms_dir / algorithm
    if not algorithm_dir.exists():
        raise FileNotFoundError(f"Missing algorithm directory: {algorithm_dir}")

    communities_file = algorithm_dir / f"{algorithm}_communities.csv"
    evaluation_out = algorithm_dir / f"{algorithm}_evaluation.csv"
    top_communities_out = algorithm_dir / f"{algorithm}_top_communities.csv"

    return communities_file, evaluation_out, top_communities_out


def load_inputs(communities_file: Path):
    """
    Load cleaned transactions, derived node labels, and community assignments.
    """
    for path in [transactions_file, node_labels_file, communities_file]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    transactions = pd.read_csv(transactions_file)
    node_labels = pd.read_csv(node_labels_file)
    communities = pd.read_csv(communities_file)

    return transactions, node_labels, communities


def attach_communities_to_transactions(transactions, communities):
    """
    Add source and target community IDs to every transaction.
    """
    source_communities = communities[["node", "community"]].rename(
        columns={
            "node": "source",
            "community": "source_community",
        }
    )

    target_communities = communities[["node", "community"]].rename(
        columns={
            "node": "target",
            "community": "target_community",
        }
    )

    tx = transactions.merge(source_communities, on="source", how="left")
    tx = tx.merge(target_communities, on="target", how="left")

    missing_source = tx["source_community"].isna().sum()
    missing_target = tx["target_community"].isna().sum()

    if missing_source > 0 or missing_target > 0:
        raise ValueError(
            f"Some transaction endpoints have no community assignment. "
            f"Missing source communities: {missing_source}. "
            f"Missing target communities: {missing_target}."
        )

    tx["same_community"] = tx["source_community"] == tx["target_community"]

    return tx


def compute_transaction_internality(tx):
    """
    Measure how often transaction endpoints are in the same community.
    """
    phishing_tx = tx[tx["class"] == 1]
    non_phishing_tx = tx[tx["class"] == 0]

    return {
        "phishing_transaction_internality": phishing_tx["same_community"].mean(),
        "non_phishing_transaction_internality": (
            non_phishing_tx["same_community"].mean()
        ),
        "all_transaction_internality": tx["same_community"].mean(),
    }


def compute_community_node_statistics(node_labels, communities):
    """
    Compute suspicious account concentration per community.
    """
    node_data = node_labels.merge(
        communities[["node", "community"]],
        on="node",
        how="left",
    )

    missing = node_data["community"].isna().sum()

    if missing > 0:
        raise ValueError(f"Some labelled nodes have no community assignment: {missing}")

    community_stats = (
        node_data.groupby("community", as_index=False)
        .agg(
            community_size=("node", "count"),
            phishing_incident_nodes=("is_phishing", "sum"),
            total_phishing_incident_transactions=(
                "phishing_tx_incident_count",
                "sum",
            ),
            total_incident_transactions=("total_tx_incident_count", "sum"),
        )
    )

    community_stats["phishing_node_rate"] = (
        community_stats["phishing_incident_nodes"]
        / community_stats["community_size"]
    )

    global_phishing_node_rate = node_data["is_phishing"].sum() / len(node_data)

    community_stats["phishing_enrichment"] = (
        community_stats["phishing_node_rate"] / global_phishing_node_rate
    )

    return community_stats, global_phishing_node_rate


def compute_top_k_concentration(community_stats):
    """
    Measure whether phishing-incident accounts concentrate in few communities.
    """
    sorted_stats = community_stats.sort_values(
        "phishing_incident_nodes",
        ascending=False,
    ).reset_index(drop=True)

    total_phishing_nodes = sorted_stats["phishing_incident_nodes"].sum()
    num_communities = len(sorted_stats)

    results = {}

    for label, fraction in [
        ("top_1pct_phishing_node_coverage", 0.01),
        ("top_5pct_phishing_node_coverage", 0.05),
        ("top_10pct_phishing_node_coverage", 0.10),
    ]:
        top_n = max(1, int(num_communities * fraction))
        covered = sorted_stats.head(top_n)["phishing_incident_nodes"].sum()

        if total_phishing_nodes == 0:
            results[label] = 0
        else:
            results[label] = covered / total_phishing_nodes

    return results


def build_top_communities(community_stats, limit=20, min_community_size=50):
    """
    Select the most phishing-enriched communities.
    """
    return community_stats.sort_values(
        ["phishing_enrichment", "phishing_incident_nodes"],
        ascending=False,
    ).query(f"community_size >= {min_community_size}").head(limit)


def build_evaluation(
    algorithm,
    graph_variant,
    tx,
    community_stats,
    global_phishing_node_rate,
    transaction_internality,
    top_k_results,
):
    """
    Build one-row evaluation summary for algorithm comparison.
    """
    summary = {
        "algorithm": algorithm,
        "graph_variant": graph_variant,
        "num_transactions": len(tx),
        "num_phishing_transactions": int((tx["class"] == 1).sum()),
        "num_non_phishing_transactions": int((tx["class"] == 0).sum()),
        "num_communities": len(community_stats),
        "global_phishing_node_rate": global_phishing_node_rate,
        "max_phishing_enrichment": community_stats["phishing_enrichment"].max(),
        "median_phishing_enrichment": community_stats[
            "phishing_enrichment"
        ].median(),
        "communities_with_phishing_nodes": int(
            (community_stats["phishing_incident_nodes"] > 0).sum()
        ),
    }

    summary.update(transaction_internality)
    summary.update(top_k_results)

    return pd.DataFrame([summary])


def main():
    args = parse_args()
    algorithm = args.algorithm.lower()
    graph_variant = args.graph_variant or DEFAULT_GRAPH_VARIANTS.get(
        algorithm,
        "unknown",
    )

    communities_file, evaluation_out, top_communities_out = get_algorithm_paths(
        algorithm
    )
    transactions, node_labels, communities = load_inputs(communities_file)

    tx = attach_communities_to_transactions(transactions, communities)
    transaction_internality = compute_transaction_internality(tx)
    community_stats, global_phishing_node_rate = compute_community_node_statistics(
        node_labels,
        communities,
    )
    top_k_results = compute_top_k_concentration(community_stats)

    top_communities = build_top_communities(
        community_stats,
        limit=args.top_limit,
        min_community_size=args.min_community_size,
    )
    evaluation = build_evaluation(
        algorithm=algorithm,
        graph_variant=graph_variant,
        tx=tx,
        community_stats=community_stats,
        global_phishing_node_rate=global_phishing_node_rate,
        transaction_internality=transaction_internality,
        top_k_results=top_k_results,
    )

    top_communities.to_csv(top_communities_out, index=False)
    evaluation.to_csv(evaluation_out, index=False)

    print(f"{algorithm.title()} evaluation complete.")
    print(f"Wrote evaluation summary to: {evaluation_out}")
    print(f"Wrote top communities to: {top_communities_out}")
    print()
    print("Key results:")
    print(
        "Phishing transaction internality: "
        f"{transaction_internality['phishing_transaction_internality']:.4f}"
    )
    print(
        "Non-phishing transaction internality: "
        f"{transaction_internality['non_phishing_transaction_internality']:.4f}"
    )
    print(f"Global phishing-incident node rate: {global_phishing_node_rate:.4f}")
    print(
        "Max phishing enrichment: "
        f"{community_stats['phishing_enrichment'].max():.4f}"
    )


if __name__ == "__main__":
    main()
