from pathlib import Path
import pandas as pd

# Folder where this script is 
project_dir = Path(__file__).resolve().parent


misc_dir = project_dir / "output" / "misc"
assert misc_dir.exists(), "Misc directory does not exist"

louvain_dir = project_dir / "output" / "algorithms" / "louvain"
# louvain_dir.mkdir(parents=True, exist_ok=True)
assert louvain_dir.exists(), "Louvain directory does not exist"

# Input files
transactions_file = misc_dir / "transactions_clean.csv"
node_labels_file = misc_dir / "node_labels.csv"

communities_file = louvain_dir / "louvain_communities.csv"

# Output files
evaluation_out = louvain_dir / "louvain_evaluation.csv"
top_communities_out = louvain_dir / "louvain_top_communities.csv"


def load_inputs():
    """
    Load the cleaned transactions, derived node labels, and Louvain communities.
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
    Add source and target community IDs to every transaction;
    This lets us check whether each transaction is internal to a community or crosses between two different communities.
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
    Suggested metric:
    It measures how often transaction endpoints belong to the same community, separately for phishing-labelled and non-phishing-labelled transactions.
    """
    phishing_tx = tx[tx["class"] == 1]
    non_phishing_tx = tx[tx["class"] == 0]

    phishing_internality = phishing_tx["same_community"].mean()
    non_phishing_internality = non_phishing_tx["same_community"].mean()
    all_transaction_internality = tx["same_community"].mean()

    return {
        "phishing_transaction_internality": phishing_internality,
        "non_phishing_transaction_internality": non_phishing_internality,
        "all_transaction_internality": all_transaction_internality,
    }


def compute_community_node_statistics(node_labels, communities):
    """
    Compute phishing-incident address concentration per community.
    A node is phishing-incident if it participates in at least one transaction with class = 1.
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
            total_phishing_incident_transactions=("phishing_tx_incident_count", "sum"),
            total_incident_transactions=("total_tx_incident_count", "sum"),
        )
    )

    community_stats["phishing_node_rate"] = (
        community_stats["phishing_incident_nodes"]
        / community_stats["community_size"]
    )

    global_phishing_node_rate = (
        node_data["is_phishing"].sum() / len(node_data)
    )

    community_stats["phishing_enrichment"] = (
        community_stats["phishing_node_rate"] / global_phishing_node_rate
    )

    return community_stats, global_phishing_node_rate


def compute_top_k_concentration(community_stats):
    """
    Measure whether phishing-incident addresses are concentrated in few communities.
    Communities are sorted by the number of phishing-incident nodes.
    Then we compute the share of all phishing-incident nodes contained in the top 1%, 5%, and 10% of communities.
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


def save_top_communities(community_stats, limit=20):
    """
    Save the most phishing-enriched communities.
    Very small communities can have extreme enrichment, so the output includes both enrichment and raw community size.
    """
    top = community_stats.sort_values(
        ["phishing_enrichment", "phishing_incident_nodes"],
        ascending=False,
    ).head(limit)

    top.to_csv(top_communities_out, index=False)


def save_evaluation(
    tx,
    community_stats,
    global_phishing_node_rate,
    transaction_internality,
    top_k_results,
):
    """
    Save one-row Louvain evaluation summary.
    """
    num_communities = len(community_stats)

    summary = {
        "algorithm": "louvain",
        "graph_variant": "undirected_count",
        "num_transactions": len(tx),
        "num_phishing_transactions": int((tx["class"] == 1).sum()),
        "num_non_phishing_transactions": int((tx["class"] == 0).sum()),
        "num_communities": num_communities,
        "global_phishing_node_rate": global_phishing_node_rate,
        "max_phishing_enrichment": community_stats["phishing_enrichment"].max(),
        "median_phishing_enrichment": community_stats["phishing_enrichment"].median(),
        "communities_with_phishing_nodes": int(
            (community_stats["phishing_incident_nodes"] > 0).sum()
        ),
    }

    summary.update(transaction_internality)
    summary.update(top_k_results)

    pd.DataFrame([summary]).to_csv(evaluation_out, index=False)


def main():
    """
    Complete Louvain post-hoc evaluation.
    Labels are used here, after community detection, to study how phishing activity is distributed over the Louvain communities.
    """
    transactions, node_labels, communities = load_inputs()

    tx = attach_communities_to_transactions(transactions, communities)

    transaction_internality = compute_transaction_internality(tx)

    community_stats, global_phishing_node_rate = compute_community_node_statistics(
        node_labels,
        communities,
    )

    top_k_results = compute_top_k_concentration(community_stats)

    save_top_communities(community_stats)

    save_evaluation(
        tx=tx,
        community_stats=community_stats,
        global_phishing_node_rate=global_phishing_node_rate,
        transaction_internality=transaction_internality,
        top_k_results=top_k_results,
    )

    print("Louvain evaluation complete.")
    print(f"Wrote evaluation summary to: {evaluation_out}")
    print(f"Wrote top communities to: {top_communities_out}")
    print()
    print("Key results:")
    print(f"Phishing transaction internality: {transaction_internality['phishing_transaction_internality']:.4f}")
    print(f"Non-phishing transaction internality: {transaction_internality['non_phishing_transaction_internality']:.4f}")
    print(f"Global phishing-incident node rate: {global_phishing_node_rate:.4f}")
    print(f"Max phishing enrichment: {community_stats['phishing_enrichment'].max():.4f}")


if __name__ == "__main__":
    main()