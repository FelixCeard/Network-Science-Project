from __future__ import annotations

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

project_dir = Path(__file__).resolve().parent

misc_dir = project_dir / "output" / "misc"
assert misc_dir.exists(), "Misc directory does not exist"

leiden_dir = project_dir / "output" / "algorithms" / "leiden"
assert leiden_dir.exists(), (
    "Leiden directory does not exist.  Run run_leiden.py first."
)

# Inputs produced by build_graphs.py
transactions_file = misc_dir / "transactions_clean.csv"
node_labels_file = misc_dir / "node_labels.csv"

# Input produced by run_leiden.py
communities_file = leiden_dir / "leiden_communities.csv"

# Outputs produced by this script
evaluation_out = leiden_dir / "leiden_evaluation.csv"
top_communities_out = leiden_dir / "leiden_top_communities.csv"

# Minimum community size used for the top-communities table.
# Very small communities (< MIN_COMMUNITY_SIZE nodes) can have high enrichment
# simply because one phishing-incident node dominates the rate.  The size
# filter suppresses this noise without discarding the data from the full
# evaluation metrics.
MIN_COMMUNITY_SIZE = 50

# How many top communities to write to the output table.
TOP_LIMIT = 20


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the cleaned transactions, derived node labels, and Leiden communities.

    Returns
    -------
    transactions : DataFrame with columns source, target, class, …
    node_labels  : DataFrame with columns node, is_phishing,
                   phishing_tx_incident_count, total_tx_incident_count
    communities  : DataFrame with columns node, algorithm, graph_variant,
                   community
    """
    for path in [transactions_file, node_labels_file, communities_file]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    transactions = pd.read_csv(transactions_file)
    node_labels = pd.read_csv(node_labels_file)
    communities = pd.read_csv(communities_file)

    return transactions, node_labels, communities


# ---------------------------------------------------------------------------
# Community structure
# ---------------------------------------------------------------------------


def compute_community_structure(communities: pd.DataFrame) -> dict:
    """
    Summarise the size distribution of the Leiden partition.

    Metrics
    -------
    num_communities       : total number of distinct communities.
    median_community_size : median number of nodes per community.
    largest_community_size: nodes in the single largest community.
    singleton_communities : communities containing exactly one node.
    communities_ge_50     : communities with ≥ 50 nodes.
    skew_ratio            : largest_community_size / median_community_size.
                            Values >> 1 indicate a heavily skewed partition.
    """
    sizes = communities.groupby("community")["node"].count()

    return {
        "num_communities": int(sizes.size),
        "median_community_size": float(sizes.median()),
        "largest_community_size": int(sizes.max()),
        "singleton_communities": int((sizes == 1).sum()),
        "communities_ge_50": int((sizes >= 50).sum()),
        "skew_ratio": float(sizes.max() / sizes.median()) if sizes.median() > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Transaction internality
# ---------------------------------------------------------------------------


def attach_communities_to_transactions(
    transactions: pd.DataFrame,
    communities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add source and target community IDs to every transaction.
    A transaction is *internal* when both endpoints belong to the same
    community (same_community == True).
    """
    source_map = communities[["node", "community"]].rename(
        columns={"node": "source", "community": "source_community"}
    )
    target_map = communities[["node", "community"]].rename(
        columns={"node": "target", "community": "target_community"}
    )

    tx = transactions.merge(source_map, on="source", how="left")
    tx = tx.merge(target_map, on="target", how="left")

    missing_source = tx["source_community"].isna().sum()
    missing_target = tx["target_community"].isna().sum()

    if missing_source > 0 or missing_target > 0:
        raise ValueError(
            f"Some transaction endpoints have no community assignment. "
            f"Missing source: {missing_source}.  Missing target: {missing_target}."
        )

    tx["same_community"] = tx["source_community"] == tx["target_community"]

    return tx


def compute_transaction_internality(tx: pd.DataFrame) -> dict:
    """
    Measure how often transaction endpoints reside in the same community,
    separately for phishing-labelled (class = 1) and non-phishing-labelled
    (class = 0) transactions.

    A higher phishing internality relative to the non-phishing baseline
    suggests that phishing activity is more localised within communities than
    general activity.  A lower value suggests phishing activity crosses
    community boundaries more often.
    """
    phishing_tx = tx[tx["class"] == 1]
    non_phishing_tx = tx[tx["class"] == 0]

    return {
        "phishing_transaction_internality": float(
            phishing_tx["same_community"].mean()
        ),
        "non_phishing_transaction_internality": float(
            non_phishing_tx["same_community"].mean()
        ),
        "all_transaction_internality": float(tx["same_community"].mean()),
    }


# ---------------------------------------------------------------------------
# Community-level phishing statistics
# ---------------------------------------------------------------------------


def compute_community_node_statistics(
    node_labels: pd.DataFrame,
    communities: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """
    Compute per-community phishing-incident address counts and enrichment.

    Enrichment is defined as:
        phishing_enrichment = (phishing_incident_nodes / community_size)
                              / global_phishing_node_rate

    A value of 1.0 means the community's phishing rate equals the global
    baseline.  A value of 5.0 means five times the baseline rate.

    Returns
    -------
    community_stats          : DataFrame with one row per community.
    global_phishing_node_rate: fraction of all nodes that are phishing-incident.
    """
    node_data = node_labels.merge(
        communities[["node", "community"]],
        on="node",
        how="left",
    )

    missing = node_data["community"].isna().sum()
    if missing > 0:
        raise ValueError(
            f"Some labelled nodes have no community assignment: {missing}"
        )

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

    global_phishing_node_rate = float(
        node_data["is_phishing"].sum() / len(node_data)
    )

    community_stats["phishing_enrichment"] = (
        community_stats["phishing_node_rate"] / global_phishing_node_rate
    )

    return community_stats, global_phishing_node_rate


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------


def compute_top_k_concentration(community_stats: pd.DataFrame) -> dict:
    """
    Measure whether phishing-incident addresses are concentrated in a small
    number of communities.

    Communities are sorted by phishing_incident_nodes (descending).  The
    metric reports what fraction of all phishing-incident nodes are contained
    in the top 1 %, 5 %, and 10 % of communities by count.

    High concentration (e.g., top-5 % covers > 80 % of phishing nodes) means
    that a small subset of communities accounts for most labelled activity.
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
        results[label] = (
            float(covered / total_phishing_nodes)
            if total_phishing_nodes > 0
            else 0.0
        )

    return results


# ---------------------------------------------------------------------------
# Top communities table
# ---------------------------------------------------------------------------


def build_top_communities(
    community_stats: pd.DataFrame,
    limit: int = TOP_LIMIT,
    min_community_size: int = MIN_COMMUNITY_SIZE,
) -> pd.DataFrame:
    """
    Return the most phishing-enriched communities with at least
    min_community_size nodes.

    The size filter is applied because single-node or very small communities
    can reach extreme enrichment values (e.g., 1 phishing node out of 2 total
    → rate = 0.5, vs. global rate ≈ 0.035 → enrichment ≈ 14×) that are
    artefacts of sample size rather than genuine structural signals.

    The primary sort is phishing_enrichment; the secondary sort is
    phishing_incident_nodes so that large communities with similar enrichment
    are ranked above smaller ones.
    """
    return (
        community_stats
        .query(f"community_size >= {min_community_size}")
        .sort_values(
            ["phishing_enrichment", "phishing_incident_nodes"],
            ascending=False,
        )
        .head(limit)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Evaluation summary
# ---------------------------------------------------------------------------


def build_evaluation(
    tx: pd.DataFrame,
    community_stats: pd.DataFrame,
    community_structure: dict,
    global_phishing_node_rate: float,
    transaction_internality: dict,
    top_k_results: dict,
) -> pd.DataFrame:
    """
    Build a one-row evaluation summary for algorithm comparison.
    """
    summary = {
        "algorithm": "leiden",
        "graph_variant": "undirected_count",
        "num_transactions": len(tx),
        "num_phishing_transactions": int((tx["class"] == 1).sum()),
        "num_non_phishing_transactions": int((tx["class"] == 0).sum()),
        "global_phishing_node_rate": global_phishing_node_rate,
        "max_phishing_enrichment": float(
            community_stats["phishing_enrichment"].max()
        ),
        "median_phishing_enrichment": float(
            community_stats["phishing_enrichment"].median()
        ),
        "communities_with_phishing_nodes": int(
            (community_stats["phishing_incident_nodes"] > 0).sum()
        ),
    }

    summary.update(community_structure)
    summary.update(transaction_internality)
    summary.update(top_k_results)

    return pd.DataFrame([summary])


# ---------------------------------------------------------------------------
# Interpretive narrative
# ---------------------------------------------------------------------------


def print_interpretation(
    community_structure: dict,
    transaction_internality: dict,
    top_k_results: dict,
    community_stats: pd.DataFrame,
    global_phishing_node_rate: float,
) -> None:
    """
    Print a structured interpretation of the Leiden results covering all five
    aspects requested:

      1. Community structure
      2. Internal consistency of labelled activity
      3. Distribution of phishing-incident addresses
      4. Enrichment patterns
      5. Interpretation and caveats

    All statements are framed as structural observations about the distribution
    of labelled activity.  Leiden is not presented as detecting phishing.
    """
    n_comm = community_structure["num_communities"]
    median_size = community_structure["median_community_size"]
    largest = community_structure["largest_community_size"]
    skew = community_structure["skew_ratio"]
    singletons = community_structure["singleton_communities"]
    large_comms = community_structure["communities_ge_50"]

    phi_int = transaction_internality["phishing_transaction_internality"]
    non_phi_int = transaction_internality["non_phishing_transaction_internality"]
    all_int = transaction_internality["all_transaction_internality"]

    cov_1 = top_k_results["top_1pct_phishing_node_coverage"]
    cov_5 = top_k_results["top_5pct_phishing_node_coverage"]
    cov_10 = top_k_results["top_10pct_phishing_node_coverage"]

    max_enrich = community_stats["phishing_enrichment"].max()
    median_enrich = community_stats["phishing_enrichment"].median()

    # Identify most-enriched large community (size >= MIN_COMMUNITY_SIZE).
    large_mask = community_stats["community_size"] >= MIN_COMMUNITY_SIZE
    if large_mask.any():
        top_large = (
            community_stats[large_mask]
            .sort_values("phishing_enrichment", ascending=False)
            .iloc[0]
        )
        top_large_enrich = top_large["phishing_enrichment"]
        top_large_size = int(top_large["community_size"])
        top_large_phi_nodes = int(top_large["phishing_incident_nodes"])
    else:
        top_large_enrich = float("nan")
        top_large_size = 0
        top_large_phi_nodes = 0

    # -----------------------------------------------------------------------
    sep = "-" * 72

    print()
    print("=" * 72)
    print("  LEIDEN EVALUATION — INTERPRETIVE SUMMARY")
    print("=" * 72)

    # 1. Community structure
    print()
    print(sep)
    print("1. COMMUNITY STRUCTURE")
    print(sep)
    print(
        f"Leiden produced {n_comm:,} communities from the undirected "
        "count-weighted graph."
    )
    print(
        f"Median community size: {median_size:.1f} nodes.  "
        f"Largest community: {largest:,} nodes."
    )
    print(
        f"Skew ratio (largest / median): {skew:.1f}x.  "
        f"Singleton communities: {singletons:,}.  "
        f"Communities with ≥ 50 nodes: {large_comms:,}."
    )
    if skew > 20:
        skew_label = "heavily skewed"
        skew_note = (
            "A small number of large communities co-exist with many small or "
            "singleton communities, which is typical for scale-free transaction "
            "networks."
        )
    elif skew > 5:
        skew_label = "moderately skewed"
        skew_note = (
            "The partition is moderately uneven, with some large communities "
            "and a long tail of smaller ones."
        )
    else:
        skew_label = "relatively balanced"
        skew_note = (
            "The partition is comparatively even, with communities of broadly "
            "similar size."
        )
    print(f"The partition is {skew_label}. {skew_note}")

    # 2. Transaction internality
    print()
    print(sep)
    print("2. INTERNAL CONSISTENCY OF LABELLED ACTIVITY")
    print(sep)
    print(
        f"Overall transaction internality:          {all_int:.4f} "
        f"({all_int * 100:.1f}% of transactions are within a single community)"
    )
    print(
        f"Phishing-labelled (class=1) internality:  {phi_int:.4f} "
        f"({phi_int * 100:.1f}%)"
    )
    print(
        f"Non-phishing (class=0) internality:       {non_phi_int:.4f} "
        f"({non_phi_int * 100:.1f}%)"
    )
    delta = phi_int - non_phi_int
    if abs(delta) < 0.01:
        print(
            "Phishing and non-phishing transactions have similar internality "
            f"(Δ = {delta:+.4f}). Phishing-labelled activity is neither more "
            "nor less community-contained than the baseline."
        )
    elif delta > 0:
        print(
            f"Phishing-labelled transactions are more likely to stay within a "
            f"single community than non-phishing transactions (Δ = {delta:+.4f}). "
            "This suggests that phishing-incident addresses tend to form "
            "structurally cohesive local clusters."
        )
    else:
        print(
            f"Phishing-labelled transactions cross community boundaries more "
            f"often than non-phishing transactions (Δ = {delta:+.4f}). "
            "This may indicate that phishing activity in this dataset involves "
            "addresses that span multiple detected communities."
        )

    # 3. Distribution of phishing-incident addresses
    print()
    print(sep)
    print("3. DISTRIBUTION OF PHISHING-INCIDENT ADDRESSES")
    print(sep)
    print(
        f"Global phishing-incident address rate: "
        f"{global_phishing_node_rate:.4f} "
        f"({global_phishing_node_rate * 100:.2f}% of all addresses)"
    )
    print(
        f"Top  1% of communities (by phishing-node count) contain "
        f"{cov_1 * 100:.1f}% of all phishing-incident addresses."
    )
    print(
        f"Top  5% of communities contain "
        f"{cov_5 * 100:.1f}% of all phishing-incident addresses."
    )
    print(
        f"Top 10% of communities contain "
        f"{cov_10 * 100:.1f}% of all phishing-incident addresses."
    )
    if cov_10 >= 0.80:
        conc_label = "strongly concentrated"
        conc_note = (
            "The large majority of phishing-incident addresses appear in a "
            "small fraction of communities, suggesting that phishing-linked "
            "activity is structurally isolated within specific dense subgraphs."
        )
    elif cov_10 >= 0.50:
        conc_label = "moderately concentrated"
        conc_note = (
            "A meaningful fraction of phishing-incident addresses appear in a "
            "small number of communities, though a substantial portion is also "
            "spread across the rest of the graph."
        )
    else:
        conc_label = "broadly distributed"
        conc_note = (
            "Phishing-incident addresses are spread relatively evenly across "
            "communities, which may reflect the cross-community nature of the "
            "transaction network rather than tight local clustering of phishing "
            "activity."
        )
    print(
        f"Overall, phishing-incident addresses are {conc_label}. {conc_note}"
    )

    # 4. Enrichment patterns
    print()
    print(sep)
    print("4. ENRICHMENT PATTERNS")
    print(sep)
    print(
        f"Maximum phishing enrichment (all communities): {max_enrich:.2f}x "
        "the global baseline."
    )
    print(
        f"Median phishing enrichment (all communities):  {median_enrich:.2f}x."
    )
    print()
    print(
        "NOTE ON SMALL-COMMUNITY ENRICHMENT: Communities with very few nodes "
        "can produce extreme enrichment values purely because of small-sample "
        "effects.  For example, a community of 5 nodes that contains 1 "
        "phishing-incident address has a local rate of 20%, which at a global "
        f"rate of ~{global_phishing_node_rate * 100:.1f}% yields an enrichment "
        f"of ~{0.20 / global_phishing_node_rate:.0f}x.  This number is "
        "arithmetically correct but not structurally meaningful."
    )
    print()
    if not large_comms:
        print(
            "No communities with ≥ 50 nodes were found, so size-filtered "
            "enrichment statistics cannot be reported."
        )
    else:
        print(
            f"Among communities with ≥ {MIN_COMMUNITY_SIZE} nodes "
            f"({large_comms:,} communities), the highest enrichment is "
            f"{top_large_enrich:.2f}x, found in a community of "
            f"{top_large_size:,} nodes containing "
            f"{top_large_phi_nodes:,} phishing-incident addresses."
        )
        if top_large_enrich >= 5.0:
            print(
                "This represents a substantial elevation relative to the "
                "global baseline and is worth examining in detail."
            )
        elif top_large_enrich >= 2.0:
            print(
                "This represents a moderate elevation relative to the global "
                "baseline."
            )
        else:
            print(
                "Even the most-enriched large community is close to the "
                "global baseline, suggesting that phishing activity does not "
                "preferentially concentrate in any single large structural unit."
            )

    # 5. Interpretation
    print()
    print(sep)
    print("5. INTERPRETATION")
    print(sep)
    print(
        "Leiden is a community detection algorithm that optimises a graph "
        "quality function (ModularityVertexPartition by default, or a CPM "
        "variant when a resolution parameter is supplied).  It partitions "
        "addresses into cohesive structural groups by finding boundaries "
        "where the density of transactions drops relative to a random "
        "baseline.  Leiden does not use labels at any stage and is not "
        "designed for anomaly detection."
    )
    print()
    print(
        "The observations above describe how phishing-labelled activity "
        "happens to be distributed over communities that were formed entirely "
        "from graph topology.  Any association is exploratory and descriptive: "
        "it characterises the structural context of labelled accounts and "
        "transactions, not a predictive signal."
    )
    print()
    print(
        "Practical implications of the observed structure:"
    )
    if phi_int > non_phi_int + 0.02:
        print(
            "  • Phishing-incident transactions are more community-internal "
            "than non-phishing transactions, consistent with phishing "
            "campaigns forming locally dense transaction clusters."
        )
    elif phi_int < non_phi_int - 0.02:
        print(
            "  • Phishing-incident transactions cross community boundaries "
            "more often, which may reflect attacks that target victims across "
            "multiple network neighbourhoods."
        )
    else:
        print(
            "  • Phishing and non-phishing transactions show similar community "
            "containment, suggesting no strong structural separation at the "
            "transaction level."
        )
    if cov_10 >= 0.80:
        print(
            "  • The high concentration of phishing-incident addresses in a "
            "small number of communities suggests that a community-aware "
            "triage strategy—inspecting communities with above-baseline "
            "enrichment—could be a useful starting point for forensic "
            "investigation, though it would not be a complete detection "
            "strategy."
        )
    else:
        print(
            "  • Phishing-incident addresses are spread across many "
            "communities, which limits the practical value of a community-level "
            "triage strategy in isolation."
        )
    print()
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """
    Complete Leiden post-hoc evaluation.

    Labels are used only here, after community detection, to study how
    phishing-labelled activity is distributed over the Leiden communities.
    """
    transactions, node_labels, communities = load_inputs()

    # --- community structure -----------------------------------------------
    community_structure = compute_community_structure(communities)

    # --- transaction internality -------------------------------------------
    tx = attach_communities_to_transactions(transactions, communities)
    transaction_internality = compute_transaction_internality(tx)

    # --- per-community statistics ------------------------------------------
    community_stats, global_phishing_node_rate = compute_community_node_statistics(
        node_labels, communities
    )

    # --- concentration metrics ---------------------------------------------
    top_k_results = compute_top_k_concentration(community_stats)

    # --- top communities table ---------------------------------------------
    top_communities = build_top_communities(
        community_stats,
        limit=TOP_LIMIT,
        min_community_size=MIN_COMMUNITY_SIZE,
    )

    # --- evaluation summary ------------------------------------------------
    evaluation = build_evaluation(
        tx=tx,
        community_stats=community_stats,
        community_structure=community_structure,
        global_phishing_node_rate=global_phishing_node_rate,
        transaction_internality=transaction_internality,
        top_k_results=top_k_results,
    )

    # --- write outputs -----------------------------------------------------
    top_communities.to_csv(top_communities_out, index=False)
    evaluation.to_csv(evaluation_out, index=False)

    # --- console output ----------------------------------------------------
    print("Leiden evaluation complete.")
    print(f"Wrote evaluation summary to:  {evaluation_out}")
    print(f"Wrote top communities to:     {top_communities_out}")

    print()
    print("Key metrics:")
    print(
        f"  Phishing transaction internality:     "
        f"{transaction_internality['phishing_transaction_internality']:.4f}"
    )
    print(
        f"  Non-phishing transaction internality: "
        f"{transaction_internality['non_phishing_transaction_internality']:.4f}"
    )
    print(f"  Global phishing-incident node rate:   {global_phishing_node_rate:.4f}")
    print(f"  Max phishing enrichment:              {community_stats['phishing_enrichment'].max():.4f}")
    print(f"  Number of communities:                {community_structure['num_communities']:,}")
    print(f"  Median community size:                {community_structure['median_community_size']:.1f}")
    print(f"  Largest community:                    {community_structure['largest_community_size']:,}")

    # Full interpretive narrative.
    print_interpretation(
        community_structure=community_structure,
        transaction_internality=transaction_internality,
        top_k_results=top_k_results,
        community_stats=community_stats,
        global_phishing_node_rate=global_phishing_node_rate,
    )


if __name__ == "__main__":
    main()
