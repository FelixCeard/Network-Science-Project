from __future__ import annotations

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

project_dir = Path(__file__).resolve().parent

leiden_dir = project_dir / "output" / "algorithms" / "leiden"
evaluation_file = leiden_dir / "leiden_evaluation.csv"
top_communities_file = leiden_dir / "leiden_top_communities.csv"
communities_file = leiden_dir / "leiden_communities.csv"
misc_dir = project_dir / "output" / "misc"
node_labels_file = misc_dir / "node_labels.csv"

for path in [evaluation_file, top_communities_file, communities_file, node_labels_file]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

evaluation = pd.read_csv(evaluation_file).iloc[0]
top = pd.read_csv(top_communities_file)
communities = pd.read_csv(communities_file)
node_labels = pd.read_csv(node_labels_file)

# Rebuild full community stats (needed for size-distribution plots and
# percentile tables that are not in the top-20 file).
full_stats = (
    node_labels.merge(communities[["node", "community"]], on="node", how="left")
    .groupby("community", as_index=False)
    .agg(
        community_size=("node", "count"),
        phishing_incident_nodes=("is_phishing", "sum"),
    )
)
full_stats["phishing_node_rate"] = (
    full_stats["phishing_incident_nodes"] / full_stats["community_size"]
)
global_rate = float(evaluation["global_phishing_node_rate"])
full_stats["phishing_enrichment"] = full_stats["phishing_node_rate"] / global_rate

sep = "-" * 72

# ---------------------------------------------------------------------------
# 1. High-level evaluation metrics
# ---------------------------------------------------------------------------

print("=" * 72)
print("LEIDEN RESULTS — EXTENDED ANALYSIS")
print("=" * 72)

print()
print(sep)
print("A. HIGH-LEVEL EVALUATION METRICS")
print(sep)
print(f"  Algorithm                          : leiden")
print(f"  Graph variant                      : {evaluation['graph_variant']}")
print(f"  Total transactions                 : {int(evaluation['num_transactions']):,}")
print(f"  Phishing transactions (class=1)    : {int(evaluation['num_phishing_transactions']):,}")
print(f"  Non-phishing transactions (class=0): {int(evaluation['num_non_phishing_transactions']):,}")
print(f"  Number of communities              : {int(evaluation['num_communities']):,}")
print(f"  Global phishing-incident node rate : {float(evaluation['global_phishing_node_rate']):.4f}")
print(f"  Phishing tx internality            : {float(evaluation['phishing_transaction_internality']):.4f}")
print(f"  Non-phishing tx internality        : {float(evaluation['non_phishing_transaction_internality']):.4f}")
print(f"  All-tx internality                 : {float(evaluation['all_transaction_internality']):.4f}")
print(f"  Max phishing enrichment            : {float(evaluation['max_phishing_enrichment']):.4f}")
print(f"  Median phishing enrichment         : {float(evaluation['median_phishing_enrichment']):.4f}")
print(f"  Communities with phishing nodes    : {int(evaluation['communities_with_phishing_nodes']):,}")
print(f"  Top-1%  phishing node coverage     : {float(evaluation['top_1pct_phishing_node_coverage']):.4f}")
print(f"  Top-5%  phishing node coverage     : {float(evaluation['top_5pct_phishing_node_coverage']):.4f}")
print(f"  Top-10% phishing node coverage     : {float(evaluation['top_10pct_phishing_node_coverage']):.4f}")

# ---------------------------------------------------------------------------
# 2. Community size distribution
# ---------------------------------------------------------------------------

sizes = full_stats["community_size"]

print()
print(sep)
print("B. COMMUNITY SIZE DISTRIBUTION")
print(sep)
print(f"  Total communities      : {len(sizes):,}")
print(f"  Min size               : {int(sizes.min()):,}")
print(f"  25th percentile        : {int(sizes.quantile(0.25)):,}")
print(f"  Median (50th pct)      : {float(sizes.median()):.1f}")
print(f"  75th percentile        : {int(sizes.quantile(0.75)):,}")
print(f"  90th percentile        : {int(sizes.quantile(0.90)):,}")
print(f"  99th percentile        : {int(sizes.quantile(0.99)):,}")
print(f"  Max size               : {int(sizes.max()):,}")
print(f"  Mean size              : {float(sizes.mean()):.1f}")
print(f"  Std dev                : {float(sizes.std()):.1f}")
print(f"  Singleton communities  : {int((sizes == 1).sum()):,}")
print(f"  Communities < 10 nodes : {int((sizes < 10).sum()):,}")
print(f"  Communities 10-49      : {int(((sizes >= 10) & (sizes < 50)).sum()):,}")
print(f"  Communities 50-499     : {int(((sizes >= 50) & (sizes < 500)).sum()):,}")
print(f"  Communities >= 500     : {int((sizes >= 500).sum()):,}")

# ---------------------------------------------------------------------------
# 3. Enrichment distribution across size bands
# ---------------------------------------------------------------------------

print()
print(sep)
print("C. ENRICHMENT BY COMMUNITY SIZE BAND")
print(sep)

bands = [
    ("Small  (< 10)",       full_stats["community_size"] < 10),
    ("Medium (10–49)",      (full_stats["community_size"] >= 10) & (full_stats["community_size"] < 50)),
    ("Large  (50–499)",     (full_stats["community_size"] >= 50) & (full_stats["community_size"] < 500)),
    ("X-Large (>= 500)",    full_stats["community_size"] >= 500),
]

print(f"  {'Band':<22} {'Count':>6}  {'Median enrich':>14}  {'Max enrich':>10}  {'Pct with phishing':>18}")
print(f"  {'-'*22} {'-'*6}  {'-'*14}  {'-'*10}  {'-'*18}")
for label, mask in bands:
    band = full_stats[mask]
    if len(band) == 0:
        continue
    median_enrich = band["phishing_enrichment"].median()
    max_enrich = band["phishing_enrichment"].max()
    pct_with_phi = (band["phishing_incident_nodes"] > 0).mean() * 100
    print(
        f"  {label:<22} {len(band):>6}  {median_enrich:>14.2f}  {max_enrich:>10.2f}  {pct_with_phi:>17.1f}%"
    )

# ---------------------------------------------------------------------------
# 4. Top communities table (full, not just top 5)
# ---------------------------------------------------------------------------

print()
print(sep)
print("D. TOP PHISHING-ENRICHED COMMUNITIES (size >= 50, sorted by enrichment)")
print(sep)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)
pd.set_option("display.float_format", "{:.4f}".format)

display_cols = [
    "community",
    "community_size",
    "phishing_incident_nodes",
    "phishing_node_rate",
    "phishing_enrichment",
    "total_phishing_incident_transactions",
    "total_incident_transactions",
]
# Only show columns that exist in the file
display_cols = [c for c in display_cols if c in top.columns]
print(top[display_cols].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Coverage curve (how many communities needed to cover X% of phishing nodes)
# ---------------------------------------------------------------------------

print()
print(sep)
print("E. PHISHING-NODE COVERAGE CURVE")
print(sep)

sorted_full = full_stats.sort_values("phishing_incident_nodes", ascending=False).reset_index(drop=True)
total_phi = sorted_full["phishing_incident_nodes"].sum()
cumulative = sorted_full["phishing_incident_nodes"].cumsum()

print(f"  {'Communities (N)':>16}  {'% of total communities':>23}  {'% phishing nodes covered':>25}")
print(f"  {'-'*16}  {'-'*23}  {'-'*25}")
n_communities = len(sorted_full)
for target_pct in [0.50, 0.75, 0.90, 0.95, 1.00]:
    reached = (cumulative >= total_phi * target_pct).idxmax() + 1
    pct_comms = reached / n_communities * 100
    print(
        f"  {reached:>16,}  {pct_comms:>22.1f}%  {target_pct * 100:>24.0f}%"
    )

# ---------------------------------------------------------------------------
# 6. Internality gap interpretation
# ---------------------------------------------------------------------------

print()
print(sep)
print("F. INTERNALITY GAP")
print(sep)
phi_int = float(evaluation["phishing_transaction_internality"])
non_phi_int = float(evaluation["non_phishing_transaction_internality"])
delta = phi_int - non_phi_int
relative = delta / non_phi_int * 100
print(f"  Phishing internality    : {phi_int:.4f}")
print(f"  Non-phishing internality: {non_phi_int:.4f}")
print(f"  Absolute gap (Δ)        : {delta:+.4f}")
print(f"  Relative gap            : {relative:+.2f}% above non-phishing baseline")

print()
print("=" * 72)
print("Analysis complete.")
print("=" * 72)
