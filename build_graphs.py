from pathlib import Path
import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

project_dir = Path(__file__).resolve().parent

samples_dir = project_dir / 'output' / "samples"
samples_dir.mkdir(parents=True, exist_ok=True)
visualizations_dir = project_dir / 'output' / "visualizations"
visualizations_dir.mkdir(parents=True, exist_ok=True)
misc_dir = project_dir / 'output' / "misc"
misc_dir.mkdir(parents=True, exist_ok=True)

raw_file = project_dir / "dataset" / "archive" / "1st dataset - imbalanced.csv"



# main output files
transactions_out = misc_dir / "transactions_clean.csv"
edges_directed_out = misc_dir / "edges_directed_count.csv"
edges_undirected_out = misc_dir / "edges_undirected_count.csv"
node_labels_out = misc_dir / "node_labels.csv"
dataset_stats_out = misc_dir / "dataset_statistics.csv"

# sample output files
sample_transactions_out = samples_dir / "sample_transactions_clean.csv"
sample_edges_directed_out = samples_dir / "sample_edges_directed_count.csv"
sample_edges_undirected_out = samples_dir / "sample_edges_undirected_count.csv"
sample_node_labels_out = samples_dir / "sample_node_labels.csv"

# visualizations
sample_graph_visualization_out = visualizations_dir / "graph_sample_visualization.png"
full_graph_visualization_out = visualizations_dir / "graph_full_visualization.png"



# columns for the raw CSV
required_columns = [
    "TxHash",
    "BlockHeight",
    "TimeStamp",
    "From",
    "To",
    "Value",
    "Class",
]

def validate_input_file(path: Path) -> None:
    """check that the raw dataset file exists"""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

def validate_columns(df: pd.DataFrame) -> None:
    """check that the raw dataset contains the columns needed for graph construction"""
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
            + f"\nAvailable columns: {list(df.columns)}"
        )
        


def normalize_address(series: pd.Series) -> pd.Series:
    """convert addresses to lowercase strings and remove surrounding whitespace"""
    return series.astype("string").str.strip().str.lower()

def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    clean the raw transaction table;
    creates standardized source/target columns, removes rows without valid endpoints, converts value and class to numeric types, and returns transaction table;
    """
    raw_rows = len(df)
    df = df.copy()
    df["source"] = normalize_address(df["From"])
    df["target"] = normalize_address(df["To"])

    # remove rows where source or target is missing or empty
    missing_like = {"", "nan", "none", "null", "<na>"}
    valid_source = ~df["source"].isin(missing_like) & df["source"].notna()
    valid_target = ~df["target"].isin(missing_like) & df["target"].notna()
    df = df[valid_source & valid_target].copy()
    dropped_rows = raw_rows - len(df)
    
    # convert numeric fields, value is kept, but count-weighted graphs are primary
    df["value"] = pd.to_numeric(df["Value"], errors="coerce").fillna(0)
    df["class"] = pd.to_numeric(df["Class"], errors="coerce").fillna(0).astype(int)
    clean = pd.DataFrame(
        {
            "tx_hash": df["TxHash"].astype("string"),
            "block_height": df["BlockHeight"],
            "timestamp": df["TimeStamp"],
            "source": df["source"],
            "target": df["target"],
            "value": df["value"],
            "class": df["class"],
        }
    )
    return clean, dropped_rows

def build_directed_edges(transactions: pd.DataFrame) -> pd.DataFrame:

    """
    builds the directed aggregated graph;
    each row is one directed edge source;
    the weight is the number of transactions from source to target.
    """

    directed = (
        transactions.groupby(["source", "target"], as_index=False)
        .agg(
            weight=("tx_hash", "count"),
            phishing_tx_count=("class", "sum"),
        )
    )
    directed["total_tx_count"] = directed["weight"]
    
    return directed[
        ["source", "target", "weight", "phishing_tx_count", "total_tx_count"]
    ]
    
def build_undirected_edges(transactions: pd.DataFrame) -> pd.DataFrame:

    """
    build the undirected aggregated graph;
    direction is ignored by sorting the two endpoints of each transaction;
    for louvain and leiden algos.
    """
    pairs = transactions[["source", "target", "class", "tx_hash"]].copy()

    # Sort endpoints so u-v and v-u become the same undirected pair.
    pairs["u"] = pairs[["source", "target"]].min(axis=1)
    pairs["v"] = pairs[["source", "target"]].max(axis=1)
    undirected = (
        pairs.groupby(["u", "v"], as_index=False)
        .agg(
            weight=("tx_hash", "count"),
            phishing_tx_count=("class", "sum"),
        )
        .rename(columns={"u": "source", "v": "target"})
    )
    undirected["total_tx_count"] = undirected["weight"]
    
    return undirected[
        ["source", "target", "weight", "phishing_tx_count", "total_tx_count"]
    ]

def build_node_labels(transactions: pd.DataFrame) -> pd.DataFrame:

    """
    make address-level labels from transaction-level labels;
    node is marked as phishing-incident if it participates in at least one transaction with class = 1.
    """
    outgoing = transactions[["source", "class"]].rename(columns={"source": "node"})
    incoming = transactions[["target", "class"]].rename(columns={"target": "node"})
    incidents = pd.concat([outgoing, incoming], ignore_index=True)

    node_labels = (
        incidents.groupby("node", as_index=False)
        .agg(
            phishing_tx_incident_count=("class", "sum"),
            total_tx_incident_count=("class", "count"),
        )
    )
    node_labels["is_phishing"] = (
        node_labels["phishing_tx_incident_count"] > 0
    ).astype(int)
    return node_labels[

        [
            "node",
            "is_phishing",
            "phishing_tx_incident_count",
            "total_tx_incident_count",
        ]
    ]

def build_dataset_statistics(
    raw_rows: int,
    cleaned_rows: int,
    dropped_rows: int,
    transactions: pd.DataFrame,
    directed_edges: pd.DataFrame,
    undirected_edges: pd.DataFrame,
    node_labels: pd.DataFrame,
) -> pd.DataFrame:

    """calculate one-row statistics for the dataset and constructed graphs"""

    class_counts = transactions["class"].value_counts().to_dict()
    class_0 = int(class_counts.get(0, 0))
    class_1 = int(class_counts.get(1, 0))
    unique_nodes = int(node_labels["node"].nunique())
    phishing_nodes = int(node_labels["is_phishing"].sum())

    stats = {
        "raw_rows": raw_rows,
        "cleaned_rows": cleaned_rows,
        "dropped_missing_source_or_target": dropped_rows,
        "unique_nodes": unique_nodes,
        "directed_edges": len(directed_edges),
        "undirected_edges": len(undirected_edges),
        "class_0_transactions": class_0,
        "class_1_transactions": class_1,
        "class_1_rate": class_1 / cleaned_rows if cleaned_rows else 0,
        "phishing_incident_nodes": phishing_nodes,
        "phishing_incident_node_rate": phishing_nodes / unique_nodes
        
        if unique_nodes
        
        else 0,
    }
    
    return pd.DataFrame([stats])

def write_outputs(
    transactions: pd.DataFrame,
    directed_edges: pd.DataFrame,
    undirected_edges: pd.DataFrame,
    node_labels: pd.DataFrame,
    stats: pd.DataFrame,
) -> None:

    """make main CSV files"""
    transactions.to_csv(transactions_out, index=False)
    directed_edges.to_csv(edges_directed_out, index=False)
    undirected_edges.to_csv(edges_undirected_out, index=False)
    node_labels.to_csv(node_labels_out, index=False)
    stats.to_csv(dataset_stats_out, index=False)
    
def write_sample_outputs(transactions: pd.DataFrame, sample_size: int = 10_000) -> None:

    """
    smaller sample files for testing;
    useful before running Louvain, Leiden, or Infomap on the full graph so it takes less time
    """
    sample = transactions.head(sample_size).copy()
    sample_directed = build_directed_edges(sample)
    sample_undirected = build_undirected_edges(sample)
    sample_node_labels = build_node_labels(sample)
    sample.to_csv(sample_transactions_out, index=False)
    sample_directed.to_csv(sample_edges_directed_out, index=False)
    sample_undirected.to_csv(sample_edges_undirected_out, index=False)
    sample_node_labels.to_csv(sample_node_labels_out, index=False)
    
    
def write_sample_graph_visualization(
    undirected_edges: pd.DataFrame,
    node_labels: pd.DataFrame,
    max_edges: int = 300,
) -> None:

    """
    Create a small visualization of a sampled undirected graph
    This is meant as a sanity-check picture. It uses only the first max_edges and edges, so it is not a statistical sample of the whole graph
    """
    sample_edges = undirected_edges.head(max_edges).copy()
    graph = nx.Graph()
    for _, row in sample_edges.iterrows():
        graph.add_edge(row["source"], row["target"], weight=row["weight"])
    phishing_nodes = set(
        node_labels.loc[node_labels["is_phishing"] == 1, "node"]
    )
    node_colors = [
        "red" if node in phishing_nodes else "lightgray"
        for node in graph.nodes()
    ]
    node_sizes = [
        50 if node in phishing_nodes else 15
        for node in graph.nodes()
    ]
    plt.figure(figsize=(12, 10))
    layout = layout = nx.random_layout(graph, seed=42)
    nx.draw_networkx_edges(
        graph,
        layout,
        alpha=0.25,
        width=0.5,
    )
    nx.draw_networkx_nodes(
        graph,
        layout,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0,
    )
    plt.title(
        f"Sample Ethereum Transaction Graph "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(sample_graph_visualization_out, dpi=200)
    plt.close()


def write_full_graph_visualization(
    undirected_edges: pd.DataFrame,
    node_labels: pd.DataFrame,
) -> None:
    """
    Create a visualization of the full undirected graph;
    This will be visually dense and not useful, but it confirms that the complete graph can be loaded and drawn
    """
    graph = nx.Graph()
    for _, row in undirected_edges.iterrows():
        graph.add_edge(row["source"], row["target"], weight=row["weight"])
    phishing_nodes = set(
        node_labels.loc[node_labels["is_phishing"] == 1, "node"]
    )
    node_colors = [
        "red" if node in phishing_nodes else "lightgray"
        for node in graph.nodes()
    ]
    node_sizes = [
        8 if node in phishing_nodes else 1
        for node in graph.nodes()
    ]
    plt.figure(figsize=(18, 18))

    # Spring layout on the full graph can be slow. Fewer iterations keeps it usable.
    layout = nx.spring_layout(graph, seed=42, iterations=20, k=0.05)

    nx.draw_networkx_edges(
        graph,
        layout,
        alpha=0.03,
        width=0.1,
    )

    nx.draw_networkx_nodes(
        graph,
        layout,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0,
    )
    plt.title(
        f"Full Ethereum Transaction Graph "
        f"({graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges)"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(full_graph_visualization_out, dpi=300)
    plt.close()

def main() -> None:
    """Run the complete graph setup"""
    validate_input_file(raw_file)
    print("Reading raw file...")
    raw = pd.read_csv(raw_file)
    print("Raw file read successfully")
    # Remove accidental spaces around column names
    raw.columns = raw.columns.str.strip()
    print("Columns stripped successfully")
    validate_columns(raw)
    print("Columns validated successfully")
    raw_rows = len(raw)

    # Build all processed tables.
    print("Cleaning transactions...")
    transactions, dropped_rows = clean_transactions(raw)
    print("Transactions cleaned successfully")
    print("Building directed edges...")
    directed_edges = build_directed_edges(transactions)
    print("Directed edges built successfully")
    print("Building undirected edges...")
    undirected_edges = build_undirected_edges(transactions)
    print("Undirected edges built successfully")
    print("Building node labels...")
    node_labels = build_node_labels(transactions)
    print("Node labels built successfully")

    # Summarize the resulting graph and labels
    print("Building dataset statistics...")
    stats = build_dataset_statistics(
        raw_rows=raw_rows,
        cleaned_rows=len(transactions),
        dropped_rows=dropped_rows,
        transactions=transactions,
        directed_edges=directed_edges,
        undirected_edges=undirected_edges,
        node_labels=node_labels,
    )
    print("Dataset statistics built successfully")
    
    
    # Save main and sample outputs
    print("Writing outputs...")
    write_outputs(
        transactions=transactions,
        directed_edges=directed_edges,
        undirected_edges=undirected_edges,
        node_labels=node_labels,
        stats=stats,
    )
    print("Outputs written successfully")
    
    write_sample_outputs(transactions)
    print("Sample outputs written successfully")
    write_sample_graph_visualization(undirected_edges, node_labels)
    print("Sample graph visualization written successfully")
    # write_full_graph_visualization(undirected_edges, node_labels)
    # print("Full graph visualization written successfully")
    
    # Print running complete verification
    print("Graph setup complete.")
    print()
    print(stats.T.to_string(header=False))
    print()
    print(f"Wrote cleaned transactions to: {transactions_out:.0f}")
    print(f"Wrote directed edge list to: {edges_directed_out}")
    print(f"Wrote undirected edge list to: {edges_undirected_out}")
    print(f"Wrote node labels to: {node_labels_out}")
    print(f"Wrote dataset statistics to: {dataset_stats_out}")
    print(f"Wrote sample graph visualization to: {sample_graph_visualization_out}")
    print(f"Wrote full graph visualization to: {full_graph_visualization_out}")

if __name__ == "__main__":
    main()