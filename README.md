# Ethereum Phishing Community Detection

## Setup

I recommend using `uv`, which you can setup with `uv sync`.

Otherwise, every package is saved in a `requirements.txt` file, which you can read with pip.

## Purpose

This folder contains the shared graph setup for the Ethereum phishing community detection project.

The goal is to build standardized graph files from the raw Ethereum transaction dataset so that Louvain, Leiden, and Infomap can be run on comparable inputs.

## Dataset

The raw dataset is stored at:

```text
dataset/archive/1st dataset - imbalanced.csv
```

We use the imbalanced dataset for the main experiments because the project studies the structural distribution of phishing activity in the graph. A balanced dataset would distort the natural phishing rate and therefore affect enrichment, internality, and concentration measurements.

The relevant raw columns are:

```text
TxHash, BlockHeight, TimeStamp, From, To, Value, ContractAddress, Input, Class
```

`Class` is a transaction-level label:

```text
Class = 1: phishing-labelled transaction
Class = 0: non-phishing-labelled transaction
```

The dataset does not directly provide address-level labels. Therefore, address labels are derived only for post-hoc analysis:

```text
is_phishing = 1 if the address participates in at least one Class = 1 transaction
is_phishing = 0 otherwise
```

These derived labels must not be used during community detection.

## Graph construction

Each Ethereum address is represented as a node.

Each transaction is represented as a directed edge from `From` to `To`.

Repeated transactions between the same pair of addresses are aggregated.

The main edge weight is transaction count.

### Directed graph

Output file:

```text
edges_directed_count.csv
```

Columns:

```text
source,target,weight,phishing_tx_count,total_tx_count
```

This graph preserves transaction direction and is used for Infomap.

### Undirected graph

Output file:

```text
edges_undirected_count.csv
```

Columns:

```text
source,target,weight,phishing_tx_count,total_tx_count
```

This graph symmetrizes transactions. For an unordered pair of addresses `{u,v}`, the weight is:

```text
weight(u,v) = transactions from u to v + transactions from v to u
```

This graph is used for Louvain and Leiden.

## Preprocessing

The script `build_graphs.py` performs the following steps:

1. Loads the raw CSV.
2. Strips whitespace from column names.
3. Checks that the required columns exist.
4. Normalizes Ethereum addresses by stripping whitespace and converting them to lowercase.
5. Removes rows with missing `From` or `To` values.
6. Converts `Value` and `Class` to numeric values.
7. Creates cleaned transaction, edge-list, node-label, and statistics files.
8. Creates sample files for quick testing.

Run:

```bash
python3 build_graphs.py
```

## Output files

Main files:

```text
transactions_clean.csv
edges_directed_count.csv
edges_undirected_count.csv
node_labels.csv
dataset_statistics.csv
```

Sample files:

```text
sample_transactions_clean.csv
sample_edges_directed_count.csv
sample_edges_undirected_count.csv
sample_node_labels.csv
```

Visualization files, if generated:

```text
graph_sample_visualization.png
graph_full_visualization.png
```

## Dataset statistics after preprocessing

```text
Raw transactions:                    84,665
Cleaned transactions:                84,557
Dropped missing source/target rows:     108
Unique addresses:                    38,780
Directed aggregated edges:           52,477
Undirected aggregated edges:         50,271
Class 0 transactions:                79,109
Class 1 transactions:                 5,448
Class 1 transaction rate:              6.44%
Phishing-incident addresses:          1,340
Phishing-incident address rate:        3.46%
```

## Algorithm inputs

Use these files:

```text
Louvain: edges_undirected_count.csv
Leiden:  edges_undirected_count.csv
Infomap: edges_directed_count.csv
```

Do not use `Class`, `transactions_clean.csv`, or `node_labels.csv` during community detection.

Labels are only used after community detection to evaluate how phishing-labelled transactions and phishing-incident addresses are distributed over detected communities.