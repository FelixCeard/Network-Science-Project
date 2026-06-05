import argparse
import json
import sys
from pathlib import Path

import flowtorch.bijectors as bij
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


project_dir = Path(__file__).resolve().parent.parent
subgroup_dir = Path(__file__).resolve().parent
syflow_dir = subgroup_dir / "syflow"

misc_dir = project_dir / "output" / "misc"
algorithms_dir = project_dir / "output" / "algorithms"

sys.path.insert(0, str(syflow_dir))

from src.syflow import And_Finder, syflow  # noqa: E402


TARGET_CHOICES = (
    "phishing_enrichment",
    "phishing_node_rate",
    "phishing_incident_nodes",
)

LABEL_FEATURE_COLUMNS = {
    "total_phishing_edge_weight",
    "phishing_edge_weight_fraction",
    "internal_phishing_edge_weight",
    "boundary_phishing_edge_weight",
}

METADATA_COLUMNS = {
    "community",
    "algorithm",
    "graph_variant",
}


class SyflowConfig:
    def __init__(self, alpha=0.3, lamb=2):
        self.lr_flow = 5e-2
        self.lr_classifier = 2e-2
        self.alpha = alpha
        self.lambd = lamb
        self.pop_train_epochs = 1000
        self.subgroup_train_epochs = 1000
        self.final_fit_epochs = 0
        self.temperature = 0.2
        self.bin_deviation = 0.2
        self.use_weights = True
        self.seed = 10

        def flow_gen():
            return bij.Spline(count_bins=12)

        self.flow_gen = flow_gen


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SyFlow on community-level structural features to discover "
            "interpretable descriptions of exceptional subgroups."
        )
    )
    parser.add_argument(
        "--algorithm",
        default="infomap",
        help="Algorithm whose subgroup features and communities to use.",
    )
    parser.add_argument(
        "--features-file",
        default=None,
        help="Optional path to subgroup features CSV.",
    )
    parser.add_argument(
        "--target",
        choices=TARGET_CHOICES,
        default="phishing_enrichment",
        help="Community-level target whose distribution SyFlow should explain.",
    )
    parser.add_argument(
        "--n-subgroups",
        type=int,
        default=3,
        help="Number of exceptional subgroups to discover.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="SyFlow alpha hyperparameter.",
    )
    parser.add_argument(
        "--lamb",
        type=float,
        default=2.0,
        help="SyFlow lambda hyperparameter.",
    )
    return parser.parse_args()


def run_syflow_discovery(
    X,
    Y,
    config: SyflowConfig,
    n_subgroups: int,
    feature_names: list[str],
):
    device = torch.device("cpu")
    cut_points = torch.zeros((X.shape[1], 2))
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)
    scaler_y = StandardScaler()
    Y_scaled = scaler_y.fit_transform(Y)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float64)
    Y_tensor = torch.tensor(Y_scaled, dtype=torch.float64)

    subgroups = []
    priors = []
    rules = []
    pop_flow = None

    print("Logging Hyperparameters")
    print("Alpha:", config.alpha)
    print("Lambda:", config.lambd)
    print("Temperature:", config.temperature, "\n")

    for subgroup_index in range(n_subgroups):
        print(f"Discovering Subgroup #{subgroup_index + 1}")
        for feature_index in range(X.shape[1]):
            cut_points[feature_index, 0] = torch.quantile(
                X_tensor[:, feature_index], 0
            )
            cut_points[feature_index, 1] = torch.quantile(
                X_tensor[:, feature_index], 1
            )
        cut_points = torch.sort(cut_points, dim=1)[0]
        classifier = And_Finder(
            cut_points,
            temperature=config.temperature,
            use_weights=config.use_weights,
            bin_deviation=config.bin_deviation,
        )
        flows, classifier = syflow(
            X_tensor,
            Y_tensor,
            classifier,
            flow_population=pop_flow,
            subgroup_priors=priors,
            pop_train_epochs=config.pop_train_epochs,
            subgroup_train_epochs=config.subgroup_train_epochs,
            final_fit_epochs=config.final_fit_epochs,
            device=device,
            verbose=False,
            lr_flow=config.lr_flow,
            alpha=config.alpha,
            lr_classifier=config.lr_classifier,
            lambd=config.lambd,
            config=config,
        )
        pop_flow = flows[0]
        priors.append(flows[1])
        classifier = classifier.to(torch.device("cpu"))
        subgroup = (
            torch.argmax(classifier(X_tensor), dim=1).detach().numpy() == 1
        )
        subgroups.append(subgroup)
        rules.append(
            classifier.get_rules(
                cut_points,
                scaler=scaler_x,
                feature_names=feature_names,
                X=X_scaled,
            )
        )

    return subgroups, rules


def load_community_targets(algorithm: str) -> pd.DataFrame:
    communities_file = (
        algorithms_dir / algorithm / f"{algorithm}_communities.csv"
    )
    node_labels_file = misc_dir / "node_labels.csv"

    for path in [communities_file, node_labels_file]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")

    communities = pd.read_csv(communities_file)
    node_labels = pd.read_csv(node_labels_file)

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
        )
    )
    global_phishing_node_rate = node_data["is_phishing"].sum() / len(node_data)
    community_stats["phishing_node_rate"] = (
        community_stats["phishing_incident_nodes"]
        / community_stats["community_size"]
    )
    community_stats["phishing_enrichment"] = (
        community_stats["phishing_node_rate"] / global_phishing_node_rate
    )
    return community_stats


def prepare_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = [
        column
        for column in features.columns
        if column not in METADATA_COLUMNS
        and column not in LABEL_FEATURE_COLUMNS
    ]

    X = features[feature_columns].copy()

    for column in X.columns:
        if X[column].dtype == bool:
            X[column] = X[column].astype(float)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    constant_columns = [
        column
        for column in X.columns
        if X[column].nunique(dropna=True) <= 1
    ]
    X = X.drop(columns=constant_columns)

    median_values = X.median(numeric_only=True)
    X = X.fillna(median_values)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0.0)

    remaining_constant_columns = [
        column for column in X.columns if X[column].nunique(dropna=True) <= 1
    ]
    X = X.drop(columns=remaining_constant_columns)

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        bad_columns = [
            column
            for column in X.columns
            if not np.isfinite(X[column].to_numpy(dtype=float)).all()
        ]
        X = X.drop(columns=bad_columns)

    if X.empty:
        raise ValueError("No usable feature columns remain after preprocessing.")

    return X, list(X.columns)


def build_dataset(
    algorithm: str,
    features_file: Path | None,
    target_column: str,
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    if features_file is None:
        features_file = subgroup_dir / f"{algorithm}_subgroup_features.csv"
    if not features_file.exists():
        raise FileNotFoundError(f"Missing features file: {features_file}")

    features = pd.read_csv(features_file)
    targets = load_community_targets(algorithm)

    dataset = features.merge(
        targets[["community", *TARGET_CHOICES]],
        on="community",
        how="left",
    )
    if dataset[target_column].isna().any():
        raise ValueError(f"Missing target values for column: {target_column}")

    X, feature_names = prepare_features(dataset)
    y = dataset[target_column].to_numpy(dtype=float)
    community_ids = dataset["community"].astype(int).tolist()
    return X.to_numpy(dtype=float), y, feature_names, community_ids


def save_results(
    algorithm: str,
    target_column: str,
    community_ids: list[int],
    subgroups: list[np.ndarray],
    rules: list[str],
    feature_names: list[str],
    config: SyflowConfig,
) -> None:
    output_base = subgroup_dir / f"{algorithm}_syflow"
    output_base.mkdir(parents=True, exist_ok=True)

    membership_rows = []
    for subgroup_id, mask in enumerate(subgroups, start=1):
        for community_id, in_subgroup in zip(community_ids, mask):
            membership_rows.append(
                {
                    "community": community_id,
                    "subgroup_id": subgroup_id,
                    "in_subgroup": bool(in_subgroup),
                }
            )

    membership_out = output_base / f"{algorithm}_syflow_membership.csv"
    pd.DataFrame(membership_rows).to_csv(membership_out, index=False)

    rules_out = output_base / f"{algorithm}_syflow_rules.csv"
    pd.DataFrame(
        [
            {
                "subgroup_id": subgroup_id,
                "rule": rule,
                "subgroup_size": int(mask.sum()),
            }
            for subgroup_id, (mask, rule) in enumerate(
                zip(subgroups, rules),
                start=1,
            )
        ]
    ).to_csv(rules_out, index=False)

    summary = {
        "algorithm": algorithm,
        "target": target_column,
        "n_communities": len(community_ids),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "n_subgroups": len(subgroups),
        "alpha": config.alpha,
        "lamb": config.lambd,
        "temperature": config.temperature,
        "rules": rules,
        "subgroup_sizes": [int(mask.sum()) for mask in subgroups],
    }
    summary_out = output_base / f"{algorithm}_syflow_summary.json"
    summary_out.write_text(json.dumps(summary, indent=2))

    print(f"SyFlow complete for {algorithm}.")
    print(f"Communities: {len(community_ids)}")
    print(f"Features: {len(feature_names)}")
    print(f"Target: {target_column}")
    print(f"Wrote membership to: {membership_out}")
    print(f"Wrote rules to: {rules_out}")
    print(f"Wrote summary to: {summary_out}")
    for subgroup_id, rule in enumerate(rules, start=1):
        print(f"Subgroup {subgroup_id}: {rule}")


def main():
    args = parse_args()
    features_file = Path(args.features_file) if args.features_file else None

    X, y, feature_names, community_ids = build_dataset(
        algorithm=args.algorithm,
        features_file=features_file,
        target_column=args.target,
    )

    config = SyflowConfig(alpha=args.alpha, lamb=args.lamb)
    subgroups, rules = run_syflow_discovery(
        X,
        y.reshape(-1, 1),
        config,
        args.n_subgroups,
        feature_names,
    )

    save_results(
        algorithm=args.algorithm,
        target_column=args.target,
        community_ids=community_ids,
        subgroups=subgroups,
        rules=rules,
        feature_names=feature_names,
        config=config,
    )


if __name__ == "__main__":
    main()
