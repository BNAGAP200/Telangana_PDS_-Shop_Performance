"""Data preparation helpers for the Telangana PDS Streamlit dashboard.

This module intentionally does not retrain K-Means or DBSCAN. The modelling
notebook is the source of cluster assignments. The dashboard consumes:

    data/processed/monthly_unified.csv
    data/processed/shop_cluster_results.csv

PCA coordinates are reconstructed from the saved model features because the
notebook plots PCA but does not save the component coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


SHOP_KEY = ["distCode", "shopNo"]
MONTH_KEY = ["distCode", "shopNo", "year", "month"]

MODEL_FEATURES = [
    "average_transactions",
    "transaction_volatility",
    "average_cards",
    "average_utilization",
    "average_portability_ratio",
    "average_monthly_rice",
    "average_monthly_wheat",
    "average_rice_wheat_ratio",
]

FEATURE_LABELS = {
    "average_transactions": "Avg transactions",
    "transaction_volatility": "Transaction volatility",
    "average_cards": "Avg ration cards",
    "average_utilization": "Avg utilization ratio",
    "average_portability_ratio": "Avg portability ratio",
    "average_monthly_rice": "Avg monthly rice",
    "average_monthly_wheat": "Avg monthly wheat",
    "average_rice_wheat_ratio": "Avg rice/wheat ratio",
}

# These names come from the cluster_names dictionary in the submitted
# feature-engineering notebook. That dictionary was not applied when the CSV
# was saved, so the dashboard restores those intended behavioural labels.
NOTEBOOK_PERSONAS = {
    0: "Stable Regular Shops",
    1: "High-Volume Shops",
    2: "Low-Utilization Shops",
    3: "Portability Hubs",
}

METADATA_COLUMNS = [
    "district_name",
    "officeName",
    "address",
    "longitude",
    "latitude",
    "fpsStatus",
    "fpsType",
]


def read_csv(path_or_buffer) -> pd.DataFrame:
    """Read a dashboard CSV with low-memory inference disabled."""
    return pd.read_csv(path_or_buffer, low_memory=False)


def find_data_file(base: Path, filename: str) -> Path | None:
    """Find a processed file without relying on notebook absolute paths."""
    candidates = [
        base / "data" / "processed" / filename,
        base / "data" / filename,
        base / filename,
        base.parent / "data" / "processed" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _to_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def _normalize_shop_keys(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    missing = [column for column in SHOP_KEY if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing shop key columns: {missing}")

    for column in SHOP_KEY:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame


def prepare_monthly(monthly: pd.DataFrame) -> pd.DataFrame:
    """Validate and standardize the notebook's monthly_unified dataset."""
    monthly = _normalize_shop_keys(monthly)

    required = MONTH_KEY + ["noOfTrans", "otherShopTransCnt", "totalRcs"]
    missing = [column for column in required if column not in monthly.columns]
    if missing:
        raise ValueError(f"monthly_unified.csv is missing required columns: {missing}")

    for column in ["year", "month"]:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce").astype("Int64")

    invalid_keys = monthly[MONTH_KEY].isna().any(axis=1)
    if invalid_keys.any():
        raise ValueError(
            f"monthly_unified.csv contains {int(invalid_keys.sum())} rows with missing keys."
        )

    duplicated = monthly.duplicated(MONTH_KEY)
    if duplicated.any():
        raise ValueError(
            f"monthly_unified.csv contains {int(duplicated.sum())} duplicate shop-month keys."
        )

    numeric_columns = [
        "noOfRcs",
        "noOfTrans",
        "otherShopTransCnt",
        "totalRcs",
        "totalUnits",
        "totalAmount",
        "riceAfsc",
        "riceFsc",
        "riceAap",
        "wheat",
        "sugar",
        "rgdal",
        "kerosene",
        "salt",
        "total_rice",
        "utilization_ratio",
        "portability_ratio",
        "rice_wheat_ratio",
        "longitude",
        "latitude",
    ]
    _to_numeric(monthly, numeric_columns)

    if "period" in monthly.columns:
        monthly["period"] = pd.to_datetime(monthly["period"], errors="coerce")
    else:
        monthly["period"] = pd.to_datetime(
            {
                "year": monthly["year"],
                "month": monthly["month"],
                "day": 1,
            },
            errors="coerce",
        )

    if monthly["period"].isna().any():
        raise ValueError("monthly_unified.csv contains invalid year/month combinations.")

    if "total_rice" not in monthly.columns:
        rice_cols = [c for c in ["riceAfsc", "riceFsc", "riceAap"] if c in monthly]
        if rice_cols:
            monthly["total_rice"] = monthly[rice_cols].fillna(0).sum(axis=1)

    if "utilization_ratio" not in monthly.columns:
        monthly["utilization_ratio"] = (
            monthly["noOfTrans"] / monthly["totalRcs"].replace(0, np.nan)
        )

    if "portability_ratio" not in monthly.columns:
        monthly["portability_ratio"] = (
            monthly["otherShopTransCnt"] / monthly["noOfTrans"].replace(0, np.nan)
        )

    if "rice_wheat_ratio" not in monthly.columns and "total_rice" in monthly.columns:
        monthly["rice_wheat_ratio"] = (
            monthly["total_rice"] / monthly["wheat"].replace(0, np.nan)
        )

    # Prefer the transaction district name, then the location district name.
    district = pd.Series(pd.NA, index=monthly.index, dtype="string")
    for candidate in ["distName", "distName_location", "distName_transaction"]:
        if candidate in monthly.columns:
            district = district.fillna(monthly[candidate].astype("string"))
    monthly["district_name"] = district.fillna(monthly["distCode"].astype("string"))

    return monthly


def _is_generic_cluster_name(value: object, cluster_id: int) -> bool:
    if pd.isna(value):
        return True
    cleaned = str(value).strip().lower().replace("_", " ")
    return cleaned in {
        f"cluster {cluster_id}".lower(),
        str(cluster_id),
    }


def prepare_clusters(clusters: pd.DataFrame) -> pd.DataFrame:
    """Validate notebook cluster output and restore persona/anomaly fields."""
    clusters = _normalize_shop_keys(clusters)

    if "cluster_id" not in clusters.columns:
        raise ValueError("shop_cluster_results.csv is missing cluster_id.")

    clusters["cluster_id"] = pd.to_numeric(
        clusters["cluster_id"], errors="coerce"
    ).astype("Int64")

    invalid = clusters[SHOP_KEY + ["cluster_id"]].isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"shop_cluster_results.csv contains {int(invalid.sum())} rows with invalid keys/cluster IDs."
        )

    duplicated = clusters.duplicated(SHOP_KEY)
    if duplicated.any():
        raise ValueError(
            "shop_cluster_results.csv must contain one row per district/shop; "
            f"found {int(duplicated.sum())} duplicate rows."
        )

    _to_numeric(clusters, MODEL_FEATURES + ["dbscan_label"])

    if "is_noise" in clusters.columns:
        if clusters["is_noise"].dtype != bool:
            clusters["is_noise"] = (
                clusters["is_noise"]
                .astype("string")
                .str.strip()
                .str.lower()
                .isin(["true", "1", "yes"])
            )
    elif "dbscan_label" in clusters.columns:
        clusters["is_noise"] = clusters["dbscan_label"].eq(-1)
    else:
        clusters["is_noise"] = False

    # Preserve source cluster_name, but expose the intended notebook persona name.
    if "cluster_name" not in clusters.columns:
        clusters["cluster_name"] = pd.NA

    def choose_persona(row: pd.Series) -> str:
        cid = int(row["cluster_id"])
        source_name = row.get("cluster_name", pd.NA)
        if not _is_generic_cluster_name(source_name, cid):
            return str(source_name)
        return NOTEBOOK_PERSONAS.get(cid, f"Cluster {cid}")

    clusters["cluster_persona"] = clusters.apply(choose_persona, axis=1)
    clusters["cluster_label"] = clusters.apply(
        lambda row: f"Cluster {int(row['cluster_id'])} — {row['cluster_persona']}", axis=1
    )
    clusters["anomaly_status"] = np.where(
        clusters["is_noise"], "Outlier (DBSCAN noise)", "Typical"
    )

    return add_pca_coordinates(clusters)


def add_pca_coordinates(clusters: pd.DataFrame) -> pd.DataFrame:
    """Recreate PCA coordinates from the same saved model features.

    The notebook used median imputation followed by StandardScaler, then PCA.
    This function follows that same transformation and adds PCA1/PCA2/PCA3.
    """
    clusters = clusters.copy()

    existing = [c for c in ["PCA1", "PCA2", "PCA3"] if c in clusters.columns]
    if len(existing) >= 2:
        return clusters

    available = [feature for feature in MODEL_FEATURES if feature in clusters.columns]
    if len(available) < 2 or len(clusters) < 2:
        return clusters

    X = clusters[available].replace([np.inf, -np.inf], np.nan).copy()
    medians = X.median(numeric_only=True)
    X = X.fillna(medians)

    # If a feature still has no usable values, exclude it from PCA only.
    usable = [column for column in X.columns if X[column].notna().all()]
    if len(usable) < 2:
        return clusters

    X_scaled = StandardScaler().fit_transform(X[usable])
    n_components = min(3, len(usable), len(clusters))
    coords = PCA(n_components=n_components).fit_transform(X_scaled)

    for index in range(n_components):
        clusters[f"PCA{index + 1}"] = coords[:, index]

    return clusters


def build_dashboard_tables(
    monthly: pd.DataFrame, clusters: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create monthly-with-clusters and one-row-per-clustered-shop tables."""
    monthly = prepare_monthly(monthly)
    clusters = prepare_clusters(clusters)

    latest_metadata_cols = [c for c in METADATA_COLUMNS if c in monthly.columns]
    latest_metadata = (
        monthly.sort_values("period")
        .drop_duplicates(SHOP_KEY, keep="last")
        [SHOP_KEY + latest_metadata_cols]
    )

    shops = clusters.merge(
        latest_metadata,
        on=SHOP_KEY,
        how="left",
        validate="one_to_one",
    )

    cluster_fields = SHOP_KEY + [
        "cluster_id",
        "cluster_name",
        "cluster_persona",
        "cluster_label",
        "dbscan_label",
        "is_noise",
        "anomaly_status",
    ]
    cluster_fields = [c for c in cluster_fields if c in clusters.columns]

    monthly_with_clusters = monthly.merge(
        clusters[cluster_fields],
        on=SHOP_KEY,
        how="left",
        validate="many_to_one",
    )

    return monthly_with_clusters, shops


def aggregate_shop_performance(monthly: pd.DataFrame) -> pd.DataFrame:
    """Aggregate any selected period into shop-level performance metrics.

    Cluster assignment remains the full-period notebook assignment, while these
    metrics respond to the dashboard's Year/District filters.
    """
    if monthly.empty:
        return pd.DataFrame()

    data = monthly.copy()
    grouped = data.groupby(SHOP_KEY, as_index=False)

    result = grouped.agg(
        reporting_months=("noOfTrans", "count"),
        total_transactions=("noOfTrans", "sum"),
        average_transactions=("noOfTrans", "mean"),
        transaction_volatility=("noOfTrans", "std"),
        average_cards=("totalRcs", "mean"),
        average_utilization=("utilization_ratio", "mean"),
        average_portability_ratio=("portability_ratio", "mean"),
        total_portability_transactions=("otherShopTransCnt", "sum"),
        average_monthly_rice=("total_rice", "mean") if "total_rice" in data.columns else ("noOfTrans", "size"),
        average_monthly_wheat=("wheat", "mean") if "wheat" in data.columns else ("noOfTrans", "size"),
        average_rice_wheat_ratio=("rice_wheat_ratio", "mean") if "rice_wheat_ratio" in data.columns else ("noOfTrans", "size"),
    )

    # When a commodity column is absent, replace the placeholder count with NaN.
    for source, target in [
        ("total_rice", "average_monthly_rice"),
        ("wheat", "average_monthly_wheat"),
        ("rice_wheat_ratio", "average_rice_wheat_ratio"),
    ]:
        if source not in data.columns:
            result[target] = np.nan

    carry_columns = [
        "cluster_id",
        "cluster_persona",
        "cluster_label",
        "dbscan_label",
        "is_noise",
        "anomaly_status",
        "district_name",
        "officeName",
        "address",
        "longitude",
        "latitude",
        "fpsStatus",
        "fpsType",
    ]
    carry_columns = [column for column in carry_columns if column in data.columns]

    latest = (
        data.sort_values("period")
        .drop_duplicates(SHOP_KEY, keep="last")
        [SHOP_KEY + carry_columns]
    )

    return result.merge(latest, on=SHOP_KEY, how="left", validate="one_to_one")


def cluster_profile(shop_metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean feature profile plus shop/outlier counts for each persona."""
    if shop_metrics.empty or "cluster_label" not in shop_metrics.columns:
        return pd.DataFrame()

    available = [feature for feature in MODEL_FEATURES if feature in shop_metrics.columns]
    valid = shop_metrics[shop_metrics["cluster_label"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()

    means = valid.groupby("cluster_label")[available].mean()
    counts = valid.groupby("cluster_label").size().rename("number_of_shops")
    outliers = (
        valid.groupby("cluster_label")["is_noise"].sum().rename("dbscan_outliers")
        if "is_noise" in valid.columns
        else pd.Series(0, index=means.index, name="dbscan_outliers")
    )

    profile = pd.concat([counts, outliers, means], axis=1)
    profile["outlier_rate_pct"] = np.where(
        profile["number_of_shops"] > 0,
        100 * profile["dbscan_outliers"] / profile["number_of_shops"],
        np.nan,
    )

    ordered = ["number_of_shops", "dbscan_outliers", "outlier_rate_pct"] + available
    return profile[ordered].rename(columns=FEATURE_LABELS).round(3)


def typical_outlier_profile(shop_metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare DBSCAN-typical and DBSCAN-noise shop averages."""
    if shop_metrics.empty or "anomaly_status" not in shop_metrics.columns:
        return pd.DataFrame()

    available = [feature for feature in MODEL_FEATURES if feature in shop_metrics.columns]
    if not available:
        return pd.DataFrame()

    profile = shop_metrics.groupby("anomaly_status")[available].mean().rename(
        columns=FEATURE_LABELS
    )
    counts = shop_metrics.groupby("anomaly_status").size().rename("number_of_shops")
    return pd.concat([counts, profile], axis=1).round(3)


def compare_shop_to_cluster(
    shop_metrics: pd.DataFrame,
    dist_code: int,
    shop_no: int,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return selected shop vs its cluster peers for the current time selection."""
    if shop_metrics.empty:
        return pd.DataFrame(), None

    selected = shop_metrics[
        shop_metrics["distCode"].eq(dist_code) & shop_metrics["shopNo"].eq(shop_no)
    ]
    if selected.empty:
        return pd.DataFrame(), None

    shop = selected.iloc[0]
    if pd.isna(shop.get("cluster_id")):
        return pd.DataFrame(), shop

    peers = shop_metrics[shop_metrics["cluster_id"].eq(shop["cluster_id"])]
    available = [feature for feature in MODEL_FEATURES if feature in shop_metrics.columns]
    cluster_mean = peers[available].mean()

    rows = []
    for feature in available:
        shop_value = shop.get(feature, np.nan)
        mean_value = cluster_mean.get(feature, np.nan)
        difference = shop_value - mean_value if pd.notna(shop_value) and pd.notna(mean_value) else np.nan
        difference_pct = (
            difference / mean_value * 100
            if pd.notna(difference) and pd.notna(mean_value) and mean_value != 0
            else np.nan
        )
        rows.append(
            {
                "Metric": FEATURE_LABELS.get(feature, feature),
                "Shop": shop_value,
                "Cluster average": mean_value,
                "Difference": difference,
                "Difference %": difference_pct,
            }
        )

    return pd.DataFrame(rows), shop


def find_area_type_column(frame: pd.DataFrame) -> str | None:
    """Locate an explicit Urban/Rural reference field if the data contains one."""
    candidates = [
        "urban_rural",
        "urbanRural",
        "area_type",
        "areaType",
        "district_type",
        "districtType",
    ]
    return next((column for column in candidates if column in frame.columns), None)


def cluster_purity(frame: pd.DataFrame, area_column: str) -> tuple[pd.DataFrame, float]:
    """Calculate simple majority-label cluster purity against an explicit area label."""
    data = frame.dropna(subset=["cluster_id", area_column]).copy()
    if data.empty:
        return pd.DataFrame(), np.nan

    counts = (
        data.groupby(["cluster_id", area_column]).size().rename("shops").reset_index()
    )
    totals = counts.groupby("cluster_id")["shops"].sum().rename("cluster_total")
    winners = counts.loc[counts.groupby("cluster_id")["shops"].idxmax()].copy()
    winners = winners.merge(totals, on="cluster_id")
    winners["cluster_purity"] = winners["shops"] / winners["cluster_total"]
    overall = winners["shops"].sum() / len(data)
    return winners.sort_values("cluster_id"), float(overall)
