from pathlib import Path

import numpy as np
import pandas as pd




SHOP_KEY = ["distCode", "shopNo"]
MONTHLY_KEY = ["distCode", "shopNo", "year", "month"]


MODEL_FEATURE_COLUMNS = [
    "average_transactions",
    "transaction_volatility",
    "average_cards",
    "average_utilization", 
    "average_portability_ratio",
    "average_monthly_rice",
    "average_monthly_wheat",
    "average_rice_wheat_ratio",
]


def read_csv(path):
    return pd.read_csv(Path(path), low_memory=False)


def _normalize_keys(frame):
    frame = frame.copy()
    for column in SHOP_KEY:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("Int64")
    return frame


def prepare_monthly(monthly):
    monthly = _normalize_keys(monthly)

    required = MONTHLY_KEY + ["noOfTrans", "otherShopTransCnt", "totalRcs"]
    missing = [column for column in required if column not in monthly.columns]              
    if missing:
        raise ValueError(f"Monthly file is missing required columns: {missing}")

    for column in MONTHLY_KEY:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce").astype("Int64")

    if monthly[MONTHLY_KEY].isna().any(axis=1).any():
        raise ValueError("Monthly file contains missing shop, district, year, or month keys.")
    if monthly.duplicated(MONTHLY_KEY).any():
        raise ValueError("Monthly file contains repeated shop-month keys.")

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
        raise ValueError("Monthly file contains invalid periods.")

    numeric_columns = [
        "noOfTrans",
        "otherShopTransCnt",
        "totalRcs",
        "totalAmount",
        "riceAfsc",
        "riceFsc",
        "riceAap",
        "wheat",
        "sugar",
        "kerosene",
        "longitude",
        "latitude",
    ]
    for column in numeric_columns:
        if column in monthly.columns:
            monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

    if "total_rice" not in monthly.columns:
        rice_columns = [
            column for column in ["riceAfsc", "riceFsc", "riceAap"]
            if column in monthly.columns
        ]
        if rice_columns:
            monthly["total_rice"] = monthly[rice_columns].fillna(0).sum(axis=1)

    if "utilization_ratio" not in monthly.columns:
        monthly["utilization_ratio"] = (
            monthly["noOfTrans"] / monthly["totalRcs"].replace(0, np.nan)
        )
    if "portability_ratio" not in monthly.columns:
        monthly["portability_ratio"] = (
            monthly["otherShopTransCnt"]
            / monthly["noOfTrans"].replace(0, np.nan)
        )

    district_candidates = [
        "distName",
        "distName_location",
        "distName_transaction",
    ]
    district_column = next(
        (column for column in district_candidates if column in monthly.columns),
        None,
    )
    if district_column:
        monthly["district_name"] = monthly[district_column].astype("string")
        monthly["district_name"] = monthly["district_name"].fillna(
            monthly["distCode"].astype("string")
        )
    else:
        monthly["district_name"] = monthly["distCode"].astype("string")

    return monthly


def prepare_clusters(clusters):
    clusters = _normalize_keys(clusters)

    required = SHOP_KEY + ["cluster_id"]
    missing = [column for column in required if column not in clusters.columns]
    if missing:
        raise ValueError(f"Cluster file is missing required columns: {missing}")

    if clusters[SHOP_KEY].isna().any(axis=1).any():
        raise ValueError("Cluster file contains missing shop or district keys.")
    if clusters.duplicated(SHOP_KEY).any():
        raise ValueError("Cluster file must contain one row per district and shop.")

    clusters["cluster_id"] = pd.to_numeric(
        clusters["cluster_id"], errors="coerce"
    ).astype("Int64")
    if clusters["cluster_id"].isna().any():
        raise ValueError("Cluster file contains invalid cluster IDs.")

    if "cluster_name" not in clusters.columns:
        clusters["cluster_name"] = clusters["cluster_id"].map(
            lambda value: f"Cluster {int(value)}"
        )
    else:
        clusters["cluster_name"] = clusters["cluster_name"].fillna(
            clusters["cluster_id"].map(lambda value: f"Cluster {int(value)}")
        )

    if "is_noise" in clusters.columns:
        clusters["is_noise"] = (
            clusters["is_noise"]
            .astype("string")
            .str.lower()
            .isin(["true", "1", "yes"])
        )
    elif "dbscan_label" in clusters.columns:
        clusters["dbscan_label"] = pd.to_numeric(
            clusters["dbscan_label"], errors="coerce"
        )
        clusters["is_noise"] = clusters["dbscan_label"].eq(-1)
    else:
        clusters["is_noise"] = False

    for column in MODEL_FEATURE_COLUMNS:
        if column in clusters.columns:
            clusters[column] = pd.to_numeric(clusters[column], errors="coerce")

    return clusters


def build_dashboard_tables(monthly, clusters):
    monthly = prepare_monthly(monthly)
    clusters = prepare_clusters(clusters)

    metadata_candidates = [
        "district_name",
        "officeName",
        "address",
        "longitude",
        "latitude",
        "fpsStatus",
        "fpsType",
    ]
    metadata_columns = [
        column for column in metadata_candidates if column in monthly.columns
    ]

    shop_metadata = (
        monthly
        .sort_values("period")
        .drop_duplicates(SHOP_KEY, keep="last")
        [SHOP_KEY + metadata_columns]
    )

    shops = clusters.merge(
        shop_metadata,
        on=SHOP_KEY,
        how="left",
        validate="one_to_one",
    )

    cluster_fields = SHOP_KEY + [
        "cluster_id",
        "cluster_name",
        "is_noise",
    ]
    monthly_with_clusters = monthly.merge(
        clusters[cluster_fields],
        on=SHOP_KEY,
        how="left",
        validate="many_to_one",
    )

    return monthly_with_clusters, shops

