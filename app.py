import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_data import (
    SHOP_KEY,
    build_dashboard_tables,
    read_csv,
)

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


st.set_page_config(
    page_title="Telangana PDS Shop Performance",
    page_icon="📊",
    layout="wide",
)


APP_FOLDER = Path(__file__).resolve().parent
PROJECT_FOLDER = APP_FOLDER

default_monthly_path = (
    PROJECT_FOLDER / "data" / "processed" / "monthly_unified.csv"
)
default_cluster_path = (
    PROJECT_FOLDER / "data" / "processed" / "shop_cluster_results.csv"
)

monthly_path = Path(
    os.getenv("PDS_MONTHLY_FILE", str(default_monthly_path))
)
cluster_path = Path(
    os.getenv("PDS_CLUSTER_FILE", str(default_cluster_path))
)


@st.cache_data(show_spinner="Loading dashboard data...")
def load_dashboard_data(monthly_file, cluster_file, monthly_modified, cluster_modified):
    del monthly_modified, cluster_modified
    monthly = read_csv(monthly_file)
    clusters = read_csv(cluster_file)
    return build_dashboard_tables(monthly, clusters)


def format_number(value, decimals=0):
    if pd.isna(value):
        return "Not available"
    return f"{value:,.{decimals}f}"


st.title("Telangana PDS Shop Performance Dashboard")
st.caption(
    "Monthly transaction monitoring, K-Means shop personas, and DBSCAN anomaly flags. "
    "Anomaly flags indicate unusual behaviour, not confirmed fraud."
)

missing_files = [
    str(path) for path in [monthly_path, cluster_path] if not path.exists()
]
if missing_files:
    st.error("Required dashboard data files were not found.")
    st.code(
        "Expected files:\n"
        f"{monthly_path}\n"
        f"{cluster_path}\n\n"
        "Save model_data as shop_cluster_results.csv after adding cluster_id, "
        "cluster_name, dbscan_label, and is_noise."
    )
    st.stop()

try:
    monthly_data, shop_data = load_dashboard_data(
        str(monthly_path),
        str(cluster_path),
        monthly_path.stat().st_mtime,
        cluster_path.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Dashboard data failed validation: {error}")
    st.stop()


st.sidebar.header("Filters")

district_options = sorted(
    shop_data["district_name"].dropna().astype(str).unique().tolist()
) if "district_name" in shop_data.columns else []
selected_districts = st.sidebar.multiselect(
    "District",
    options=district_options,
    default=district_options,
)

year_options = sorted(
    monthly_data["year"].dropna().astype(int).unique().tolist()
)
selected_years = st.sidebar.multiselect(
    "Year",
    options=year_options,
    default=year_options,
)

cluster_options = sorted(
    shop_data["cluster_name"].dropna().astype(str).unique().tolist()
)
selected_clusters = st.sidebar.multiselect(
    "Cluster persona",
    options=cluster_options,
    default=cluster_options,
)

noise_only = st.sidebar.checkbox("DBSCAN anomalies only", value=False)


shop_filtered = shop_data.copy()
monthly_filtered = monthly_data.copy()

if selected_districts and "district_name" in shop_filtered.columns:
    shop_filtered = shop_filtered[
        shop_filtered["district_name"].astype(str).isin(selected_districts)
    ]
    monthly_filtered = monthly_filtered[
        monthly_filtered["district_name"].astype(str).isin(selected_districts)
    ]
if selected_years:
    monthly_filtered = monthly_filtered[
        monthly_filtered["year"].astype("Int64").isin(selected_years)
    ]
if selected_clusters:
    shop_filtered = shop_filtered[
        shop_filtered["cluster_name"].astype(str).isin(selected_clusters)
    ]
    monthly_filtered = monthly_filtered[
        monthly_filtered["cluster_name"].astype(str).isin(selected_clusters)
    ]
if noise_only:
    shop_filtered = shop_filtered[shop_filtered["is_noise"]]
    monthly_filtered = monthly_filtered[monthly_filtered["is_noise"].fillna(False)]


if monthly_filtered.empty or shop_filtered.empty:
    st.warning("No records match the selected filters.")
    st.stop()


transaction_rows = monthly_filtered[monthly_filtered["noOfTrans"].notna()].copy()

active_shop_count = transaction_rows[SHOP_KEY].drop_duplicates().shape[0]
total_transactions = transaction_rows["noOfTrans"].sum(min_count=1)
portability_transactions = transaction_rows["otherShopTransCnt"].sum(min_count=1)
average_utilization = transaction_rows["utilization_ratio"].replace(
    [np.inf, -np.inf], np.nan
).mean()
noise_shop_count = int(shop_filtered["is_noise"].sum())

kpi_columns = st.columns(5)
kpi_columns[0].metric("Reporting shops", format_number(active_shop_count))
kpi_columns[1].metric("Transactions", format_number(total_transactions))
kpi_columns[2].metric(
    "Other-shop transactions", format_number(portability_transactions)
)
kpi_columns[3].metric("Average utilization", format_number(average_utilization, 2))
kpi_columns[4].metric("DBSCAN anomalies", format_number(noise_shop_count))

st.caption(
    f"Source coverage: {monthly_filtered['period'].min():%b %Y} to "
    f"{monthly_filtered['period'].max():%b %Y}. "
    "Transactions are summed from noOfTrans; utilization is the mean of valid "
    "shop-month noOfTrans / totalRcs ratios."
)


st.subheader("Monthly movement")
monthly_trend = (
    transaction_rows
    .groupby("period", as_index=False)
    .agg(
        Transactions=("noOfTrans", "sum"),
        Other_shop_transactions=("otherShopTransCnt", "sum"),
    )
    .sort_values("period")
)
trend_long = monthly_trend.melt(
    id_vars="period",
    var_name="Metric",
    value_name="Value",
)
trend_chart = px.line(
    trend_long,
    x="period",
    y="Value",
    color="Metric",
    markers=True,
    labels={"period": "Month", "Value": "Transactions"},
)
trend_chart.update_layout(legend_title_text="", hovermode="x unified")
st.plotly_chart(trend_chart, use_container_width=True)


left_column, right_column = st.columns(2)

with left_column:
    st.subheader("Shop personas")
    cluster_counts = (
        shop_filtered["cluster_name"]
        .value_counts()
        .rename_axis("Cluster persona")
        .reset_index(name="Shops")
    )
    cluster_chart = px.bar(
        cluster_counts,
        x="Cluster persona",
        y="Shops",
        color="Cluster persona",
        text_auto=True,
    )
    cluster_chart.update_layout(showlegend=False)
    st.plotly_chart(cluster_chart, use_container_width=True)

with right_column:
    st.subheader("Commodity distribution by district")
    commodity_columns = [
        column for column in ["total_rice", "wheat"]
        if column in transaction_rows.columns
    ]
    if commodity_columns:
        district_commodities = (
            transaction_rows
            .groupby("district_name", as_index=False)[commodity_columns]
            .mean()
            .melt(
                id_vars="district_name",
                var_name="Commodity",
                value_name="Average monthly quantity",
            )
        )
        commodity_chart = px.bar(
            district_commodities,
            x="district_name",
            y="Average monthly quantity",
            color="Commodity",
            barmode="group",
            labels={"district_name": "District"},
        )
        st.plotly_chart(commodity_chart, use_container_width=True)
    else:
        st.info("Rice and wheat columns are unavailable.")


st.subheader("Cluster profile")
available_features = [
    column for column in MODEL_FEATURE_COLUMNS if column in shop_filtered.columns
]
if available_features:
    cluster_profile = (
        shop_filtered
        .groupby("cluster_name")[available_features]
        .mean()
        .round(2)
    )
    cluster_profile.insert(
        0,
        "number_of_shops",
        shop_filtered.groupby("cluster_name").size(),
    )
    st.dataframe(cluster_profile, use_container_width=True)
else:
    st.info("Model feature columns are not available in the cluster-results file.")


st.subheader("Geospatial shop map")
if {"latitude", "longitude"}.issubset(shop_filtered.columns):
    map_data = shop_filtered.dropna(subset=["latitude", "longitude"]).copy()
    map_data = map_data[
        map_data["latitude"].between(-90, 90)
        & map_data["longitude"].between(-180, 180)
    ]
    if not map_data.empty:
        map_chart = px.scatter_mapbox(
            map_data,
            lat="latitude",
            lon="longitude",
            color="cluster_name",
            hover_name="shopNo",
            hover_data={
                "distCode": True,
                "district_name": True,
                "is_noise": True,
                "latitude": False,
                "longitude": False,
            },
            zoom=6,
            height=600,
        )
        map_chart.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend_title_text="Cluster persona",
        )
        st.plotly_chart(map_chart, use_container_width=True)
    else:
        st.info("No valid coordinates are available for the current filters.")
else:
    st.info("Latitude and longitude columns are unavailable.")


st.subheader("Shop search")
shop_search = st.text_input("Enter a shop number")
if shop_search.strip():
    matching_shops = shop_filtered[
        shop_filtered["shopNo"].astype("string").eq(shop_search.strip())
    ]
    if matching_shops.empty:
        st.warning("No matching shop was found within the selected filters.")
    else:
        st.dataframe(matching_shops, use_container_width=True)
        matching_keys = matching_shops[SHOP_KEY].drop_duplicates()
        shop_history = monthly_filtered.merge(
            matching_keys,
            on=SHOP_KEY,
            how="inner",
            validate="many_to_many",
        )
        shop_history = shop_history[shop_history["noOfTrans"].notna()]
        if not shop_history.empty:
            shop_chart = px.line(
                shop_history.sort_values("period"),
                x="period",
                y="noOfTrans",
                color="district_name",
                markers=True,
                labels={"period": "Month", "noOfTrans": "Transactions"},
                title="Shop monthly transaction history",
            )
            st.plotly_chart(shop_chart, use_container_width=True)


st.subheader("DBSCAN anomaly review")
noise_table = shop_filtered[shop_filtered["is_noise"]].copy()
if noise_table.empty:
    st.info("No DBSCAN anomalies match the selected filters.")
else:
    sort_column = next(
        (
            column for column in [
                "average_utilization",
                "transaction_volatility",
                "average_transactions",
            ]
            if column in noise_table.columns
        ),
        None,
    )
    if sort_column:
        noise_table = noise_table.sort_values(sort_column, ascending=False)
    review_columns = [
        column for column in (
            SHOP_KEY
            + ["district_name", "cluster_name", "dbscan_label", "is_noise"]
            + available_features
        )
        if column in noise_table.columns
    ]
    st.dataframe(noise_table[review_columns].head(100), use_container_width=True)


st.download_button(
    "Download filtered shop results",
    data=shop_filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_shop_cluster_results.csv",
    mime="text/csv",
)


with st.expander("Metric definitions and limitations"):
    st.markdown(
        """
        - **Transactions:** sum of `noOfTrans` for reporting shop-month rows.
        - **Other-shop transactions:** sum of `otherShopTransCnt`.
        - **Average utilization:** mean of valid monthly `noOfTrans / totalRcs` ratios.
        - **Shop personas:** K-Means labels produced from the engineered shop-level features.
        - **DBSCAN anomaly:** a shop assigned label `-1`; this is an investigation signal, not proof of fraud.
        - Card-only months are not treated as zero-transaction months.
        - Shops without valid coordinates are excluded from the map but remain in other views.
        """
    )

