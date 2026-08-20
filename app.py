from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard import (
    FEATURE_LABELS,
    MODEL_FEATURES,
    SHOP_KEY,
    aggregate_shop_performance,
    build_dashboard_tables,
    cluster_profile,
    cluster_purity,
    compare_shop_to_cluster,
    find_area_type_column,
    find_data_file,
    read_csv,
    typical_outlier_profile,
)


st.set_page_config(
    page_title="Telangana PDS Shop Performance",
    page_icon="📊",
    layout="wide",
)

BASE = Path(__file__).resolve().parent


def fmt(value, decimals=0):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}"


def pct(value, decimals=1):
    if pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}%"


@st.cache_data(show_spinner="Loading dashboard data...")
def load_from_paths(monthly_path: str, cluster_path: str, monthly_mtime: float, cluster_mtime: float):
    # mtimes are intentionally part of the cache key.
    del monthly_mtime, cluster_mtime
    return build_dashboard_tables(read_csv(monthly_path), read_csv(cluster_path))


@st.cache_data(show_spinner="Preparing uploaded dashboard data...")
def load_from_uploads(monthly_bytes: bytes, cluster_bytes: bytes):
    monthly = read_csv(io.BytesIO(monthly_bytes))
    clusters = read_csv(io.BytesIO(cluster_bytes))
    return build_dashboard_tables(monthly, clusters)


st.title("Telangana PDS Analytics")
st.caption("Multi-Dimensional Shop Performance Clustering and Anomaly Profiling")

monthly_path = find_data_file(BASE, "monthly_unified.csv")
cluster_path = find_data_file(BASE, "shop_cluster_results.csv")

if monthly_path and cluster_path:
    try:
        monthly_data, shop_model = load_from_paths(
            str(monthly_path),
            str(cluster_path),
            monthly_path.stat().st_mtime,
            cluster_path.stat().st_mtime,
        )
        st.sidebar.caption(f"Monthly: {monthly_path.name}\n\nClusters: {cluster_path.name}")
    except Exception as error:
        st.error(f"Could not load processed notebook outputs: {error}")
        st.stop()
else:
    st.info(
        "Place `monthly_unified.csv` and `shop_cluster_results.csv` in "
        "`data/processed/` beside this app, or upload both files below."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        monthly_upload = st.file_uploader("monthly_unified.csv", type="csv")
    with col_b:
        cluster_upload = st.file_uploader("shop_cluster_results.csv", type="csv")

    if monthly_upload is None or cluster_upload is None:
        st.stop()

    try:
        monthly_data, shop_model = load_from_uploads(
            monthly_upload.getvalue(), cluster_upload.getvalue()
        )
    except Exception as error:
        st.error(f"Could not prepare uploaded files: {error}")
        st.stop()


# --------------------------- Sidebar filters ---------------------------
st.sidebar.header("Filters")

all_districts = sorted(monthly_data["district_name"].dropna().astype(str).unique())
all_years = sorted(monthly_data["year"].dropna().astype(int).unique())
all_clusters = sorted(shop_model["cluster_label"].dropna().astype(str).unique())

selected_districts = st.sidebar.multiselect(
    "District",
    all_districts,
    default=all_districts,
)
selected_years = st.sidebar.multiselect(
    "Year",
    all_years,
    default=all_years,
)
selected_clusters = st.sidebar.multiselect(
    "Behavioral persona",
    all_clusters,
    default=all_clusters,
)

# Empty multiselect means "none" rather than silently resetting to all.
monthly_filtered = monthly_data.copy()
if selected_districts:
    monthly_filtered = monthly_filtered[
        monthly_filtered["district_name"].astype(str).isin(selected_districts)
    ]
else:
    monthly_filtered = monthly_filtered.iloc[0:0]

if selected_years:
    monthly_filtered = monthly_filtered[
        monthly_filtered["year"].astype("Int64").isin(selected_years)
    ]
else:
    monthly_filtered = monthly_filtered.iloc[0:0]

if selected_clusters:
    monthly_filtered = monthly_filtered[
        monthly_filtered["cluster_label"].isna()
        | monthly_filtered["cluster_label"].astype(str).isin(selected_clusters)
    ]
else:
    monthly_filtered = monthly_filtered.iloc[0:0]

if monthly_filtered.empty:
    st.warning("No monthly records match the current District / Year / Persona filters.")
    st.stop()

period_shop_metrics = aggregate_shop_performance(monthly_filtered)
clustered_period_shops = period_shop_metrics[period_shop_metrics["cluster_id"].notna()].copy()

st.caption(
    "Cluster assignments are the full-period K-Means results produced by the modelling notebook. "
    "The Year filter recalculates observed shop performance for the selected years; it does not retrain K-Means."
)

# --------------------------- Dashboard tabs ---------------------------
overview_tab, profile_tab, pca_tab, map_tab, search_tab, validation_tab = st.tabs(
    [
        "Overview",
        "Cluster Profiles",
        "PCA",
        "Geospatial",
        "Shop Search",
        "Validation",
    ]
)


with overview_tab:
    transaction_rows = monthly_filtered[monthly_filtered["noOfTrans"].notna()].copy()
    reporting_shops = transaction_rows[SHOP_KEY].drop_duplicates().shape[0]
    clustered_shops = clustered_period_shops[SHOP_KEY].drop_duplicates().shape[0]
    outliers = (
        int(clustered_period_shops["is_noise"].fillna(False).sum())
        if "is_noise" in clustered_period_shops.columns
        else 0
    )
    total_transactions = transaction_rows["noOfTrans"].sum()
    avg_utilization = transaction_rows["utilization_ratio"].replace([np.inf, -np.inf], np.nan).mean()
    portability = transaction_rows["otherShopTransCnt"].sum()

    cols = st.columns(6)
    cols[0].metric("Reporting shops", fmt(reporting_shops))
    cols[1].metric("Clustered shops", fmt(clustered_shops))
    cols[2].metric("DBSCAN outliers", fmt(outliers))
    cols[3].metric("Transactions", fmt(total_transactions))
    cols[4].metric("Other-shop transactions", fmt(portability))
    cols[5].metric("Avg utilization", fmt(avg_utilization, 3))

    trend = (
        transaction_rows.groupby("period", as_index=False)
        .agg(
            total_transactions=("noOfTrans", "sum"),
            other_shop_transactions=("otherShopTransCnt", "sum"),
        )
        .sort_values("period")
    )

    left, right = st.columns(2)
    with left:
        fig = px.line(
            trend,
            x="period",
            y="total_transactions",
            markers=True,
            title="Monthly Transaction Trend",
            labels={"period": "Month", "total_transactions": "Transactions"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.line(
            trend,
            x="period",
            y="other_shop_transactions",
            markers=True,
            title="Monthly Portability Trend",
            labels={"period": "Month", "other_shop_transactions": "Other-shop transactions"},
        )
        st.plotly_chart(fig, use_container_width=True)

    if not clustered_period_shops.empty:
        distribution = (
            clustered_period_shops["cluster_label"]
            .value_counts()
            .rename_axis("Cluster")
            .reset_index(name="Shops")
        )
        fig = px.bar(
            distribution,
            x="Cluster",
            y="Shops",
            color="Cluster",
            text_auto=True,
            title="Behavioral Persona Distribution",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)


with profile_tab:
    st.subheader("Cluster Profile Report")
    profile = cluster_profile(clustered_period_shops)
    if profile.empty:
        st.info("No clustered shops are available for the selected filters.")
    else:
        st.dataframe(profile, use_container_width=True)
        st.caption(
            "Feature means are recalculated from the selected years while keeping the notebook's K-Means persona assignment."
        )

        numeric_profile = profile.reset_index()
        chart_feature = st.selectbox(
            "Compare one profile feature",
            [FEATURE_LABELS[f] for f in MODEL_FEATURES if FEATURE_LABELS[f] in numeric_profile.columns],
        )
        fig = px.bar(
            numeric_profile,
            x="cluster_label",
            y=chart_feature,
            color="cluster_label",
            title=f"{chart_feature} by Persona",
            labels={"cluster_label": "Cluster / Persona"},
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Typical vs Outlier Shops")
    anomaly_profile = typical_outlier_profile(clustered_period_shops)
    if anomaly_profile.empty:
        st.info("DBSCAN anomaly information is unavailable.")
    else:
        st.dataframe(anomaly_profile, use_container_width=True)


with pca_tab:
    st.subheader("PCA Visualization of K-Means Shop Clusters")
    st.caption(
        "PCA is reconstructed from the same eight saved shop-level features using median imputation and StandardScaler, matching the modelling notebook's preprocessing."
    )

    pca_view = shop_model.copy()
    if selected_districts:
        pca_view = pca_view[pca_view["district_name"].astype(str).isin(selected_districts)]
    if selected_clusters:
        pca_view = pca_view[pca_view["cluster_label"].astype(str).isin(selected_clusters)]

    # Restrict PCA display to shops that reported in the selected year(s).
    reporting_keys = period_shop_metrics[SHOP_KEY].drop_duplicates()
    pca_view = pca_view.merge(reporting_keys, on=SHOP_KEY, how="inner")

    if {"PCA1", "PCA2"}.issubset(pca_view.columns) and not pca_view.empty:
        dimension = st.radio(
            "PCA view",
            ["2D", "3D"],
            horizontal=True,
            disabled="PCA3" not in pca_view.columns,
        )

        hover = [c for c in ["shopNo", "district_name", "anomaly_status"] if c in pca_view.columns]
        if dimension == "3D" and "PCA3" in pca_view.columns:
            fig = px.scatter_3d(
                pca_view,
                x="PCA1",
                y="PCA2",
                z="PCA3",
                color="cluster_label",
                symbol="anomaly_status" if "anomaly_status" in pca_view.columns else None,
                hover_data=hover,
                opacity=0.55,
                title="3D PCA Cluster View",
            )
        else:
            fig = px.scatter(
                pca_view,
                x="PCA1",
                y="PCA2",
                color="cluster_label",
                symbol="anomaly_status" if "anomaly_status" in pca_view.columns else None,
                hover_data=hover,
                opacity=0.55,
                title="2D PCA Cluster View",
            )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("PCA could not be calculated because too few usable model features were available.")


with map_tab:
    st.subheader("Geospatial Cluster Map")
    map_data = clustered_period_shops.dropna(
        subset=["latitude", "longitude", "cluster_label"]
    ).copy()
    map_data = map_data[
        map_data["latitude"].between(-90, 90)
        & map_data["longitude"].between(-180, 180)
    ]

    if map_data.empty:
        st.info("No valid latitude/longitude rows are available for the selected filters.")
    else:
        hover_cols = [
            c
            for c in ["shopNo", "district_name", "cluster_persona", "anomaly_status", "average_transactions"]
            if c in map_data.columns
        ]
        fig = px.scatter_mapbox(
            map_data,
            lat="latitude",
            lon="longitude",
            color="cluster_label",
            hover_name="shopNo",
            hover_data=hover_cols,
            zoom=6,
            height=600,
            title="Shops Color-Coded by Cluster ID / Persona",
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 45, "l": 0, "b": 0},
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Geospatial Hotspot Map")
        hotspot_cluster = st.selectbox(
            "Hotspot persona",
            sorted(map_data["cluster_label"].dropna().astype(str).unique()),
        )
        hotspot = map_data[map_data["cluster_label"].astype(str).eq(hotspot_cluster)]
        if not hotspot.empty:
            fig = px.density_mapbox(
                hotspot,
                lat="latitude",
                lon="longitude",
                radius=20,
                zoom=6,
                height=600,
                hover_name="shopNo",
                title=f"Hotspot Density — {hotspot_cluster}",
            )
            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r": 0, "t": 45, "l": 0, "b": 0},
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Brighter areas indicate a higher geographic concentration of the selected cluster type.")


with search_tab:
    st.subheader("Shop Search and Cluster Benchmark")
    st.caption(
        "Search is performed across all districts. Performance and peer averages use the Year selection from the sidebar."
    )

    query = st.text_input("Enter shopNo", placeholder="Example: 1901001").strip()
    if query:
        numeric_query = pd.to_numeric(query, errors="coerce")
        if pd.isna(numeric_query):
            st.warning("shopNo must be numeric.")
        else:
            shop_no = int(numeric_query)
            candidates = (
                monthly_data[monthly_data["shopNo"].eq(shop_no)]
                .sort_values("period")
                .drop_duplicates(SHOP_KEY, keep="last")
                .copy()
            )
            if candidates.empty:
                st.warning(f"shopNo {shop_no} was not found in the processed monthly data.")
            else:
                if len(candidates) > 1:
                    candidates["choice"] = candidates.apply(
                        lambda row: f"{int(row['distCode'])} | {row.get('district_name', 'Unknown')} | {int(row['shopNo'])}",
                        axis=1,
                    )
                    selected_choice = st.selectbox("Multiple matches found", candidates["choice"].tolist())
                    candidate = candidates[candidates["choice"].eq(selected_choice)].iloc[0]
                else:
                    candidate = candidates.iloc[0]

                dist_code = int(candidate["distCode"])

                benchmark_monthly = monthly_data.copy()
                if selected_years:
                    benchmark_monthly = benchmark_monthly[
                        benchmark_monthly["year"].astype("Int64").isin(selected_years)
                    ]
                else:
                    benchmark_monthly = benchmark_monthly.iloc[0:0]

                benchmark_metrics = aggregate_shop_performance(benchmark_monthly)
                comparison, shop = compare_shop_to_cluster(
                    benchmark_metrics,
                    dist_code=dist_code,
                    shop_no=shop_no,
                )

                if shop is None:
                    st.warning("The shop has no reporting rows in the selected year(s).")
                else:
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Shop", str(shop_no))
                    c2.metric("District", str(shop.get("district_name", dist_code)))
                    c3.metric("Persona", str(shop.get("cluster_persona", "Not clustered")))
                    c4.metric("DBSCAN status", str(shop.get("anomaly_status", "Unavailable")))

                    if comparison.empty:
                        st.info("This shop is not assigned to a K-Means cluster, so a peer benchmark is unavailable.")
                    else:
                        display_comparison = comparison.copy()
                        for column in ["Shop", "Cluster average", "Difference"]:
                            display_comparison[column] = display_comparison[column].round(3)
                        display_comparison["Difference %"] = display_comparison["Difference %"].round(1)
                        st.dataframe(display_comparison, use_container_width=True, hide_index=True)

                        chart = comparison.melt(
                            id_vars="Metric",
                            value_vars=["Shop", "Cluster average"],
                            var_name="Series",
                            value_name="Value",
                        )
                        fig = px.bar(
                            chart,
                            x="Metric",
                            y="Value",
                            color="Series",
                            barmode="group",
                            title="Selected Shop vs Cluster Average",
                        )
                        st.plotly_chart(fig, use_container_width=True)

                shop_months = monthly_data[
                    monthly_data["distCode"].eq(dist_code)
                    & monthly_data["shopNo"].eq(shop_no)
                    & monthly_data["noOfTrans"].notna()
                ].copy()
                if selected_years:
                    shop_months = shop_months[
                        shop_months["year"].astype("Int64").isin(selected_years)
                    ]
                if not shop_months.empty:
                    fig = px.line(
                        shop_months.sort_values("period"),
                        x="period",
                        y="noOfTrans",
                        markers=True,
                        title="Shop Monthly Transactions",
                        labels={"period": "Month", "noOfTrans": "Transactions"},
                    )
                    st.plotly_chart(fig, use_container_width=True)


with validation_tab:
    st.subheader("Data and Evaluation Checks")
    monthly_duplicates = int(monthly_data.duplicated(["distCode", "shopNo", "year", "month"]).sum())
    cluster_duplicates = int(shop_model.duplicated(SHOP_KEY).sum())
    unclustered_shops = (
        monthly_data[SHOP_KEY].drop_duplicates()
        .merge(shop_model[SHOP_KEY].drop_duplicates(), on=SHOP_KEY, how="left", indicator=True)
        .query("_merge == 'left_only'")
        .shape[0]
    )

    check_rows = pd.DataFrame(
        {
            "Check": [
                "Monthly duplicate shop-month keys",
                "Duplicate clustered shop keys",
                "Monthly shops without K-Means assignment",
                "DBSCAN noise shops",
            ],
            "Value": [
                monthly_duplicates,
                cluster_duplicates,
                unclustered_shops,
                int(shop_model["is_noise"].sum()) if "is_noise" in shop_model.columns else np.nan,
            ],
        }
    )
    st.dataframe(check_rows, hide_index=True, use_container_width=True)

    area_column = find_area_type_column(monthly_data)
    st.subheader("Cluster Purity: Urban vs Rural")
    if area_column is None:
        st.warning(
            "The processed notebook data does not contain an explicit Urban/Rural reference label, "
            "so cluster purity cannot be calculated defensibly. Add a genuine area-type field to the source data before using this metric."
        )
    else:
        purity_source = shop_model.merge(
            monthly_data.sort_values("period").drop_duplicates(SHOP_KEY, keep="last")[[*SHOP_KEY, area_column]],
            on=SHOP_KEY,
            how="left",
        )
        purity_table, overall_purity = cluster_purity(purity_source, area_column)
        st.metric("Overall cluster purity", pct(overall_purity * 100 if pd.notna(overall_purity) else np.nan))
        st.dataframe(purity_table, use_container_width=True, hide_index=True)

    st.caption(
        "Silhouette Score and Elbow Curve remain part of the modelling notebook, where K-Means was trained. "
        "This dashboard focuses on monitoring, profiles, PCA, anomalies, maps, and shop-level peer comparison."
    )
