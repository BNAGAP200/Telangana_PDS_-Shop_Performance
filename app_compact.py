import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard_data_fixed import SHOP_KEY, build_dashboard_tables, read_csv


FEATURES = [
    "average_transactions",
    "transaction_volatility",
    "average_cards",
    "average_utilization",
    "average_portability_ratio",
    "average_monthly_rice",
    "average_monthly_wheat",
    "average_rice_wheat_ratio",
]

LABELS = {
    "average_transactions": "Avg transactions",
    "transaction_volatility": "Transaction volatility",
    "average_cards": "Avg cards",
    "average_utilization": "Avg utilization",
    "average_portability_ratio": "Portability ratio",
    "average_monthly_rice": "Avg monthly rice",
    "average_monthly_wheat": "Avg monthly wheat",
    "average_rice_wheat_ratio": "Rice/wheat ratio",
}

st.set_page_config(
    page_title="Telangana PDS Shop Performance",
    page_icon="📊",
    layout="wide",
)

BASE = Path(__file__).resolve().parent
MONTHLY_FILE = Path(
    os.getenv(
        "PDS_MONTHLY_FILE",
        BASE / "data" / "processed" / "monthly_unified.csv",
    )
)
CLUSTER_FILE = Path(
    os.getenv(
        "PDS_CLUSTER_FILE",
        BASE / "data" / "processed" / "shop_cluster_results.csv",
    )
)


@st.cache_data
def load_data(monthly_file, cluster_file, monthly_mtime, cluster_mtime):
    del monthly_mtime, cluster_mtime
    return build_dashboard_tables(
        read_csv(monthly_file),
        read_csv(cluster_file),
    )


def fmt(value, decimals=0):
    return "N/A" if pd.isna(value) else f"{value:,.{decimals}f}"


for path in [MONTHLY_FILE, CLUSTER_FILE]:
    if not path.exists():
        st.error(f"Missing required file: {path}")
        st.stop()

try:
    monthly_data, shop_data = load_data(
        str(MONTHLY_FILE),
        str(CLUSTER_FILE),
        MONTHLY_FILE.stat().st_mtime,
        CLUSTER_FILE.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Could not load dashboard data: {error}")
    st.stop()


st.title("Telangana PDS Shop Performance Dashboard")
st.caption(
    "Shop performance monitoring, cluster profiling, and geospatial hotspot analysis."
)

st.sidebar.header("Filters")

districts = sorted(
    shop_data["district_name"].dropna().astype(str).unique()
)
years = sorted(monthly_data["year"].dropna().astype(int).unique())
clusters = sorted(
    shop_data["cluster_name"].dropna().astype(str).unique()
)

selected_districts = st.sidebar.multiselect(
    "District", districts, default=districts
)
selected_years = st.sidebar.multiselect(
    "Year", years, default=years
)
selected_clusters = st.sidebar.multiselect(
    "Cluster persona", clusters, default=clusters
)

shops = shop_data.copy()
monthly = monthly_data.copy()

if selected_districts:
    shops = shops[
        shops["district_name"].astype(str).isin(selected_districts)
    ]
    monthly = monthly[
        monthly["district_name"].astype(str).isin(selected_districts)
    ]

if selected_years:
    monthly = monthly[
        monthly["year"].astype("Int64").isin(selected_years)
    ]

if selected_clusters:
    shops = shops[
        shops["cluster_name"].astype(str).isin(selected_clusters)
    ]
    monthly = monthly[
        monthly["cluster_name"].astype(str).isin(selected_clusters)
    ]

if shops.empty or monthly.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

transactions = monthly[monthly["noOfTrans"].notna()].copy()

performance_tab, profile_tab, map_tab = st.tabs(
    ["Shop Performance", "Cluster Profile", "Geospatial Hotspots"]
)


with performance_tab:
    total_shops = transactions[SHOP_KEY].drop_duplicates().shape[0]
    total_transactions = transactions["noOfTrans"].sum()
    avg_utilization = (
        transactions["utilization_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .mean()
    )

    cluster_counts = shops["cluster_name"].value_counts()
    typical_count = int(
        cluster_counts[
            cluster_counts.index.str.contains("typical", case=False, regex=False)
        ].sum()
    )
    outlier_count = int(
        cluster_counts[
            cluster_counts.index.str.contains("outlier", case=False, regex=False)
        ].sum()
    )

    cols = st.columns(5)
    cols[0].metric("Reporting shops", fmt(total_shops))
    cols[1].metric("Typical shops", fmt(typical_count))
    cols[2].metric("Outlier shops", fmt(outlier_count))
    cols[3].metric("Transactions", fmt(total_transactions))
    cols[4].metric("Avg utilization", fmt(avg_utilization, 2))

    trend = (
        transactions.groupby("period", as_index=False)["noOfTrans"]
        .sum()
        .sort_values("period")
    )

    fig = px.line(
        trend,
        x="period",
        y="noOfTrans",
        markers=True,
        labels={"period": "Month", "noOfTrans": "Transactions"},
        title="Monthly Transaction Movement",
    )
    st.plotly_chart(fig, use_container_width=True)


with profile_tab:
    counts = (
        shops["cluster_name"]
        .value_counts()
        .rename_axis("Cluster")
        .reset_index(name="Shops")
    )

    fig = px.bar(
        counts,
        x="Cluster",
        y="Shops",
        color="Cluster",
        text_auto=True,
        title="Shop Cluster Distribution",
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    available = [c for c in FEATURES if c in shops.columns]

    if available:
        profile = (
            shops.groupby("cluster_name")[available]
            .mean()
            .round(2)
            .rename(columns=LABELS)
        )
        profile.insert(
            0,
            "Number of shops",
            shops.groupby("cluster_name").size(),
        )
        profile.index.name = "Cluster"
        st.subheader("Cluster Profile Report")
        st.dataframe(profile, use_container_width=True)

        st.caption(
            "Compare the cluster averages to identify what distinguishes "
            "Typical and Outlier shop groups."
        )
    else:
        st.info("Cluster profile features are unavailable.")


with map_tab:
    required = {"latitude", "longitude", "cluster_name"}

    if required.issubset(shops.columns):
        map_data = shops.dropna(
            subset=["latitude", "longitude", "cluster_name"]
        ).copy()

        map_data = map_data[
            map_data["latitude"].between(-90, 90)
            & map_data["longitude"].between(-180, 180)
        ]

        map_clusters = sorted(
            map_data["cluster_name"].astype(str).unique()
        )

        if map_clusters:
            selected_map_cluster = st.selectbox(
                "Cluster type",
                map_clusters,
            )
            hotspot = map_data[
                map_data["cluster_name"].astype(str).eq(selected_map_cluster)
            ]

            fig = px.density_mapbox(
                hotspot,
                lat="latitude",
                lon="longitude",
                radius=20,
                zoom=6,
                height=600,
                hover_name="shopNo" if "shopNo" in hotspot.columns else None,
            )
            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"Brighter areas show higher concentrations of "
                f"{selected_map_cluster} shops."
            )
        else:
            st.info("No valid mapped shops are available.")
    else:
        st.info("Latitude and longitude are unavailable.")
