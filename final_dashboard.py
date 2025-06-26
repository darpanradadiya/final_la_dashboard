# combined_dashboard.py — Unified Streamlit app
# ------------------------------------------------------------
# This file merges *both* of your original dashboards **line-for-line**.
# Nothing was deleted or functionally altered—only wrapped so you can
# choose each view from the sidebar. Copy it as-is and run:
#
#     streamlit run combined_dashboard.py
#
# ------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import folium
from folium.plugins import HeatMap, MarkerCluster
from pathlib import Path
from itertools import chain
from streamlit_folium import st_folium
import branca
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────────────────────
# 0 ▸ Global page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="📊 Unified LA Dashboards",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# 1 ▸ Shared constants
# ─────────────────────────────────────────────────────────────
LA_BOUNDS = [[33.7, -118.7], [34.4, -117.6]]
CENTER = [(LA_BOUNDS[0][0] + LA_BOUNDS[1][0]) / 2,
          (LA_BOUNDS[0][1] + LA_BOUNDS[1][1]) / 2]

# ─────────────────────────────────────────────────────────────
# 2 ▸ ULTA CO-VISIT HELPERS (from ulta_dashboard.py)
# ─────────────────────────────────────────────────────────────
CSV_PATH = Path(
    "/Users/darpanradadiya/Downloads/ULTA_BEAUTY_DASHBOARD/Data/la_foot_traffic.csv"
)

@st.cache_data(show_spinner=False)
def load_ulta_partners(csv: Path):
    if not csv.exists():
        return None, None, None, None

    df = pd.read_csv(csv)

    # safe parser: dict-string → list
    def parse(raw):
        if not isinstance(raw, str) or raw.strip() in ("", "{}", "NaN"):
            return []
        try:
            d = eval(raw.replace("'", '"').replace('""', '"'))
            return list(d) if isinstance(d, dict) else []
        except Exception:
            return []

    df["brand_list"] = df["related_same_day_brand"].apply(parse)

    # find “Ulta” token
    ulta_tok = next(
        (b for b in set(chain.from_iterable(df["brand_list"])) if "ulta" in b.lower()),
        "Ulta Beauty",
    )

    ulta_mask = df["brand_list"].apply(lambda L: ulta_tok in L)
    ulta_days = int(ulta_mask.sum())

    joint = (
        df[ulta_mask]["brand_list"].explode()
          .loc[lambda s: s != ulta_tok]
          .value_counts()
          .rename("joint_visits")
    )
    overall = df["brand_list"].explode().value_counts().rename("brand_visits")

    stats = pd.concat([joint, overall], axis=1)
    stats["support/Confidence"] = stats["joint_visits"] / ulta_days
    stats["lift"] = stats["support/Confidence"] / (stats["brand_visits"] / len(df))
    stats = stats.reset_index()
    stats.rename(columns={stats.columns[0]: "partner"}, inplace=True)
    return ulta_tok, ulta_days, stats, df

ULTA_TOKEN, ULTA_DAYS, PARTNERS, RAW_DF = load_ulta_partners(CSV_PATH)

# ─────────────────────────────────────────────────────────────
# 3 ▸ OTHER DATA LOADERS (exactly from second dashboard)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_ad_data():
    return pd.read_csv("../Data/final_dashboard_data.csv")

@st.cache_data
def load_geodata():
    # Foot traffic
    df_ft = pd.read_csv("../data/la_foot_traffic.csv")
    df_ft["raw_visit_counts"] = df_ft["raw_visit_counts"].fillna(0)
    upscale = [
        "Restaurants and Other Eating Places",
        "Jewelry, Luggage, and Leather Goods Stores",
        "Fitness and Recreational Sports Centers",
    ]
    df_ft = df_ft[df_ft["top_category"].isin(upscale)]
    df_ft = df_ft[
        df_ft.latitude.between(LA_BOUNDS[0][0], LA_BOUNDS[1][0])
        & df_ft.longitude.between(LA_BOUNDS[0][1], LA_BOUNDS[1][1])
    ]
    gdf_ft = gpd.GeoDataFrame(
        df_ft,
        geometry=gpd.points_from_xy(df_ft.longitude, df_ft.latitude),
        crs="EPSG:4326",
    )

    # Current chargers
    tesla = gpd.read_file("../data/current_tesla_chargers.geojson").to_crs(epsg=4326)

    # Recommendations
    recs = gpd.read_file("../data/final_scored_locations.geojson").to_crs(epsg=4326)

    # Distance to nearest charger (for later filtering)
    t_proj = tesla.to_crs(epsg=32611)
    r_proj = recs.to_crs(epsg=32611)
    recs["dist_to_tesla"] = r_proj.geometry.apply(lambda p: t_proj.distance(p).min())

    # Clip to LA bounding box
    recs = recs[
        recs.geometry.y.between(LA_BOUNDS[0][0], LA_BOUNDS[1][0])
        & recs.geometry.x.between(LA_BOUNDS[0][1], LA_BOUNDS[1][1])
    ]

    return gdf_ft, tesla, recs

@st.cache_data
def load_store_data():
    df = pd.read_csv("../data/imputed_combined_dataset.csv")
    df["raw_visit_counts"] = df["raw_visit_counts"].fillna(0)
    return df

# Pre-load so each view is instant
ad_df = load_ad_data()
gdf_ft, tesla_chargers, recs = load_geodata()
df_store = load_store_data()

# ─────────────────────────────────────────────────────────────
# 4 ▸ SIDEBAR VIEW PICKER
# ─────────────────────────────────────────────────────────────
view = st.sidebar.radio(
    "Select Dashboard",
    [
        "Ulta Co-Visits",
        "Ad Placement",
        "Tesla Charger Siting",
        "LA Foot Traffic",
        "Target Store Forecast",
    ],
)

# ─────────────────────────────────────────────────────────────
# 5 ▸ ULTA CO-VISIT VIEW  (original ulta_dashboard.py)
# ─────────────────────────────────────────────────────────────
if view == "Ulta Co-Visits":
    if PARTNERS is None:
        st.error(
            f"CSV not found at:\n{CSV_PATH}\n→ check the path or edit CSV_PATH."
        )
        st.stop()

    # ── Sidebar filters
    st.sidebar.header("Filters")
    min_joint = st.sidebar.slider(
        "Min joint visits", 1, int(PARTNERS["joint_visits"].max()), 5, 1
    )
    sort_col = st.sidebar.selectbox(
        "Sort table by", ["lift", "joint_visits", "support/Confidence"]
    )
    highlight = st.sidebar.multiselect(
        "Highlight partners", PARTNERS["partner"].head(15).tolist()
    )
    st.sidebar.divider()
    st.sidebar.header("Map")
    map_partner = st.sidebar.selectbox(
        "Partner to map", ["—"] + PARTNERS["partner"].tolist()
    )

    # ── KPI tiles
    c1, c2, c3 = st.columns(3)
    c1.metric("Ulta rows", f"{ULTA_DAYS:,}")
    c2.metric("Co-visit brands", f"{len(PARTNERS):,}")
    c3.metric("Top lift", f"{PARTNERS['lift'].max():.0f}×")

    # ── Filter & scatter
    flt = (
        PARTNERS.query("joint_visits >= @min_joint")
        .sort_values(sort_col, ascending=False)
    )

    st.markdown("### 📊 Lift vs. joint-visit count")
    fig = px.scatter(
        flt,
        x="joint_visits",
        y="lift",
        text="partner",
        labels={
            "joint_visits": "Joint visits with Ulta (week)",
            "lift": "Co-visit lift",
        },
        height=420,
    )
    fig.update_traces(
        textposition="top center",
        marker=dict(
            color=np.where(flt["partner"].isin(highlight), "crimson", "steelblue")
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Table + download
    st.markdown("### 🏆 Partner table")
    st.dataframe(flt.reset_index(drop=True), use_container_width=True, height=360)
    st.download_button(
        "⬇ Download CSV",
        flt.to_csv(index=False).encode(),
        "ulta_covisit_partners.csv",
    )

    # ── Map
    if map_partner != "—":
        st.markdown(f"### 🗺️ Ulta & **{map_partner}** store locations")

        # Ulta POIs
        ulta_geo = RAW_DF[
            RAW_DF["location_name"].str.contains("ulta", case=False, na=False)
        ][["location_name", "latitude", "longitude"]].dropna().drop_duplicates()

        # partner POIs
        part_geo = RAW_DF[
            RAW_DF["location_name"].str.contains(map_partner, case=False, na=False)
        ][["location_name", "latitude", "longitude"]].dropna().drop_duplicates()

        if ulta_geo.empty or part_geo.empty:
            st.info("No coordinates found for one of the brands.")
        else:
            mdf = pd.concat(
                [ulta_geo.assign(category="Ulta"), part_geo.assign(category=map_partner)]
            ).reset_index(drop=True)
            mfig = px.scatter_mapbox(
                mdf,
                lat="latitude",
                lon="longitude",
                color="category",
                hover_name="location_name",
                zoom=8,
                height=500,
                mapbox_style="open-street-map",
            )
            st.plotly_chart(mfig, use_container_width=True)

    # ── Notes
    with st.expander("🔬 Method"):
        st.markdown(
            f"""
* Anchor token detected → **{ULTA_TOKEN}**  
* One-week LA slice of SafeGraph Weekly Patterns.  
* **Lift** = P(partner | Ulta) ÷ P(partner overall).  
* Map uses simple `location_name` substring to find stores.
"""
        )

# ─────────────────────────────────────────────────────────────
# 6 ▸ AD PLACEMENT VIEW  (unchanged)
# ─────────────────────────────────────────────────────────────
elif view == "Ad Placement":
    st.title("📍 Optimal Ad Placement Zones in Los Angeles")

    st.sidebar.header("Filter Demographics")
    quartile_stats = (
        ad_df.groupby("income_quartile")["median_income"].agg(["min", "max"]).astype(int)
    )
    quartile_names = {"Q1": "Low Income", "Q2": "Lower-middle", "Q3": "Upper-middle", "Q4": "High Income"}
    quartile_labels = {
        q: f"{quartile_names[q]} (${row['min']:,}–${row['max']:,})"
        for q, row in quartile_stats.iterrows()
    }

    income_q = st.sidebar.multiselect(
        "Income Quartile", options=list(quartile_labels.keys()), format_func=lambda x: quartile_labels[x]
    )
    age_grp = st.sidebar.multiselect(
        "Dominant Age Group", options=sorted(ad_df["dominant_age_group"].unique())
    )
    race_grp = st.sidebar.multiselect(
        "Dominant Race Group", options=sorted(ad_df["dominant_race_group"].unique())
    )
    top_n = st.sidebar.slider("Top N CBGs to View", 5, 50, 10)
    beta = st.sidebar.slider("Weight for Visit Count (β)", 0.0, 2.0, 1.0, 0.1)
    gamma = st.sidebar.slider("Weight for Dwell Time (γ)", 0.0, 2.0, 1.0, 0.1)

    filtered = ad_df.copy()
    if income_q:
        filtered = filtered[filtered["income_quartile"].isin(income_q)]
    if age_grp:
        filtered = filtered[filtered["dominant_age_group"].isin(age_grp)]
    if race_grp:
        filtered = filtered[filtered["dominant_race_group"].isin(race_grp)]

    filtered = (
        filtered.sort_values("normalized_score", ascending=False)
        .drop_duplicates("GEOID")
        .copy()
    )
    filtered["score"] = (filtered["Visit_Count"] ** beta) * (filtered["median_dwell"] ** gamma)
    mn, mx = filtered["score"].min(), filtered["score"].max()
    filtered["normalized_score"] = (
        (filtered["score"] - mn) / (mx - mn) if mx > mn else 0
    )

    top_df = filtered.head(top_n)

    st.markdown("Use sidebar filters to explore zones based on visits & demographics.")
    fig = px.scatter_mapbox(
        top_df,
        lat="latitude",
        lon="longitude",
        color="normalized_score",
        hover_name="GEOID",
        hover_data=[
            "Visit_Count",
            "median_income",
            "dominant_age_group",
            "dominant_race_group",
        ],
        zoom=9,
        height=500,
        color_continuous_scale="YlOrRd",
    )
    fig.update_layout(mapbox_style="carto-darkmatter")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Summary Table")
    st.dataframe(
        top_df[
            [
                "GEOID",
                "normalized_score",
                "Visit_Count",
                "median_income",
                "dominant_age_group",
                "dominant_race_group",
            ]
        ]
    )

# ─────────────────────────────────────────────────────────────
# 7 ▸ TESLA CHARGER SITING VIEW  (unchanged)
# ─────────────────────────────────────────────────────────────
elif view == "Tesla Charger Siting":
    st.title("🔌 Tesla Charger Siting")

    st.sidebar.markdown("### 🎛️ Weight Adjustments")
    labels = ["Foot Traffic", "EV Density", "Road Access", "Proximity", "Parking"]
    defaults = [0.35, 0.25, 0.15, 0.15, 0.10]
    weights = [
        st.sidebar.slider(l, 0.0, 1.0, d) for l, d in zip(labels, defaults)
    ]
    w = [x / sum(weights) for x in weights]

    recs["adjusted"] = (
        w[0] * recs["suitability_score"]
        + w[1] * recs["ev_score"]
        + w[2] * recs["access_score"]
        + w[3] * recs["proximity_score"]
        + w[4] * recs["parking_score"]
    )

    min_score = st.sidebar.slider("Min Score", 0.0, 1.0, 0.5, 0.01)
    top_n = st.sidebar.number_input("Top N Sites", 5, 50, 10)

    top_sites = recs[recs["adjusted"] >= min_score].nlargest(top_n, "adjusted")

    tabs = st.tabs(["📈 Foot Traffic", "🔌 Current Chargers", "🚗 Recommendations"])

    # --- Heatmap
    with tabs[0]:
        st.subheader("LA Foot Traffic Heatmap")
        m = folium.Map(location=CENTER, zoom_start=10, tiles="cartodbpositron")
        HeatMap(
            [
                [r.latitude, r.longitude, r.raw_visit_counts]
                for _, r in gdf_ft.iterrows()
            ],
            radius=15,
            blur=10,
            min_opacity=0.3,
        ).add_to(m)
        folium.LayerControl().add_to(m)
        m.fit_bounds(LA_BOUNDS)
        st_folium(m, height=900, use_container_width=True, key="tesla_ft")

    # --- Current chargers
    with tabs[1]:
        st.subheader("Existing Tesla Chargers")
        m = folium.Map(location=CENTER, zoom_start=10, tiles="cartodbpositron")
        cluster = MarkerCluster().add_to(m)
        for _, r in tesla_chargers.iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x],
                radius=8,
                color="red",
                fill=True,
            ).add_to(cluster)
        folium.LayerControl().add_to(m)
        m.fit_bounds(LA_BOUNDS)
        st_folium(m, height=900, use_container_width=True, key="tesla_ch")

    # --- Recommendations
    with tabs[2]:
        st.subheader("Top Recommendations")
        m = folium.Map(location=CENTER, zoom_start=12, tiles="cartodbpositron")

        # Current chargers layer
        fg_ch = folium.FeatureGroup("Current Chargers")
        for _, r in tesla_chargers.iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x],
                radius=6,
                color="red",
                fill=True,
                fill_opacity=0.8,
            ).add_to(fg_ch)
        fg_ch.add_to(m)

        # Recommendations layer
        cmap = branca.colormap.linear.YlOrRd_09.scale(0, 1)
        fg_rec = folium.FeatureGroup("Recommendations")
        for _, r in top_sites.iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x],
                radius=10,
                color=cmap(r["adjusted"]),
                fill=True,
                tooltip=r["location_name"],
                fill_opacity=0.9,
            ).add_to(fg_rec)
        fg_rec.add_to(m)

        folium.LayerControl().add_to(m)
        st_folium(m, height=900, use_container_width=True, key="tesla_rec")

# ─────────────────────────────────────────────────────────────
# 8 ▸ LA FOOT TRAFFIC STANDALONE VIEW  (unchanged)
# ─────────────────────────────────────────────────────────────
elif view == "LA Foot Traffic":
    st.title("📈 LA Foot Traffic Heatmap")

    m = folium.Map(location=CENTER, zoom_start=10, tiles="cartodbpositron")
    HeatMap(
        [
            [r.latitude, r.longitude, r.raw_visit_counts]
            for _, r in gdf_ft.iterrows()
        ],
        radius=15,
        blur=10,
        min_opacity=0.3,
    ).add_to(m)
    folium.LayerControl().add_to(m)
    m.fit_bounds(LA_BOUNDS)
    st_folium(m, height=900, use_container_width=True, key="la_ft")

# ─────────────────────────────────────────────────────────────
# 9 ▸ STORE FORECAST VIEW  (unchanged)
# ─────────────────────────────────────────────────────────────
elif view == "Store Forecast":
    st.title("🏬 Target Foot Traffic Forecast Dashboard")
    st.markdown(
        "<style>footer{visibility:hidden;}</style>",
        unsafe_allow_html=True,
    )

    avg_visits = df_store.groupby("placekey")["raw_visit_counts"].mean()
    max_pk = avg_visits.idxmax()
    hottest = df_store[df_store.placekey == max_pk].iloc[0]

    st.sidebar.subheader("🔥 Busiest Store Alert")
    st.sidebar.write(
        f"🏆 Most Visited: **{hottest.brands} — {hottest.street_address}**"
    )
    st.sidebar.metric("Avg Weekly Visits", int(avg_visits.max()))

    df_store["store_display"] = (
        df_store.brands.fillna("Unknown")
        + " — "
        + df_store.street_address.fillna("No Address")
    )
    choice = st.sidebar.selectbox("🔍 Choose a Store", df_store["store_display"].unique())
    pk = dict(zip(df_store["store_display"], df_store.placekey))[choice]
    store_data = (
        df_store[df_store.placekey == pk]
        .sort_values("date_range_start")
        .copy()
    )

    k1, k2, k3 = st.columns(3)
    k1.metric("📌 Selected Store", choice.split("—")[0].strip())
    k2.metric("📅 Data Points", len(store_data))
    k3.metric("📊 Avg Visits", int(store_data.raw_visit_counts.mean()))

    # --- Simple numeric forecasting (as in your original script)
    y = store_data["raw_visit_counts"]
    X = store_data.drop(
        columns=[
            "raw_visit_counts",
            "placekey",
            "date_range_start",
            "date_range_end",
        ],
        errors="ignore",
    )
    X = X.drop(
        columns=[c for c in X if "visit" in c and c != "visit_counts_from_last_week"],
        errors="ignore",
    )
    X = X.select_dtypes(include=["number"])
    X = X + np.random.normal(0, 0.1, X.shape)
    X = X.loc[:, X.notnull().any(axis=0)]

    imp = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(
        imp.fit_transform(X), columns=X.columns, index=X.index
    )

    if len(X_imputed) >= 10:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42
        )
        model = LinearRegression().fit(X_tr, y_tr)
        y_pr = model.predict(X_te)
        forecast = model.predict(X_imputed.iloc[[-1]])[0]
        staff_pct = round(min(100, forecast * 0.6), 1)
        inv_pct = round(min(100, forecast * 0.8), 1)
        rmse = np.sqrt(mean_squared_error(y_te, y_pr))
    else:
        forecast = staff_pct = inv_pct = rmse = None

    c1, c2, c3 = st.columns([1.1, 1.4, 1.5])

    # Map column
    with c1:
        store_data["is_selected"] = store_data.placekey == pk
        dfm = store_data[
            ["latitude", "longitude", "store_display", "is_selected"]
        ].dropna()
        fig_map = px.scatter_mapbox(
            dfm,
            lat="latitude",
            lon="longitude",
            color="is_selected",
            color_discrete_map={True: "red", False: "blue"},
            hover_name="store_display",
            zoom=10,
            height=300,
        )
        fig_map.update_layout(
            mapbox_style="open-street-map", margin=dict(l=0, r=0, t=0, b=0)
        )
        st.plotly_chart(fig_map, use_container_width=True)

    # Trend column
    with c2:
        st.markdown(
            "<div style='padding:0.2rem;border-radius:10px;border:1px solid #ddd;background:#f9f9f9;'><h4>📈 Foot Traffic Trend</h4>",
            unsafe_allow_html=True,
        )
        fl = px.line(store_data, x="date_range_start", y="raw_visit_counts")
        fl.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=260)
        st.plotly_chart(fl, use_container_width=True)
        with st.expander("📄 Raw Visit Data"):
            st.dataframe(
                store_data[["date_range_start", "raw_visit_counts"]], height=150
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # Forecast/op-readiness column
    with c3:
        st.markdown(
            "<div style='padding:0.2rem;border-radius:10px;border:1px solid #ddd;background:#f9f9f9;'><h4>🛋️ Operational Readiness</h4><p>Derived from forecasted visits.</p>",
            unsafe_allow_html=True,
        )
        if forecast is not None:
            st.progress(staff_pct / 100, text="Staffing Readiness")
            st.progress(inv_pct / 100, text="Inventory Readiness")
        else:
            st.info("Forecast not available.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "<div style='margin-top:1rem;padding:0.2rem;border-radius:10px;border:1px solid #ddd;background:#f9f9f9;'><h4>📈 Forecast Overview</h4>",
            unsafe_allow_html=True,
        )
        if forecast is not None:
            f1, f2, f3 = st.columns(3)
            f1.metric("📈 Forecasted Visits", int(forecast))
            f2.metric("👥 Staff Required", f"{staff_pct}%")
            f3.metric("📦 Inventory Need", f"{inv_pct}%")
        else:
            st.warning("Not enough data to display forecast.")
        st.markdown("</div>", unsafe_allow_html=True)
