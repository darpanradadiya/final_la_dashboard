import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import MarkerCluster, HeatMap
import branca

# --- Page config ---
st.set_page_config(
    page_title="🔌 Tesla Charger Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load data ---
@st.cache_data
def load_data():
    recs = gpd.read_file("../data/final_scored_locations.geojson").to_crs(epsg=4326)
    tesla = gpd.read_file("../data/current_tesla_chargers.geojson").to_crs(epsg=4326)
    df_ft = pd.read_csv("../data/la_foot_traffic.csv")
    # filter upscale foot traffic
    upscale = ["Restaurants and Other Eating Places",
               "Jewelry, Luggage, and Leather Goods Stores",
               "Fitness and Recreational Sports Centers"]
    df_ft = df_ft[df_ft['top_category'].isin(upscale)].dropna(subset=['raw_visit_counts'])
    gdf_ft = gpd.GeoDataFrame(df_ft,
                              geometry=gpd.points_from_xy(df_ft.longitude, df_ft.latitude),
                              crs="EPSG:4326")
    # compute dist_to_tesla
    tesla_proj = tesla.to_crs(epsg=32611)
    recs_proj = recs.to_crs(epsg=32611)
    recs['dist_to_tesla'] = recs_proj.geometry.apply(lambda pt: tesla_proj.distance(pt).min())
    return recs, tesla, gdf_ft

recs, tesla_chargers, gdf_ft = load_data()

# --- Sidebar settings ---
st.sidebar.header("⚙️ Settings")
min_score = st.sidebar.slider("Min Final Score", 0.0, 1.0, 0.5, 0.01)
top_n     = st.sidebar.number_input("Show Top N", 5, 50, 10)

# --- Weight adjuster ---
st.sidebar.markdown("### 🎛️ Weight Adjustments")
w1 = st.sidebar.slider("Foot Traffic", 0.0, 1.0, 0.35)
w2 = st.sidebar.slider("EV Density",    0.0, 1.0, 0.25)
w3 = st.sidebar.slider("Road Access",   0.0, 1.0, 0.15)
w4 = st.sidebar.slider("Proximity",      0.0, 1.0, 0.15)
w5 = st.sidebar.slider("Parking",        0.0, 1.0, 0.10)
# Normalize
total = w1 + w2 + w3 + w4 + w5
w1, w2, w3, w4, w5 = [w/total for w in (w1, w2, w3, w4, w5)]

# --- Compute adjusted score ---
recs['adjusted'] = (
    w1*recs['suitability_score'] +
    w2*recs['ev_score'] +
    w3*recs['access_score'] +
    w4*recs['proximity_score'] +
    w5*recs['parking_score']
)
filtered = recs[recs['adjusted'] >= min_score].sort_values('adjusted', ascending=False)
top = filtered.head(top_n)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Foot Traffic", "🔌 Current Chargers", "🚗 Recommendations"])

# --- Tab 1: Foot Traffic ---
with tab1:
    st.subheader("Foot Traffic Heatmap & Sample Data")
    col1, col2 = st.columns([3, 2])
    with col1:
        m_ft = folium.Map(tiles='cartodbpositron')
        if not gdf_ft.empty:
            bounds = [[gdf_ft.geometry.y.min(), gdf_ft.geometry.x.min()],
                      [gdf_ft.geometry.y.max(), gdf_ft.geometry.x.max()]]
            m_ft.fit_bounds(bounds)
        heat_data = [[pt.y, pt.x, cnt] for pt, cnt in zip(gdf_ft.geometry, gdf_ft.raw_visit_counts)]
        HeatMap(heat_data, radius=20, blur=15, max_zoom=12).add_to(m_ft)
        st_folium(m_ft, height=900, width=1300)
    with col2:
        st.markdown("#### Sample Foot Traffic Data")
        st.dataframe(gdf_ft[['location_name','top_category','raw_visit_counts']].head(10), use_container_width=True)

# --- Tab 2: Current Tesla Chargers ---
with tab2:
    st.subheader("Existing Tesla Destination Chargers in LA")
    col1, col2 = st.columns([3, 2])
    with col1:
        m_t = folium.Map(tiles='cartodbpositron')
        if not tesla_chargers.empty:
            b = [[tesla_chargers.geometry.y.min(), tesla_chargers.geometry.x.min()],
                 [tesla_chargers.geometry.y.max(), tesla_chargers.geometry.x.max()]]
            m_t.fit_bounds(b)
        cluster = MarkerCluster().add_to(m_t)
        for _, r in tesla_chargers.iterrows():
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x], radius=8,
                color='red', fill=True, fill_opacity=0.9,
                popup=r.get('location_name','Tesla Charger')
            ).add_to(cluster)
        st_folium(m_t, height=900, width=1300)
    with col2:
        st.markdown("#### Charger List")
        st.dataframe(tesla_chargers[['location_name']].drop_duplicates(), use_container_width=True)

# --- Tab 3: Recommendations ---
with tab3:
    st.subheader(f"Recommended Sites (Score ≥ {min_score:.2f})")
    col1, col2 = st.columns([3, 2])
    with col1:
        m_r = folium.Map(tiles='cartodbpositron')
        cmap = branca.colormap.linear.YlOrRd_09.scale(0,1)
        cmap.caption = 'Adjusted Score'
        cmap.add_to(m_r)
        pts = pd.concat([pd.DataFrame({'lat': tesla_chargers.geometry.y, 'lon': tesla_chargers.geometry.x}),
                         pd.DataFrame({'lat': top.geometry.y,   'lon': top.geometry.x})])
        m_r.fit_bounds([[pts.lat.min(), pts.lon.min()], [pts.lat.max(), pts.lon.max()]])
        cl = MarkerCluster().add_to(m_r)
        for _, r in tesla_chargers.iterrows():
            folium.CircleMarker([r.geometry.y, r.geometry.x], radius=6, color='red', fill=True, fill_opacity=0.8).add_to(cl)
        for _, r in top.iterrows():
            c = cmap(r['adjusted'])
            popup = (f"<b>{r['location_name']}</b><br>Score: {r['adjusted']:.3f}<br>"
                     f"Traffic: {r['suitability_score']:.2f}, EV: {r['ev_score']:.2f}<br>"
                     f"Dist to Charger: {int(r['dist_to_tesla'])} m")
            folium.CircleMarker(
                [r.geometry.y, r.geometry.x], radius=10,
                color=c, fill=True, fill_color=c, fill_opacity=0.8,
                popup=popup
            ).add_to(m_r)
        st_folium(m_r, height=900, width=1300)
    with col2:
        st.markdown("#### Top Recommendations")
        st.dataframe(top[['location_name','adjusted']].reset_index(drop=True), use_container_width=True)

# --- Export ---
csv = top[['location_name','adjusted']].to_csv(index=False).encode('utf-8')
st.download_button('Download CSV', csv, 'recommended_sites.csv')

st.caption('Built with Streamlit, Folium, GeoPandas & SafeGraph')
