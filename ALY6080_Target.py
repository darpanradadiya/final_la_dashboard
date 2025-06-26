import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error

# Configure full-width layout and hide Streamlit footer
st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("C:/Users/vagha/imputed_combined_dataset.csv")

df = load_data()

# Sidebar Store Selection
st.sidebar.header("🔍 Select a Store")
df['store_display'] = df['brands'].fillna('Unknown') + " — " + df['street_address'].fillna('No Address')
store_map = dict(zip(df['store_display'], df['placekey']))
selected_display = st.sidebar.selectbox("Choose a Store", list(store_map.keys()))
selected_placekey = store_map[selected_display]
store_data = df[df['placekey'] == selected_placekey].sort_values('date_range_start').reset_index(drop=True)

# Sidebar Busiest Store Info
avg_visits = df.groupby("placekey")['raw_visit_counts'].mean()
max_placekey = avg_visits.idxmax()
store_row = df[df['placekey'] == max_placekey].iloc[0]
store_display_max = f"{store_row['brands']} — {store_row['street_address']}"
st.sidebar.subheader("🔥 Busiest Store Alert")
st.sidebar.write(f"🏆 Most Visited: **{store_display_max}**")
st.sidebar.metric("Avg Weekly Visits", int(avg_visits.max()))

# KPI Row (Top)
st.markdown("## 🏬 Target Foot Traffic Forecast Dashboard")
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("📌 Selected Store", selected_display.split('—')[0].strip())
kpi2.metric("📅 Data Points", len(store_data))
kpi3.metric("📊 Avg Visits", int(store_data['raw_visit_counts'].mean()))

# Prepare data for model
y = store_data["raw_visit_counts"]
X = store_data.drop(columns=["raw_visit_counts", "placekey", "date_range_start", "date_range_end"], errors='ignore')
X = X.drop(columns=[col for col in X if "visit" in col and col != "visit_counts_from_last_week"], errors='ignore')
X = X.select_dtypes(include=["number"])
X = X + np.random.normal(0, 0.1, X.shape)
valid_cols = X.columns[X.notna().any()].tolist()
X_valid = X[valid_cols]
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_valid), columns=valid_cols, index=X.index)

# Forecasting logic
if len(X_imputed) >= 10:
    X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    forecast = model.predict([X_imputed.iloc[-1]])[0]
    staff_pct = round(min(100, forecast * 0.6), 1)
    inventory_pct = round(min(100, forecast * 0.8), 1)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
else:
    forecast = staff_pct = inventory_pct = rmse = None

# Dashboard Body: 3 Column Layout
col1, col2, col3 = st.columns([1.1, 1.4, 1.5])

# Column 1 - Map Only
with col1:
    # Mark the selected store
    df['is_selected'] = df['placekey'] == selected_placekey

    # Prepare map data with only necessary columns
    map_df = df[['latitude', 'longitude', 'store_display', 'is_selected']].dropna()

    # Plotly map: selected store in red, others in blue
    fig_map = px.scatter_mapbox(
        map_df,
        lat="latitude",
        lon="longitude",
        color="is_selected",
        color_discrete_map={True: "red", False: "blue"},
        hover_name="store_display",
        zoom=10,
        height=300
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    st.plotly_chart(fig_map, use_container_width=True)


# Column 2 - Line Chart + Table
with col2:
    st.markdown("""
            <div style="padding: 0.2rem; border-radius: 10px; border: 1px solid #ddd; background-color: #f9f9f9;">
                <h4>📈 Foot Traffic Trend</h4>""", unsafe_allow_html=True)
    fig = px.line(store_data, x='date_range_start', y='raw_visit_counts')
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=260)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Raw Visit Data", expanded=False):
        st.dataframe(store_data[['date_range_start', 'raw_visit_counts']], height=150)
    st.markdown("</div>", unsafe_allow_html=True)
# Column 3 - Operational Readiness + Forecast Overview with Boxed Style
with col3:
    with st.container():
        st.markdown("""
            <div style="padding: 0.2rem; border-radius: 10px; border: 1px solid #ddd; background-color: #f9f9f9;">
                <h4>🛋️ Operational Readiness</h4>
                <p>Operational planning values are derived from forecasted visits.</p>
        """, unsafe_allow_html=True)

        if 'forecast' in locals():
            st.progress(min(100, forecast * 0.6) / 100, text="Staffing Readiness")
            st.progress(min(100, forecast * 0.8) / 100, text="Inventory Readiness")
        else:
            st.info("Forecast not available.")

        st.markdown("</div>", unsafe_allow_html=True)

        with st.container():
         st.markdown("""
            <div style="margin-top: 1rem; padding: 0.2rem; border-radius: 10px; border: 1px solid #ddd; background-color: #f9f9f9;">
                <h4>📈 Forecast Overview</h4>
        """, unsafe_allow_html=True)

        if 'forecast' in locals():
            forecast_col1, forecast_col2, forecast_col3 = st.columns(3)
            forecast_col1.metric("📈 Forecasted Visits", int(forecast))
            forecast_col2.metric("👥 Staff Required", f"{staff_pct}%")
            forecast_col3.metric("📦 Inventory Need", f"{inventory_pct}%")
            st.caption(f"Model → R²: {r2_score(y_test, y_pred):.3f}, RMSE: {rmse:.2f}")
        else:
            st.warning("Not enough data to display forecast.")

        st.markdown("</div>", unsafe_allow_html=True)

