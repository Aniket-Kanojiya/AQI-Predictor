"""
AQI Predictor — Streamlit Web Application
==========================================
Interactive app that predicts Air Quality Index (AQI) for Indian cities.
Users can either input pollutant values manually or select a historical date
to see recorded vs predicted AQI, along with city-level trend charts.

Run with:  streamlit run app/app.py  (from the project root)
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so we can import from src/
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_cleaning import get_aqi_bucket, AQI_BUCKETS
from src.features import get_season

# ============================= CONFIGURATION ================================

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_aqi.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "aqi_model.pkl"
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"

# Pollutant columns used for manual input
POLLUTANT_COLS = [
    "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
    "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene",
]

# Typical ranges for slider bounds (based on Indian AQI data)
POLLUTANT_RANGES = {
    "PM2.5": (0.0, 500.0, 60.0),
    "PM10": (0.0, 600.0, 120.0),
    "NO": (0.0, 300.0, 20.0),
    "NO2": (0.0, 200.0, 40.0),
    "NOx": (0.0, 400.0, 50.0),
    "NH3": (0.0, 200.0, 25.0),
    "CO": (0.0, 30.0, 2.0),
    "SO2": (0.0, 200.0, 15.0),
    "O3": (0.0, 300.0, 40.0),
    "Benzene": (0.0, 40.0, 3.0),
    "Toluene": (0.0, 80.0, 10.0),
    "Xylene": (0.0, 50.0, 2.0),
}


# ============================= HELPER FUNCTIONS ==============================


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and cache the cleaned AQI dataset."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    return df


@st.cache_resource
def load_model():
    """Load and cache the trained ML model."""
    model = joblib.load(MODEL_PATH)
    return model


@st.cache_data
def load_metrics() -> dict:
    """Load evaluation metrics from JSON."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return {}


@st.cache_resource
def load_feature_columns() -> list:
    """Load the feature columns list that the model expects."""
    cols = joblib.load(FEATURE_COLS_PATH)
    return cols


def build_feature_row(
    city: str,
    date: pd.Timestamp,
    pollutant_values: dict,
    feature_columns: list,
    lag_values: dict = None,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame with all engineered features matching
    what the model was trained on.

    Args:
        city: City name
        date: Timestamp of the prediction
        pollutant_values: Dict mapping pollutant names to float values
        feature_columns: Expected columns in exact order
        lag_values: Optional dict for lag features

    Returns:
        Single-row DataFrame ready for model.predict()
    """
    if lag_values is None:
        lag_values = {}

    row = dict(pollutant_values)

    # Time features
    row["month"] = date.month
    row["day_of_week"] = date.dayofweek
    row["day_of_year"] = date.dayofyear
    row["season"] = get_season(date.month)

    # Lag features
    row["AQI_lag_1"] = lag_values.get("AQI_lag_1", 0.0)
    row["AQI_lag_7"] = lag_values.get("AQI_lag_7", 0.0)
    row["PM25_lag_1"] = lag_values.get("PM25_lag_1", 0.0)

    # City one-hot
    for c in ["Chennai", "Delhi", "Kolkata", "Mumbai"]:
        row[f"City_{c}"] = 1.0 if city == c else 0.0

    # Create DataFrame
    df_row = pd.DataFrame([row])

    # Encode season if it's still a string column
    if "season" in df_row.columns:
        season_dummies = pd.get_dummies(df_row["season"], prefix="season", drop_first=False)
        df_row = pd.concat([df_row.drop(columns=["season"]), season_dummies], axis=1)

    # Align columns to what the model expects
    return df_row.reindex(columns=feature_columns, fill_value=0.0)


def create_aqi_gauge(aqi_value: float) -> go.Figure:
    """
    Create a Plotly gauge chart for AQI visualization.

    Args:
        aqi_value: Predicted or recorded AQI value

    Returns:
        Plotly Figure with a styled gauge
    """
    bucket_label, bucket_color = get_aqi_bucket(aqi_value)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=aqi_value,
        title={"text": f"AQI: {bucket_label}", "font": {"size": 24}},
        number={"font": {"size": 48}},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1},
            "bar": {"color": bucket_color},
            "steps": [{"range": [low, high], "color": f"{color}30"} for low, high, _, color in AQI_BUCKETS],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": aqi_value,
            },
        },
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def render_aqi_result(aqi_val: float, label_prefix: str = "Predicted"):
    """Render gauge chart and colored banner for an AQI value."""
    st.plotly_chart(create_aqi_gauge(aqi_val), use_container_width=True)
    bucket_label, bucket_color = get_aqi_bucket(aqi_val)
    st.markdown(
        f"<div style='text-align:center; padding:8px; "
        f"background-color:{bucket_color}20; border-radius:8px; "
        f"border: 1px solid {bucket_color};'>"
        f"<strong style='color:{bucket_color}; font-size:1.1rem;'>"
        f"{label_prefix}: {aqi_val:.1f} ({bucket_label})</strong></div>",
        unsafe_allow_html=True,
    )


def create_trend_chart(df: pd.DataFrame, cities: list) -> go.Figure:
    """
    Create a Plotly line chart showing AQI trends for one or more cities.
    Each city is plotted as a separate colored line.

    Args:
        df: Cleaned AQI DataFrame
        cities: List of city names to plot

    Returns:
        Plotly Figure with the AQI trend lines
    """
    city_df = df[df["City"].isin(cities)].sort_values("Date").copy()
    if city_df.empty:
        return go.Figure()

    city_df["AQI_rolling"] = city_df.groupby("City")["AQI"].rolling(30, min_periods=1).mean().reset_index(0, drop=True)

    fig = px.line(
        city_df, x="Date", y="AQI_rolling", color="City",
        title="AQI Trend (30-day Rolling Average)",
        labels={"AQI_rolling": "AQI (30-day avg)", "Date": "Date"},
    )

    # Add colored background bands for AQI buckets
    for low, high, label, color in AQI_BUCKETS:
        fig.add_hrect(
            y0=low, y1=high,
            fillcolor=color, opacity=0.08,
            layer="below", line_width=0,
            annotation_text=label,
            annotation_position="right",
            annotation_font_size=9,
            annotation_font_color=color,
        )

    fig.update_layout(
        height=450,
        xaxis_title="Date",
        yaxis_title="AQI",
        yaxis_range=[0, min(city_df["AQI"].max() * 1.1, 550)],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    return fig


# ============================= MAIN APP ======================================


def main():
    """Main Streamlit application entry point."""

    # ---- Page configuration ----
    st.set_page_config(
        page_title="AQI Predictor — Indian Cities",
        page_icon="🌫️",
        layout="wide",
    )

    st.title("🌫️ AQI Predictor for Indian Cities")
    st.markdown(
        "Predict and explore Air Quality Index (AQI) for major Indian cities "
        "using machine learning."
    )

    # ---- Load resources ----
    try:
        df = load_data()
        model = load_model()
        feature_columns = load_feature_columns()
        metrics = load_metrics()
    except FileNotFoundError as e:
        st.error(
            f"**Missing file:** `{e.filename}`\n\n"
            "Please run the full pipeline first:\n"
            "```bash\n"
            "python src/data_cleaning.py\n"
            "python src/features.py\n"
            "python src/train.py\n"
            "```"
        )
        st.stop()

    cities = sorted(df["City"].unique())

    # ======================== SIDEBAR ========================
    st.sidebar.header("⚙️ Configuration")

    # City selection
    selected_city = st.sidebar.selectbox(
        "Select City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0
    )

    # Mode toggle
    mode = st.sidebar.radio(
        "Prediction Mode",
        ["📝 Manual Input", "📅 Historical Date"],
        help="Choose how to provide input for the AQI prediction.",
    )

    # ======================== MAIN CONTENT ========================
    col_left, col_right = st.columns([1, 1])

    # ---------- MANUAL INPUT MODE ----------
    if mode == "📝 Manual Input":
        with col_left:
            st.subheader("Enter Pollutant Levels")
            st.caption("Adjust the sliders to set pollutant concentrations (µg/m³, CO in mg/m³)")

            pollutant_values = {}
            # Create a 2-column layout for sliders
            slider_cols = st.columns(2)
            for i, pollutant in enumerate(POLLUTANT_COLS):
                low, high, default = POLLUTANT_RANGES[pollutant]
                with slider_cols[i % 2]:
                    pollutant_values[pollutant] = st.slider(
                        pollutant,
                        min_value=low,
                        max_value=high,
                        value=default,
                        step=0.1 if high <= 50 else 1.0,
                    )

            # Predict
            today = pd.Timestamp.now()
            feature_row = build_feature_row(
                selected_city, today, pollutant_values, feature_columns
            )
            predicted_aqi = float(model.predict(feature_row)[0])
            predicted_aqi = max(0, min(500, predicted_aqi))  # Clamp to valid range

        with col_right:
            st.subheader("Prediction Result")
            render_aqi_result(predicted_aqi, "Predicted AQI")

    # ---------- HISTORICAL DATE MODE ----------
    else:
        with col_left:
            st.subheader("Select a Historical Date")

            city_df = df[df["City"] == selected_city].sort_values("Date")
            min_date, max_date = city_df["Date"].min().date(), city_df["Date"].max().date()
            selected_date = st.date_input("Date", value=max_date, min_value=min_date, max_value=max_date)

            # Look up recorded data
            mask = (df["City"] == selected_city) & (df["Date"].dt.date == selected_date)
            record = df[mask]

            if record.empty:
                st.warning(f"No data available for {selected_city} on {selected_date}.")
            else:
                row = record.iloc[0]
                recorded_aqi = float(row["AQI"])
                pollutant_values = {col: float(row[col]) for col in POLLUTANT_COLS if col in row.index}
                feature_row = build_feature_row(selected_city, pd.Timestamp(selected_date), pollutant_values, feature_columns)
                predicted_aqi = max(0.0, min(500.0, float(model.predict(feature_row)[0])))

                st.markdown("**Recorded Pollutant Levels:**")
                st.dataframe(pd.DataFrame([pollutant_values]), use_container_width=True)

        with col_right:
            if not record.empty:
                st.subheader("AQI Comparison")
                tab1, tab2 = st.tabs(["📊 Recorded AQI", "🤖 Predicted AQI"])
                with tab1:
                    render_aqi_result(recorded_aqi, "Recorded")
                with tab2:
                    render_aqi_result(predicted_aqi, "Predicted")
                st.metric("Prediction Error", f"{abs(predicted_aqi - recorded_aqi):.1f} AQI points")

    # ======================== TREND CHART ========================
    st.divider()
    st.subheader("📈 AQI Trend")

    # Multi-city selector — compare trends across cities
    trend_cities = st.multiselect(
        "Select cities to compare AQI trends",
        cities,
        default=[selected_city],
        key="trend_city_selector",
    )

    if trend_cities:
        trend_fig = create_trend_chart(df, trend_cities)
        st.plotly_chart(trend_fig, use_container_width=True)
    else:
        st.info("Select at least one city above to view the AQI trend chart.")

    # ======================== MODEL INFO ========================
    with st.expander("🔬 Model Performance & Info"):
        st.markdown(f"**Best Model:** `{metrics.get('best_model', 'N/A')}`")

        if "best_params" in metrics:
            st.markdown(f"**Best Parameters:** `{metrics['best_params']}`")

        # Build comparison table
        # train.py saves keys as model names with spaces, and uppercase metric keys
        model_names = ["Linear Regression", "Random Forest", "XGBoost"]
        table_data = []
        for name in model_names:
            if name in metrics:
                m = metrics[name]
                table_data.append({
                    "Model": name,
                    "RMSE": f"{m.get('RMSE', 'N/A'):.2f}" if isinstance(m.get('RMSE'), (int, float)) else "N/A",
                    "MAE": f"{m.get('MAE', 'N/A'):.2f}" if isinstance(m.get('MAE'), (int, float)) else "N/A",
                    "R²": f"{m.get('R2', 'N/A'):.4f}" if isinstance(m.get('R2'), (int, float)) else "N/A",
                })
        if table_data:
            st.table(pd.DataFrame(table_data).set_index("Model"))

        # Final tuned model metrics
        if "final_tuned_metrics" in metrics:
            st.markdown("---")
            st.markdown("**Final Tuned Model (Test Set):**")
            fm = metrics["final_tuned_metrics"]
            col1, col2, col3 = st.columns(3)
            col1.metric("RMSE", f"{fm.get('RMSE', 0):.2f}")
            col2.metric("MAE", f"{fm.get('MAE', 0):.2f}")
            col3.metric("R²", f"{fm.get('R2', 0):.4f}")

    # ======================== FOOTER ========================
    st.divider()
    st.caption(
        "Built with Streamlit • Data: Air Quality Data in India (2015–2024) • "
        "Model: XGBoost / Random Forest"
    )


if __name__ == "__main__":
    main()
