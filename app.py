"""
SwiftRide Analytics — app.py
Production-ready Streamlit dashboard for ride-hailing analytics.
Senior Python Engineer | Data Visualization Specialist
"""

import sqlite3
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SwiftRide Analytics",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL STYLING
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1623 0%, #1a2235 100%);
        border-right: 1px solid #2a3347;
    }
    section[data-testid="stSidebar"] * { color: #c8d3e8 !important; }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 6px 12px; border-radius: 6px; transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover { background: #2a3347; }

    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
        border: 1px solid #dde3f5;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(99,120,230,0.08);
    }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #6b7a99; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
    [data-testid="stMetricValue"] { font-size: 1.5rem; font-weight: 700; color: #1a2235; }
    [data-testid="stMetricDelta"] { font-size: 0.8rem; font-weight: 600; }

    .stDataFrame { border-radius: 10px; overflow: hidden; border: 1px solid #e4e8f5; }
    hr { border: none; border-top: 1px solid #e4e8f5; margin: 24px 0; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.88rem; letter-spacing: 0.02em; }
    .stTabs [aria-selected="true"] { color: #4361ee; border-bottom-color: #4361ee; }
    h1 { font-weight: 700; color: #1a2235; letter-spacing: -0.02em; }
    h2, h3 { font-weight: 600; color: #2d3a52; }
    .stInfo { background: #eef2ff; border-left-color: #4361ee; }
    .stSuccess { background: #edfaf4; border-left-color: #2ecc71; }
    .stWarning { background: #fffbeb; border-left-color: #f59e0b; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# COLOR PALETTE
# ─────────────────────────────────────────────
PALETTE = {
    "primary":   "#4361ee",
    "secondary": "#3a0ca3",
    "accent":    "#4cc9f0",
    "success":   "#2ecc71",
    "warning":   "#f59e0b",
    "danger":    "#ef233c",
    "seq_blues": "Blues",
    "seq_rdylgn":"RdYlGn",
    "multi":     px.colors.qualitative.Bold,
}

CHART_TEMPLATE = "plotly_white"
CHART_LAYOUT = dict(
    template=CHART_TEMPLATE,
    font=dict(family="DM Sans, sans-serif", size=12, color="#2d3a52"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=48, b=32, l=16, r=16),
    title_font=dict(size=15, color="#1a2235", family="DM Sans, sans-serif"),
)

# ─────────────────────────────────────────────
# DATABASE PATH
# ─────────────────────────────────────────────
DB_PATH = "swiftride.db"

# ─────────────────────────────────────────────
# DATABASE HELPER — safe, cached, resilient
# ─────────────────────────────────────────────
@st.cache_data(ttl=60, show_spinner=False)
def run_query(query: str) -> pd.DataFrame:
    """Execute SQL against swiftride.db. Returns empty DataFrame on any error."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except sqlite3.OperationalError as e:
        st.warning(f"⚠️ Database query issue: `{e}`. Some data may be unavailable.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Unexpected error loading data: `{e}`")
        return pd.DataFrame()


def get_driver_name_col() -> str:
    """
    Auto-detect the driver name column in the drivers table.
    Handles: first_name+last_name, full_name, driver_name, name.
    Falls back to driver_id if nothing matches.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur  = conn.cursor()
        cur.execute("PRAGMA table_info(drivers)")
        cols = [row[1].lower() for row in cur.fetchall()]
        conn.close()
    except Exception:
        return "'Unknown Driver'"

    if "first_name" in cols and "last_name" in cols:
        return "d.first_name || ' ' || d.last_name"
    elif "full_name" in cols:
        return "d.full_name"
    elif "driver_name" in cols:
        return "d.driver_name"
    elif "name" in cols:
        return "d.name"
    else:
        return "CAST(d.driver_id AS TEXT)"


def safe_metric(df: pd.DataFrame, col: str, default=0):
    """Safely extract a scalar from a single-row DataFrame."""
    try:
        val = df[col].iloc[0]
        return val if pd.notna(val) else default
    except Exception:
        return default


def apply_layout(fig: go.Figure, **extra) -> go.Figure:
    """Apply global chart layout to a Plotly figure."""
    fig.update_layout(**{**CHART_LAYOUT, **extra})
    return fig


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🚗 SwiftRide Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["📊 Executive Overview", "🗺️ Trip Analytics", "🧑‍✈️ Driver Performance", "🤖 ML Fare Predictor"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("SwiftRide Analytics v2.0")
    st.caption("Powered by Streamlit + Plotly")

page_key = page.split(" ", 1)[1]


# ══════════════════════════════════════════════
# PAGE 1 — EXECUTIVE OVERVIEW
# ══════════════════════════════════════════════
if page_key == "Executive Overview":
    st.title("📊 Executive Overview")
    st.caption("High-level KPIs, revenue trends, and city performance at a glance.")

    # ── KPI Cards ──────────────────────────────
    with st.spinner("Loading KPIs…"):
        kpi_df = run_query("""
            SELECT
                SUM(fare_pkr)                                              AS total_revenue,
                COUNT(*)                                                    AS total_trips,
                SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END)        AS completed_trips,
                AVG(CASE WHEN status='completed' THEN fare_pkr END)        AS avg_fare,
                COUNT(DISTINCT driver_id)                                   AS active_drivers
            FROM trips
        """)

    total_rev      = safe_metric(kpi_df, "total_revenue", 0)
    total_trips    = int(safe_metric(kpi_df, "total_trips", 0))
    completed      = int(safe_metric(kpi_df, "completed_trips", 0))
    avg_fare       = safe_metric(kpi_df, "avg_fare", 0)
    active_drivers = int(safe_metric(kpi_df, "active_drivers", 0))
    completion_pct = (completed / total_trips * 100) if total_trips else 0
    fare_delta     = avg_fare - 350  # PKR 350 internal benchmark

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Total Revenue",   f"PKR {total_rev:,.0f}")
    c2.metric("🚕 Total Trips",      f"{total_trips:,}")
    c3.metric("✅ Completed Trips",  f"{completed:,}",
              delta=f"{completion_pct:.1f}% completion rate")
    c4.metric("🧾 Avg Fare",         f"PKR {avg_fare:,.0f}",
              delta=f"PKR {fare_delta:+,.0f} vs benchmark")
    c5.metric("🧑‍✈️ Active Drivers", f"{active_drivers:,}")

    st.markdown("---")

    # ── Revenue Trend + Trip Volume ─────────────
    tab_rev, tab_vol = st.tabs(["Monthly Revenue", "Monthly Trip Volume"])

    with st.spinner("Loading trends…"):
        trend_df = run_query("""
            SELECT
                strftime('%Y-%m', trip_date) AS month,
                SUM(fare_pkr)               AS revenue,
                COUNT(*)                    AS trips
            FROM trips
            WHERE status = 'completed'
            GROUP BY month
            ORDER BY month
        """)

    with tab_rev:
        if not trend_df.empty:
            latest_month = trend_df["month"].iloc[-1]
            fig_rev = px.area(
                trend_df, x="month", y="revenue",
                title="Monthly Revenue Trend",
                labels={"month": "Month", "revenue": "Revenue (PKR)"},
                color_discrete_sequence=[PALETTE["primary"]],
            )
            fig_rev.add_vline(x=latest_month, line_dash="dot",
                              line_color=PALETTE["danger"], line_width=1.5)
            fig_rev.add_annotation(
                x=latest_month, y=trend_df["revenue"].max() * 0.95,
                text="Latest Month", showarrow=True, arrowhead=2,
                arrowcolor=PALETTE["danger"],
                bgcolor=PALETTE["danger"], font=dict(color="white", size=11),
                xanchor="left", yref="y",
            )
            fig_rev.update_layout(xaxis_tickangle=-45)
            apply_layout(fig_rev)
            st.plotly_chart(fig_rev, use_container_width=True)
        else:
            st.info("No revenue data available.")

    with tab_vol:
        if not trend_df.empty:
            fig_vol = px.bar(
                trend_df, x="month", y="trips",
                title="Monthly Trip Volume",
                labels={"month": "Month", "trips": "Trips"},
                color_discrete_sequence=[PALETTE["accent"]],
            )
            fig_vol.update_layout(xaxis_tickangle=-45)
            apply_layout(fig_vol)
            st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.info("No trip volume data available.")

    st.markdown("---")

    # ── City Bar + Fleet Pie ────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        with st.spinner("Loading city data…"):
            city_df = run_query("""
                SELECT c.city_name, COUNT(*) AS trip_count
                FROM trips t
                JOIN cities c ON t.city_id = c.city_id
                GROUP BY c.city_name
                ORDER BY trip_count DESC
            """)
        if not city_df.empty:
            fig_city = px.bar(
                city_df, x="trip_count", y="city_name",
                orientation="h",
                title="Trips by City",
                labels={"city_name": "City", "trip_count": "Trips"},
                color="trip_count",
                color_continuous_scale=PALETTE["seq_blues"],
            )
            fig_city.update_layout(
                yaxis={"categoryorder": "total ascending"},
                coloraxis_showscale=False,
            )
            apply_layout(fig_city)
            st.plotly_chart(fig_city, use_container_width=True)
        else:
            st.info("No city data available.")

    with col2:
        with st.spinner("Loading fleet data…"):
            fleet_df = run_query("""
                SELECT vehicle_type, COUNT(*) AS trips
                FROM trips
                WHERE status = 'completed'
                GROUP BY vehicle_type
            """)
        if not fleet_df.empty:
            fig_fleet = px.pie(
                fleet_df, names="vehicle_type", values="trips",
                title="Fleet Mix",
                color_discrete_sequence=PALETTE["multi"],
                hole=0.4,
            )
            fig_fleet.update_traces(textposition="outside", textinfo="percent+label")
            apply_layout(fig_fleet)
            st.plotly_chart(fig_fleet, use_container_width=True)
        else:
            st.info("No fleet data available.")

    st.markdown("---")

    # ── City Performance Summary Table ──────────
    st.subheader("City Performance Summary")
    with st.spinner("Loading city summary…"):
        summary_df = run_query("""
            SELECT
                c.city_name                                                          AS City,
                COUNT(*)                                                             AS total_trips,
                COALESCE(SUM(CASE WHEN t.status='completed' THEN t.fare_pkr END),0) AS total_revenue,
                COALESCE(AVG(CASE WHEN t.status='completed' THEN t.fare_pkr END),0) AS avg_fare,
                ROUND(
                    100.0 * SUM(CASE WHEN t.status='completed' THEN 1 ELSE 0 END) / COUNT(*), 1
                )                                                                    AS completion_rate_pct
            FROM trips t
            JOIN cities c ON t.city_id = c.city_id
            GROUP BY c.city_name
            ORDER BY total_trips DESC
        """)

    if not summary_df.empty:
        overall_avg = summary_df["avg_fare"].mean()
        summary_df["vs_avg_pkr"] = (summary_df["avg_fare"] - overall_avg).round(0)

        display_df = summary_df.copy()
        display_df["total_revenue"]       = display_df["total_revenue"].map("PKR {:,.0f}".format)
        display_df["avg_fare"]            = display_df["avg_fare"].map("PKR {:,.0f}".format)
        display_df["completion_rate_pct"] = display_df["completion_rate_pct"].map("{:.1f}%".format)
        display_df["vs_avg_pkr"]          = display_df["vs_avg_pkr"].map("{:+,.0f} PKR".format)
        display_df = display_df.rename(columns={
            "total_trips": "Total Trips",
            "total_revenue": "Total Revenue",
            "avg_fare": "Avg Fare",
            "completion_rate_pct": "Completion %",
            "vs_avg_pkr": "Fare vs Avg",
        })
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No city summary data available.")


# ══════════════════════════════════════════════
# PAGE 2 — TRIP ANALYTICS
# ══════════════════════════════════════════════
elif page_key == "Trip Analytics":
    st.title("🗺️ Trip Analytics")
    st.caption("Demand patterns, fare dynamics, and weather impact analysis.")

    # ── Demand Heatmap ──────────────────────────
    st.subheader("Trip Demand Heatmap")
    with st.spinner("Loading heatmap…"):
        heat_df = run_query("""
            SELECT trip_hour, day_of_week, COUNT(*) AS trip_count
            FROM trips
            GROUP BY trip_hour, day_of_week
        """)

    if not heat_df.empty:
        heat_df["trip_hour"]   = pd.to_numeric(heat_df["trip_hour"],   errors="coerce")
        heat_df["day_of_week"] = pd.to_numeric(heat_df["day_of_week"], errors="coerce")
        heat_df = heat_df.dropna(subset=["trip_hour", "day_of_week"])

        pivot = heat_df.pivot_table(
            index="day_of_week", columns="trip_hour",
            values="trip_count", aggfunc="sum", fill_value=0,
        )
        # Force chronological order: days 0–6, hours 0–23
        pivot = pivot.reindex(index=range(7), columns=range(24), fill_value=0)

        day_labels  = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        hour_labels = [f"{h:02d}:00" for h in range(24)]

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=hour_labels,
            y=day_labels,
            colorscale=[[0, "#eef2ff"], [0.5, "#4361ee"], [1, "#3a0ca3"]],
            hovertemplate="Hour: %{x}<br>Day: %{y}<br>Trips: %{z}<extra></extra>",
        ))
        fig_heat.update_layout(
            title="Trip Demand by Hour & Day",
            xaxis=dict(title="Hour of Day", tickangle=-45, tickfont=dict(size=10)),
            yaxis=dict(title="Day of Week"),
        )
        apply_layout(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No heatmap data available.")

    st.markdown("---")

    # ── Fare by Vehicle + Fare vs Distance ──────
    col1, col2 = st.columns(2)

    with col1:
        with st.spinner("Loading fare by vehicle…"):
            veh_fare = run_query("""
                SELECT vehicle_type, AVG(fare_pkr) AS avg_fare
                FROM trips WHERE status='completed'
                GROUP BY vehicle_type
                ORDER BY avg_fare DESC
            """)
        if not veh_fare.empty:
            fig_vf = px.bar(
                veh_fare, x="vehicle_type", y="avg_fare",
                title="Average Fare by Vehicle Type",
                labels={"vehicle_type": "Vehicle Type", "avg_fare": "Avg Fare (PKR)"},
                color="avg_fare",
                color_continuous_scale=PALETTE["seq_blues"],
            )
            fig_vf.update_layout(coloraxis_showscale=False)
            apply_layout(fig_vf)
            st.plotly_chart(fig_vf, use_container_width=True)
        else:
            st.info("No fare-by-vehicle data available.")

    with col2:
        with st.spinner("Loading scatter…"):
            scatter_df = run_query("""
                SELECT distance_km, fare_pkr, vehicle_type
                FROM trips
                WHERE status='completed'
                ORDER BY RANDOM()
                LIMIT 600
            """)
        if not scatter_df.empty:
            # trendline removed to avoid statsmodels dependency.
            # To re-enable: pip install statsmodels, then add:
            #   trendline="ols", trendline_scope="overall",
            #   trendline_color_override=PALETTE["danger"]
            fig_sc = px.scatter(
                scatter_df, x="distance_km", y="fare_pkr",
                color="vehicle_type",
                title="Fare vs Distance",
                labels={"distance_km": "Distance (km)", "fare_pkr": "Fare (PKR)"},
                opacity=0.55,
                color_discrete_sequence=PALETTE["multi"],
            )
            apply_layout(fig_sc)
            st.plotly_chart(fig_sc, use_container_width=True)
        else:
            st.info("No scatter data available.")

    st.markdown("---")

    # ── Peak vs Off-Peak ────────────────────────
    with st.spinner("Loading peak/off-peak data…"):
        peak_df = run_query("""
            SELECT vehicle_type, is_peak_hour, AVG(fare_pkr) AS avg_fare
            FROM trips
            WHERE status='completed'
            GROUP BY vehicle_type, is_peak_hour
        """)

    if not peak_df.empty:
        peak_df["Period"] = peak_df["is_peak_hour"].map({1: "Peak", 0: "Off-Peak"})
        fig_peak = px.bar(
            peak_df, x="vehicle_type", y="avg_fare",
            color="Period", barmode="group",
            title="Peak vs Off-Peak Fares by Vehicle Type",
            labels={"vehicle_type": "Vehicle Type", "avg_fare": "Avg Fare (PKR)"},
            color_discrete_map={"Peak": PALETTE["primary"], "Off-Peak": PALETTE["accent"]},
        )
        apply_layout(fig_peak)
        st.plotly_chart(fig_peak, use_container_width=True)
    else:
        st.info("No peak/off-peak data available.")

    st.markdown("---")

    # ── Rain Impact ─────────────────────────────
    st.subheader("🌧️ Rain Impact on Fares & Volume")
    with st.spinner("Loading rain impact…"):
        rain_df = run_query("""
            SELECT
                is_raining,
                AVG(fare_pkr)                               AS avg_fare,
                COUNT(*) * 1.0 / COUNT(DISTINCT trip_date)  AS avg_trips_per_day
            FROM trips
            WHERE status='completed'
            GROUP BY is_raining
        """)

    if not rain_df.empty:
        rain_row    = rain_df[rain_df["is_raining"] == 1].iloc[0] if 1 in rain_df["is_raining"].values else None
        no_rain_row = rain_df[rain_df["is_raining"] == 0].iloc[0] if 0 in rain_df["is_raining"].values else None

        if rain_row is not None and no_rain_row is not None:
            fare_diff  = rain_row["avg_fare"] - no_rain_row["avg_fare"]
            trips_diff = rain_row["avg_trips_per_day"] - no_rain_row["avg_trips_per_day"]

            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("🌧️ Avg Fare (Rain)",    f"PKR {rain_row['avg_fare']:,.0f}",
                       delta=f"PKR {fare_diff:+,.0f} vs clear")
            rc2.metric("☀️ Avg Fare (Clear)",    f"PKR {no_rain_row['avg_fare']:,.0f}")
            rc3.metric("🌧️ Trips/Day (Rain)",    f"{rain_row['avg_trips_per_day']:.1f}",
                       delta=f"{trips_diff:+.1f} vs clear")
            rc4.metric("☀️ Trips/Day (Clear)",   f"{no_rain_row['avg_trips_per_day']:.1f}")

        st.info(
            "💡 **Surge Pricing During Rain**: SwiftRide applies surge multipliers "
            "during rainy periods to compensate drivers and balance demand."
        )
    else:
        st.info("No rain impact data available.")


# ══════════════════════════════════════════════
# PAGE 3 — DRIVER PERFORMANCE
# ══════════════════════════════════════════════
elif page_key == "Driver Performance":
    st.title("🧑‍✈️ Driver Performance")
    st.caption("Earnings, ratings, and activity metrics for the driver fleet.")

    # ── Auto-detect driver name column ──────────
    driver_name_expr = get_driver_name_col()

    # ── Top 10 Drivers (LEFT JOIN for 0-review drivers) ─
    with st.spinner("Loading driver data…"):
        top_drivers = run_query(f"""
            SELECT
                {driver_name_expr}                                         AS driver_name,
                c.city_name                                                AS city,
                d.vehicle_type,
                COUNT(t.trip_id)                                           AS total_trips,
                COALESCE(SUM(t.fare_pkr), 0)                              AS total_earnings,
                ROUND(COALESCE(AVG(r.rider_rating_given), 0), 2)          AS avg_rating,
                ROUND(
                    100.0 * SUM(CASE WHEN t.status='completed' THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(t.trip_id), 0), 1
                )                                                          AS completion_rate
            FROM trips t
            JOIN drivers d   ON t.driver_id  = d.driver_id
            JOIN cities  c   ON t.city_id    = c.city_id
            LEFT JOIN reviews r ON t.trip_id = r.trip_id
            GROUP BY d.driver_id
            ORDER BY total_earnings DESC
            LIMIT 10
        """)

    st.subheader("🏆 Top 10 Drivers by Earnings")
    if not top_drivers.empty:
        display_drivers = top_drivers.copy()
        display_drivers["total_earnings"]  = display_drivers["total_earnings"].map("PKR {:,.0f}".format)
        display_drivers["completion_rate"] = display_drivers["completion_rate"].map("{:.1f}%".format)
        display_drivers = display_drivers.rename(columns={
            "driver_name": "Driver",
            "city": "City",
            "vehicle_type": "Vehicle",
            "total_trips": "Trips",
            "total_earnings": "Total Earnings",
            "avg_rating": "Avg Rating ⭐",
            "completion_rate": "Completion %",
        })
        st.dataframe(display_drivers, use_container_width=True)
    else:
        st.info("No driver data available.")

    st.markdown("---")

    # ── City Ratings + Vehicle Earnings ─────────
    col1, col2 = st.columns(2)

    with col1:
        with st.spinner("Loading city ratings…"):
            city_rating = run_query("""
                SELECT c.city_name, ROUND(AVG(r.rider_rating_given), 2) AS avg_rating
                FROM reviews r
                JOIN trips t  ON r.trip_id = t.trip_id
                JOIN cities c ON t.city_id = c.city_id
                GROUP BY c.city_name
                ORDER BY avg_rating DESC
            """)
        if not city_rating.empty:
            overall_rating = city_rating["avg_rating"].mean()
            fig_cr = px.bar(
                city_rating, x="city_name", y="avg_rating",
                title="Avg Driver Rating by City",
                labels={"city_name": "City", "avg_rating": "Avg Rating"},
                color="avg_rating",
                color_continuous_scale=PALETTE["seq_rdylgn"],
                range_color=[city_rating["avg_rating"].min() - 0.1,
                             city_rating["avg_rating"].max() + 0.1],
            )
            fig_cr.add_hline(
                y=overall_rating, line_dash="dot",
                line_color=PALETTE["danger"],
                annotation_text=f"Fleet Avg: {overall_rating:.2f}",
                annotation_position="bottom right",
            )
            fig_cr.update_layout(coloraxis_showscale=False)
            apply_layout(fig_cr)
            st.plotly_chart(fig_cr, use_container_width=True)
        else:
            st.info("No city rating data available.")

    with col2:
        with st.spinner("Loading vehicle earnings…"):
            earn_veh = run_query("""
                SELECT vehicle_type, AVG(fare_pkr) AS avg_earnings_per_trip
                FROM trips
                WHERE status='completed'
                GROUP BY vehicle_type
                ORDER BY avg_earnings_per_trip DESC
            """)
        if not earn_veh.empty:
            fig_ev = px.bar(
                earn_veh, x="vehicle_type", y="avg_earnings_per_trip",
                title="Avg Earnings per Trip by Vehicle Type",
                labels={"vehicle_type": "Vehicle Type",
                        "avg_earnings_per_trip": "Avg Earnings (PKR)"},
                color="vehicle_type",
                color_discrete_sequence=PALETTE["multi"],
            )
            apply_layout(fig_ev)
            st.plotly_chart(fig_ev, use_container_width=True)
        else:
            st.info("No vehicle earnings data available.")

    st.markdown("---")

    # ── Active Drivers Trend ─────────────────────
    with st.spinner("Loading active driver trend…"):
        active_df = run_query("""
            SELECT
                strftime('%Y-%m', trip_date) AS month,
                COUNT(DISTINCT driver_id)    AS active_drivers
            FROM trips
            WHERE status='completed'
            GROUP BY month
            ORDER BY month
        """)

    if not active_df.empty:
        fig_ad = px.line(
            active_df, x="month", y="active_drivers",
            title="Active Drivers Per Month",
            labels={"month": "Month", "active_drivers": "Active Drivers"},
            markers=True,
            color_discrete_sequence=[PALETTE["primary"]],
        )
        fig_ad.update_traces(line=dict(width=2.5), marker=dict(size=7))
        fig_ad.update_layout(xaxis_tickangle=-45)
        apply_layout(fig_ad)
        st.plotly_chart(fig_ad, use_container_width=True)
    else:
        st.info("No active driver trend data available.")

    st.markdown("---")

    # ── Rating Distribution ──────────────────────
    with st.spinner("Loading rating distribution…"):
        rating_df = run_query(
            "SELECT rider_rating_given FROM reviews WHERE rider_rating_given IS NOT NULL"
        )

    if not rating_df.empty:
        fig_rh = px.histogram(
            rating_df, x="rider_rating_given",
            nbins=20,
            title="Distribution of Driver Ratings",
            labels={"rider_rating_given": "Rating Given by Rider"},
            color_discrete_sequence=[PALETTE["primary"]],
        )
        fig_rh.update_traces(marker_line_width=1.2, marker_line_color="white")
        apply_layout(fig_rh)
        st.plotly_chart(fig_rh, use_container_width=True)
    else:
        st.info("No rating distribution data available.")


# ══════════════════════════════════════════════
# PAGE 4 — ML FARE PREDICTOR
# ══════════════════════════════════════════════
elif page_key == "ML Fare Predictor":
    st.title("🤖 ML Fare Predictor")
    st.caption("Random Forest model trained on completed trips. Predict fares from real-time inputs.")

    @st.cache_resource(show_spinner=False)
    def train_model():
        """Train RandomForest. Returns tuple; last element is error string or None."""
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("""
                SELECT distance_km, duration_mins, is_peak_hour, is_raining,
                       surge_multiplier, day_of_week, trip_hour, vehicle_type, fare_pkr
                FROM trips
                WHERE status = 'completed'
            """, conn)
            conn.close()
        except Exception as e:
            return None, None, None, None, None, None, None, None, None, str(e)

        if df.empty or len(df) < 50:
            return None, None, None, None, None, None, None, None, None, "Insufficient training data."

        # Capture vehicle types BEFORE encoding
        veh_types_raw = sorted(df["vehicle_type"].dropna().unique().tolist())

        df = pd.get_dummies(df, columns=["vehicle_type"], drop_first=False)

        base_cols    = ["distance_km", "duration_mins", "is_peak_hour", "is_raining",
                        "surge_multiplier", "day_of_week", "trip_hour"]
        veh_dummies  = sorted([c for c in df.columns if c.startswith("vehicle_type_")])
        feature_cols = base_cols + veh_dummies

        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        X = df[feature_cols].fillna(0)
        y = df["fare_pkr"].fillna(df["fare_pkr"].median())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestRegressor(
            n_estimators=150, random_state=42,
            n_jobs=-1, max_depth=12, min_samples_leaf=5
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2   = r2_score(y_test, y_pred)
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        return model, feature_cols, veh_types_raw, r2, mae, rmse, X_test, y_test, y_pred, None

    with st.spinner("🔧 Training model… this may take a moment."):
        result = train_model()

    model, feature_cols, veh_types_raw, r2, mae, rmse, X_test, y_test, y_pred, train_err = result

    if train_err:
        st.error(f"❌ Model training failed: {train_err}")
        st.stop()

    # ── Model Performance ────────────────────────
    st.subheader("📐 Model Performance")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("R² Score",   f"{r2:.4f}",   delta="Higher = better fit")
    mc2.metric("MAE (PKR)",  f"{mae:,.0f}", delta="Lower = better precision")
    mc3.metric("RMSE (PKR)", f"{rmse:,.0f}")

    if r2 > 0.85:
        st.success(f"✅ Excellent model fit — R² = {r2:.4f}")
    elif r2 > 0.70:
        st.info(f"ℹ️ Reasonable model fit — R² = {r2:.4f}. More features could help.")
    else:
        st.warning(f"⚠️ Weak model fit — R² = {r2:.4f}. Consider adding more features.")

    st.markdown("---")

    # ── Feature Importance + Actual vs Predicted ─
    tab_fi, tab_avp = st.tabs(["Feature Importance", "Actual vs Predicted"])

    with tab_fi:
        importance_df = pd.DataFrame({
            "Feature": feature_cols,
            "Importance": model.feature_importances_,
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(
            importance_df, x="Importance", y="Feature",
            orientation="h",
            title="Feature Importance (Random Forest)",
            labels={"Importance": "Importance Score", "Feature": ""},
            color="Importance",
            color_continuous_scale=PALETTE["seq_blues"],
        )
        fig_fi.update_layout(coloraxis_showscale=False)
        apply_layout(fig_fi, height=500)
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab_avp:
        avp_df = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
        fig_avp = px.scatter(
            avp_df, x="Actual", y="Predicted",
            title="Actual vs Predicted Fares",
            labels={"Actual": "Actual Fare (PKR)", "Predicted": "Predicted Fare (PKR)"},
            opacity=0.40,
            color_discrete_sequence=[PALETTE["primary"]],
        )
        min_val = float(avp_df[["Actual", "Predicted"]].min().min())
        max_val = float(avp_df[["Actual", "Predicted"]].max().max())
        fig_avp.add_shape(
            type="line", x0=min_val, y0=min_val, x1=max_val, y1=max_val,
            line=dict(color=PALETTE["danger"], dash="dash", width=2),
        )
        fig_avp.add_annotation(
            x=max_val * 0.82, y=max_val * 0.90,
            text="Perfect Prediction Line",
            showarrow=False,
            font=dict(color=PALETTE["danger"], size=11),
        )
        apply_layout(fig_avp)
        st.plotly_chart(fig_avp, use_container_width=True)

    st.markdown("---")

    # ── Live Fare Estimator ──────────────────────
    st.subheader("💡 Live Fare Estimator")

    veh_types_display = veh_types_raw if veh_types_raw else ["Standard"]

    with st.container():
        pc1, pc2 = st.columns(2)
        with pc1:
            sel_vehicle  = st.selectbox("Vehicle Type", options=veh_types_display)
            sel_distance = st.slider("Distance (km)", min_value=1, max_value=30, value=5)
            sel_hour     = st.slider("Trip Hour (0–23)", min_value=0, max_value=23, value=8)
        with pc2:
            sel_day_name = st.selectbox(
                "Day of Week",
                options=["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"],
            )
            sel_raining = st.checkbox("🌧️ Is it raining?")

    day_map  = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                "Friday": 4, "Saturday": 5, "Sunday": 6}
    sel_day  = day_map[sel_day_name]
    is_peak  = 1 if (7 <= sel_hour <= 9 or 17 <= sel_hour <= 20) else 0
    is_rain  = 1 if sel_raining else 0
    surge    = 1.5 if (is_peak and sel_raining) else 1.3 if is_peak else 1.0
    duration = sel_distance * 4

    # Build input that exactly matches training feature set
    input_dict: dict = {
        "distance_km":      [sel_distance],
        "duration_mins":    [duration],
        "is_peak_hour":     [is_peak],
        "is_raining":       [is_rain],
        "surge_multiplier": [surge],
        "day_of_week":      [sel_day],
        "trip_hour":        [sel_hour],
    }
    for vt in veh_types_display:
        input_dict[f"vehicle_type_{vt}"] = [1 if vt == sel_vehicle else 0]

    input_df = pd.DataFrame(input_dict)

    # Align to exact training columns — add missing as 0, reorder
    for col in feature_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_cols]

    pred_fare = model.predict(input_df)[0]

    avg_fare_df = run_query(f"""
        SELECT AVG(fare_pkr) AS avg_fare
        FROM trips
        WHERE status='completed'
          AND vehicle_type='{sel_vehicle}'
    """)
    actual_avg = safe_metric(avg_fare_df, "avg_fare", pred_fare)
    diff       = pred_fare - actual_avg
    diff_sign  = "▲" if diff >= 0 else "▼"
    diff_label = f"{diff_sign} PKR {abs(diff):,.0f} vs fleet avg"

    pr1, pr2, pr3 = st.columns(3)
    pr1.metric("🔮 Predicted Fare",             f"PKR {pred_fare:,.0f}")
    pr2.metric(f"📊 Fleet Avg ({sel_vehicle})", f"PKR {actual_avg:,.0f}")
    pr3.metric("📉 Difference",                 diff_label)

    st.caption(
        f"📌 Inputs · {sel_distance} km · {duration} min · Hour {sel_hour:02d}:00 · "
        f"{'Peak 🔴' if is_peak else 'Off-Peak 🟢'} · Surge ×{surge} · "
        f"{'Rainy 🌧️' if sel_raining else 'Clear ☀️'} · Day: {sel_day_name}"
    )
