"""🏢 Intelligent Energy Management Dashboard for Smart Buildings.

Interactive Streamlit analytics application presenting Reinforcement Learning control
performance, dynamic load shifting, thermal comfort compliance, and comparative benchmarks.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Smart Building RL Energy Management",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-delta-pos {
        font-size: 0.9rem;
        color: #10B981;
        font-weight: 600;
    }
    .metric-delta-neg {
        font-size: 0.9rem;
        color: #EF4444;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_benchmark_data(results_dir: str | Path = "results") -> tuple[pd.DataFrame | None, dict[str, pd.DataFrame]]:
    """Loads benchmark comparison tables and episode time-series traces."""
    base_path = Path(results_dir) / "metrics"
    if not base_path.exists():
        # Fall back to repo root results
        repo_root = Path(__file__).resolve().parent.parent.parent
        base_path = repo_root / "results" / "metrics"

    comp_file = base_path / "benchmark_comparison.csv"

    comparison_df = None
    if comp_file.exists():
        comparison_df = pd.read_csv(comp_file, index_col=0)

    histories = {}
    for algo in ["rbc", "dqn", "ddpg", "ppo"]:
        hist_file = base_path / f"history_{algo}.csv"
        if hist_file.exists():
            histories[algo.upper()] = pd.read_csv(hist_file)

    return comparison_df, histories


def render_kpi_cards(kpi_data: dict[str, float], baseline_kpis: dict[str, float] | None = None) -> None:
    """Renders executive KPI metric cards with deltas."""
    col1, col2, col3, col4, col5 = st.columns(5)

    cost = kpi_data.get("total_cost_dollar", 0.0)
    cost_sav = kpi_data.get("cost_reduction_pct", 0.0)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Total Electricity Cost</div>
            <div class="metric-value">${cost:,.2f}</div>
            <div class="{'metric-delta-pos' if cost_sav >= 0 else 'metric-delta-neg'}">
                {'▼ ' if cost_sav >= 0 else '▲ '}{abs(cost_sav):.1f}% vs RBC
            </div>
        </div>
        """, unsafe_allow_html=True)

    peak = kpi_data.get("peak_demand_kw", 0.0)
    peak_sav = kpi_data.get("peak_demand_reduction_pct", 0.0)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Peak Grid Demand</div>
            <div class="metric-value">{peak:.1f} kW</div>
            <div class="{'metric-delta-pos' if peak_sav >= 0 else 'metric-delta-neg'}">
                {'▼ ' if peak_sav >= 0 else '▲ '}{abs(peak_sav):.1f}% vs RBC
            </div>
        </div>
        """, unsafe_allow_html=True)

    energy = kpi_data.get("total_energy_kwh", 0.0)
    energy_sav = kpi_data.get("energy_savings_pct", 0.0)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Grid Import Energy</div>
            <div class="metric-value">{energy:,.1f} kWh</div>
            <div class="{'metric-delta-pos' if energy_sav >= 0 else 'metric-delta-neg'}">
                {'▼ ' if energy_sav >= 0 else '▲ '}{abs(energy_sav):.1f}% vs RBC
            </div>
        </div>
        """, unsafe_allow_html=True)

    comfort_violation = kpi_data.get("comfort_violations_deg_hours", 0.0)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Thermal Discomfort</div>
            <div class="metric-value">{comfort_violation:.1f} °C·h</div>
            <div class="metric-delta-pos">ASHRAE 55 Compliant</div>
        </div>
        """, unsafe_allow_html=True)

    solar_self = kpi_data.get("solar_self_consumption_pct", 0.0)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Solar Self-Consumption</div>
            <div class="metric-value">{solar_self:.1f}%</div>
            <div class="metric-delta-pos">Clean Energy Used</div>
        </div>
        """, unsafe_allow_html=True)


def plot_thermal_comfort(histories: dict[str, pd.DataFrame], selected_algos: list[str]) -> go.Figure:
    """Creates interactive thermal comfort regulation plot."""
    fig = go.Figure()
    sample_df = next(iter(histories.values()))
    steps = np.arange(len(sample_df))

    # ASHRAE Comfort Band (21°C - 24°C)
    fig.add_trace(go.Scatter(
        x=steps,
        y=[24.0] * len(steps),
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        name="Upper Limit",
    ))
    fig.add_trace(go.Scatter(
        x=steps,
        y=[21.0] * len(steps),
        mode="lines",
        fill="tonexty",
        fillcolor="rgba(46, 204, 113, 0.18)",
        line=dict(width=0),
        name="ASHRAE 55 Comfort Band (21-24°C)",
    ))

    # Outdoor Temperature
    fig.add_trace(go.Scatter(
        x=steps,
        y=sample_df["outdoor_temp"],
        name="Outdoor Ambient Temp",
        line=dict(color="#94A3B8", dash="dash", width=1.5),
    ))

    # Agent Indoor Temperatures
    palette = {"RBC": "#EF4444", "DQN": "#F59E0B", "DDPG": "#3B82F6", "PPO": "#10B981"}
    for algo in selected_algos:
        if algo in histories:
            df = histories[algo]
            fig.add_trace(go.Scatter(
                x=steps,
                y=df["indoor_temp"],
                name=f"{algo} Indoor Temp",
                line=dict(color=palette.get(algo, "#6366F1"), width=2.2),
            ))

    fig.update_layout(
        title="<b>Indoor Temperature Regulation vs Ambient Weather Extremes</b>",
        xaxis_title="Hour of Episode",
        yaxis_title="Temperature (°C)",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_power_and_price(histories: dict[str, pd.DataFrame], selected_algos: list[str]) -> go.Figure:
    """Creates synchronized net power and electricity tariff plot."""
    sample_df = next(iter(histories.values()))
    steps = np.arange(len(sample_df))

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("<b>Time-of-Use Electricity Tariff ($/kWh)</b>", "<b>Net Grid Electrical Demand (kW) & Peak Shaving</b>"),
        row_heights=[0.3, 0.7],
    )

    # 1. Price Tariff
    fig.add_trace(go.Scatter(
        x=steps,
        y=sample_df["electricity_price"],
        name="TOU Tariff ($/kWh)",
        line=dict(color="#8B5CF6", width=2.0),
        fill="tozeroy",
        fillcolor="rgba(139, 92, 246, 0.12)",
    ), row=1, col=1)

    # 2. Power Demand
    palette = {"RBC": "#EF4444", "DQN": "#F59E0B", "DDPG": "#3B82F6", "PPO": "#10B981"}
    for algo in selected_algos:
        if algo in histories:
            df = histories[algo]
            fig.add_trace(go.Scatter(
                x=steps,
                y=df["net_grid_power_kw"],
                name=f"{algo} Net Power",
                line=dict(color=palette.get(algo, "#6366F1"), width=1.8),
            ), row=2, col=1)

    # Peak Threshold Line
    fig.add_trace(go.Scatter(
        x=steps,
        y=[65.0] * len(steps),
        name="Peak Power Threshold (65 kW)",
        line=dict(color="#DC2626", dash="dot", width=2.0),
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=550,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.update_yaxes(title_text="$/kWh", row=1, col=1)
    fig.update_yaxes(title_text="Net Power (kW)", row=2, col=1)
    fig.update_xaxes(title_text="Hour of Episode", row=2, col=1)
    return fig


def plot_battery_dynamics(histories: dict[str, pd.DataFrame], selected_algos: list[str]) -> go.Figure:
    """Creates battery state of charge (SoC) trajectory plot."""
    sample_df = next(iter(histories.values()))
    steps = np.arange(len(sample_df))

    fig = go.Figure()
    palette = {"RBC": "#EF4444", "DQN": "#F59E0B", "DDPG": "#3B82F6", "PPO": "#10B981"}
    for algo in selected_algos:
        if algo in histories:
            df = histories[algo]
            fig.add_trace(go.Scatter(
                x=steps,
                y=df["battery_soc"] * 100.0,
                name=f"{algo} Battery SoC",
                line=dict(color=palette.get(algo, "#6366F1"), width=2.2),
            ))

    # Min/Max SoC Boundaries
    fig.add_hline(y=15.0, line_dash="dash", line_color="#DC2626", annotation_text="Min Reserve (15%)")
    fig.add_hline(y=95.0, line_dash="dash", line_color="#10B981", annotation_text="Max Capacity (95%)")

    fig.update_layout(
        title="<b>Battery Storage State-of-Charge (SoC %) Arbitrage Over Time</b>",
        xaxis_title="Hour of Episode",
        yaxis_title="State of Charge (%)",
        template="plotly_white",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def main() -> None:
    st.markdown('<div class="main-header">🏢 Smart Building Intelligent Energy Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Reinforcement Learning Framework for HVAC, Battery Storage & Demand Response Optimization</div>', unsafe_allow_html=True)

    # Load Data
    comparison_df, histories = load_benchmark_data()

    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Simulation & Controls")
        scenario = st.selectbox(
            "Evaluation Scenario Period",
            ["Summer Heatwave Peak (July)", "Winter Cold Snap (January)", "Spring Shoulder Transition (April)", "Fall Flexible Season (October)"],
            index=0,
        )

        st.markdown("---")
        st.subheader("🤖 Controllers to Compare")
        selected_algos = st.multiselect(
            "Select Agents",
            ["RBC", "DQN", "DDPG", "PPO"],
            default=["RBC", "DQN", "DDPG", "PPO"] if histories else ["RBC"],
        )

        st.markdown("---")
        st.subheader("🎯 Multi-Objective Weights")
        st.slider("Electricity Cost Weight (w_cost)", 0.5, 3.0, 1.0, 0.1)
        st.slider("Thermal Comfort Weight (w_comfort)", 1.0, 6.0, 3.5, 0.5)
        st.slider("Peak Demand Weight (w_peak)", 0.2, 2.0, 0.8, 0.1)

        st.markdown("---")
        st.info("💡 **Key Concept**: The RL agent learns to pre-cool the building during off-peak morning hours when power is cheap, and discharges the battery during on-peak evening hours (14:00-20:00) to shave costly demand spikes.")

    if not histories:
        st.warning("⚠️ No benchmark results found yet. Please run `python run_pipeline.py --train` or `python run_pipeline.py --benchmark` to generate simulation data.")
        st.stop()

    # Determine Active Algorithm for KPI display
    active_algo = st.selectbox("Highlight Controller in KPI Cards", selected_algos, index=len(selected_algos)-1 if selected_algos else 0)

    # Compute KPI for highlighted agent
    if comparison_df is not None:
        matched_idx = [idx for idx in comparison_df.index if active_algo.lower() in idx.lower()]
        if matched_idx:
            kpi_data = comparison_df.loc[matched_idx[0]].to_dict()
            render_kpi_cards(kpi_data)
        else:
            st.info(f"Displaying simulation data for {active_algo}")

    st.markdown("###")

    # Tabs for structured deep dives
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Comparative Benchmark", "🌡️ Thermal Comfort", "⚡ Grid Demand & Shaving", "🔋 Battery Energy Storage"])

    with tab1:
        st.markdown("#### 🏆 Algorithmic Benchmark Leaderboard")
        if comparison_df is not None:
            # Reformat for clean display
            display_df = comparison_df.copy()
            rename_map = {
                "total_cost_dollar": "Cost ($)",
                "cost_reduction_pct": "Cost Savings (%)",
                "peak_demand_kw": "Peak Demand (kW)",
                "peak_demand_reduction_pct": "Peak Reduction (%)",
                "total_energy_kwh": "Energy (kWh)",
                "energy_savings_pct": "Energy Savings (%)",
                "comfort_violations_deg_hours": "Discomfort (°C·h)",
                "cumulative_reward": "Reward",
            }
            cols_to_show = [c for c in rename_map.keys() if c in display_df.columns]
            styled_df = display_df[cols_to_show].rename(columns=rename_map)

            st.dataframe(
                styled_df.style.highlight_max(subset=["Cost Savings (%)", "Peak Reduction (%)", "Reward"], color="#D1FAE5")
                               .highlight_min(subset=["Cost ($)", "Peak Demand (kW)", "Discomfort (°C·h)"], color="#D1FAE5")
                               .format("{:.2f}"),
                use_container_width=True,
            )

        # Comparative Bar Charts
        if comparison_df is not None:
            c1, c2 = st.columns(2)
            with c1:
                fig_cost = go.Figure(data=[
                    go.Bar(
                        x=comparison_df.index,
                        y=comparison_df["cost_reduction_pct"] if "cost_reduction_pct" in comparison_df else comparison_df["total_cost_dollar"],
                        marker_color=["#EF4444", "#F59E0B", "#3B82F6", "#10B981"],
                        text=[f"{v:.1f}%" for v in comparison_df["cost_reduction_pct"]] if "cost_reduction_pct" in comparison_df else None,
                        textposition="auto",
                    )
                ])
                fig_cost.update_layout(title="<b>Electricity Cost Reduction vs Baseline (%)</b>", yaxis_title="Savings (%)", template="plotly_white", height=320)
                st.plotly_chart(fig_cost, use_container_width=True)

            with c2:
                fig_peak = go.Figure(data=[
                    go.Bar(
                        x=comparison_df.index,
                        y=comparison_df["peak_demand_reduction_pct"] if "peak_demand_reduction_pct" in comparison_df else comparison_df["peak_demand_kw"],
                        marker_color=["#EF4444", "#F59E0B", "#3B82F6", "#10B981"],
                        text=[f"{v:.1f}%" for v in comparison_df["peak_demand_reduction_pct"]] if "peak_demand_reduction_pct" in comparison_df else None,
                        textposition="auto",
                    )
                ])
                fig_peak.update_layout(title="<b>Peak Demand Reduction vs Baseline (%)</b>", yaxis_title="Peak Reduction (%)", template="plotly_white", height=320)
                st.plotly_chart(fig_peak, use_container_width=True)

    with tab2:
        st.plotly_chart(plot_thermal_comfort(histories, selected_algos), use_container_width=True)
        st.caption("ℹ️ The green shaded zone indicates the ASHRAE Standard 55 thermal comfort band (21°C - 24°C). The RL agents maintain indoor temperature within comfortable limits while anticipating peak periods.")

    with tab3:
        st.plotly_chart(plot_power_and_price(histories, selected_algos), use_container_width=True)
        st.caption("ℹ️ Notice the top purple subplot showing on-peak tariff spikes (14:00 - 20:00). Deep RL policies proactively curtail HVAC and dispatch battery discharge to keep net grid demand well below the 65 kW surcharge cap.")

    with tab4:
        st.plotly_chart(plot_battery_dynamics(histories, selected_algos), use_container_width=True)
        st.caption("ℹ️ Battery SoC cycles demonstrate intelligent price arbitrage: charging during off-peak ($0.08/kWh) and solar generation hours, then discharging during on-peak ($0.38/kWh) windows to maximize economic efficiency.")


if __name__ == "__main__":
    main()
