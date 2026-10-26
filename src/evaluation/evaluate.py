"""Evaluation and Benchmarking Suite for Smart Building RL Framework.

Runs rigorous multi-episode testing comparing Rule-Based Control (RBC) against
trained DQN, DDPG, and PPO agents across standard evaluation metrics and generates
diagnostic publication-ready plots.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ..environment.building_env import SmartBuildingEnv
from ..agents.base_agent import BaseAgent
from ..agents.baseline_rbc import RuleBasedController
from ..agents.dqn_agent import DQNAgent
from ..agents.ppo_agent import PPOAgent
from ..agents.ddpg_agent import DDPGAgent
from .metrics import compute_building_kpis


def evaluate_agent_on_env(
    agent: BaseAgent,
    env: SmartBuildingEnv,
    start_step: int = 4000,
    episode_length: int = 168,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Runs a deterministic evaluation episode and returns KPIs and step-by-step history."""
    obs, info = env.reset(options={"start_step": start_step})
    done = False
    step_count = 0

    while not done and step_count < episode_length:
        action = agent.select_action(obs, evaluate=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

    history = env.history
    return history


def run_benchmark(
    env_continuous: SmartBuildingEnv,
    env_discrete: SmartBuildingEnv,
    trained_agents: dict[str, BaseAgent] | None = None,
    results_dir: str | Path = "results",
    eval_start_step: int = 4320,  # Middle of summer peak (July 1st)
    episode_length: int = 168,     # 1 full week (168 hours)
) -> pd.DataFrame:
    """Executes comparative benchmark across RBC, DQN, DDPG, and PPO."""
    results_path = Path(results_dir)
    plots_path = results_path / "plots"
    metrics_path = results_path / "metrics"
    plots_path.mkdir(parents=True, exist_ok=True)
    metrics_path.mkdir(parents=True, exist_ok=True)

    # 1. Evaluate Rule-Based Controller Baseline
    print("[EVAL] Evaluating Baseline Rule-Based Controller (RBC)...")
    rbc_agent = RuleBasedController()
    rbc_history = evaluate_agent_on_env(rbc_agent, env_continuous, start_step=eval_start_step, episode_length=episode_length)
    rbc_kpis = compute_building_kpis(rbc_history)

    all_kpis: dict[str, dict[str, float]] = {"RBC (Baseline)": rbc_kpis}
    all_histories: dict[str, list[dict[str, Any]]] = {"RBC (Baseline)": rbc_history}

    # 2. Evaluate Trained or Default RL Agents
    agents_to_test = trained_agents or {}

    # Check for DQN
    if "DQN" in agents_to_test:
        print("[EVAL] Evaluating Deep Q-Network (DQN)...")
        dqn_history = evaluate_agent_on_env(agents_to_test["DQN"], env_discrete, start_step=eval_start_step, episode_length=episode_length)
        all_kpis["DQN"] = compute_building_kpis(dqn_history, baseline_kpis=rbc_kpis)
        all_histories["DQN"] = dqn_history

    # Check for DDPG
    if "DDPG" in agents_to_test:
        print("[EVAL] Evaluating Deep Deterministic Policy Gradient (DDPG)...")
        ddpg_history = evaluate_agent_on_env(agents_to_test["DDPG"], env_continuous, start_step=eval_start_step, episode_length=episode_length)
        all_kpis["DDPG"] = compute_building_kpis(ddpg_history, baseline_kpis=rbc_kpis)
        all_histories["DDPG"] = ddpg_history

    # Check for PPO
    if "PPO" in agents_to_test:
        print("[EVAL] Evaluating Proximal Policy Optimization (PPO)...")
        ppo_history = evaluate_agent_on_env(agents_to_test["PPO"], env_continuous, start_step=eval_start_step, episode_length=episode_length)
        all_kpis["PPO"] = compute_building_kpis(ppo_history, baseline_kpis=rbc_kpis)
        all_histories["PPO"] = ppo_history

    # Convert to DataFrame
    comparison_df = pd.DataFrame(all_kpis).T
    comparison_df.index.name = "Algorithm"

    # Save outputs
    comparison_df.to_csv(metrics_path / "benchmark_comparison.csv")
    with open(metrics_path / "benchmark_kpis.json", "w") as f:
        json.dump(all_kpis, f, indent=4)

    # Save detailed episode histories
    for name, hist in all_histories.items():
        clean_name = name.split()[0].lower()
        pd.DataFrame(hist).to_csv(metrics_path / f"history_{clean_name}.csv", index=False)

    # Generate Evaluation Comparison Plots
    generate_benchmark_plots(all_histories, all_kpis, plots_path)

    print("\n[OK] Benchmark Completed Successfully! Results Summary:")
    print(comparison_df[["total_cost_dollar", "cost_reduction_pct", "peak_demand_kw", "peak_demand_reduction_pct", "comfort_violations_deg_hours", "cumulative_reward"]])

    return comparison_df


def generate_benchmark_plots(
    histories: dict[str, list[dict[str, Any]]],
    kpis: dict[str, dict[str, float]],
    save_dir: Path,
) -> None:
    """Generates comparative time-series and bar chart visualizations."""
    sns.set_theme(style="whitegrid")
    time_steps = np.arange(len(next(iter(histories.values()))))

    # --- Plot 1: Indoor Temperature Control & Thermal Comfort Band ---
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    # Target Comfort Envelope [21°C to 24°C]
    ax.axhspan(21.0, 24.0, color="#2ecc71", alpha=0.20, label="ASHRAE 55 Comfort Band (21-24°C)")

    # Plot Outdoor Temperature
    first_hist = pd.DataFrame(next(iter(histories.values())))
    ax.plot(time_steps, first_hist["outdoor_temp"], color="#95a5a6", linestyle="--", linewidth=1.5, label="Outdoor Ambient Temp")

    # Plot each agent's indoor temperature
    palette = {"RBC (Baseline)": "#e74c3c", "DQN": "#f39c12", "DDPG": "#3498db", "PPO": "#27ae60"}
    for name, hist in histories.items():
        df_h = pd.DataFrame(hist)
        color = palette.get(name, "#34495e")
        ax.plot(time_steps, df_h["indoor_temp"], label=f"{name} Indoor Temp", linewidth=2.0, color=color)

    ax.set_title("Thermal Comfort Regulation vs Outdoor Weather Swing (1-Week Test)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Episode", fontsize=11)
    ax.set_ylabel("Temperature (°C)", fontsize=11)
    ax.set_ylim(16.0, 38.0)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    fig.savefig(save_dir / "indoor_temp_comparison.png")
    plt.close(fig)

    # --- Plot 2: Net Grid Power & Peak Demand Shaving ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, dpi=300)

    # Top: Dynamic Electricity Price
    ax1.plot(time_steps, first_hist["electricity_price"], color="#8e44ad", linewidth=2.0, label="TOU Electricity Tariff ($/kWh)")
    ax1.set_ylabel("Price ($/kWh)", fontsize=11)
    ax1.set_title("Dynamic Time-of-Use Electricity Tariffs ($/kWh)", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")

    # Bottom: Net Grid Electrical Demand
    for name, hist in histories.items():
        df_h = pd.DataFrame(hist)
        color = palette.get(name, "#34495e")
        ax2.plot(time_steps, df_h["net_grid_power_kw"], label=name, linewidth=1.8, color=color)

    ax2.axhline(65.0, color="#c0392b", linestyle=":", linewidth=2.0, label="Peak Demand Limit (65 kW)")
    ax2.set_title("Net Grid Demand (kW) & Peak Shaving Performance", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Hour of Episode", fontsize=11)
    ax2.set_ylabel("Net Power (kW)", fontsize=11)
    ax2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_dir / "net_power_comparison.png")
    plt.close(fig)

    # --- Plot 3: Battery Energy Storage State of Charge (SoC) Arbitrage ---
    fig, ax = plt.subplots(figsize=(13, 5), dpi=300)
    for name, hist in histories.items():
        df_h = pd.DataFrame(hist)
        color = palette.get(name, "#34495e")
        ax.plot(time_steps, df_h["battery_soc"] * 100.0, label=f"{name} SoC", linewidth=2.0, color=color)

    ax.axhline(15.0, color="red", linestyle="--", alpha=0.5, label="Min SoC Limit (15%)")
    ax.axhline(95.0, color="green", linestyle="--", alpha=0.5, label="Max SoC Limit (95%)")
    ax.set_title("Battery State-of-Charge (SoC %) Arbitrage Over Time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Episode", fontsize=11)
    ax.set_ylabel("Battery SoC (%)", fontsize=11)
    ax.set_ylim(10.0, 100.0)
    ax.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(save_dir / "battery_soc_comparison.png")
    plt.close(fig)

    # --- Plot 4: Key Evaluation KPI Bar Comparisons ---
    df_kpi = pd.DataFrame(kpis).T
    fig, ((bx1, bx2), (bx3, bx4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    # Cost
    sns.barplot(x=df_kpi.index, y=df_kpi["total_cost_dollar"], ax=bx1, palette=["#e74c3c", "#f39c12", "#3498db", "#27ae60"])
    bx1.set_title("Total Electricity Cost ($)", fontweight="bold")
    bx1.set_ylabel("Dollars ($)")

    # Peak Demand
    sns.barplot(x=df_kpi.index, y=df_kpi["peak_demand_kw"], ax=bx2, palette=["#e74c3c", "#f39c12", "#3498db", "#27ae60"])
    bx2.set_title("Peak Grid Demand (kW)", fontweight="bold")
    bx2.set_ylabel("Peak Power (kW)")

    # Comfort Violations
    sns.barplot(x=df_kpi.index, y=df_kpi["comfort_violations_deg_hours"], ax=bx3, palette=["#e74c3c", "#f39c12", "#3498db", "#27ae60"])
    bx3.set_title("Thermal Comfort Violations (°C-Hours)", fontweight="bold")
    bx3.set_ylabel("Degree-Hours Outside Comfort Band")

    # Cumulative Reward
    sns.barplot(x=df_kpi.index, y=df_kpi["cumulative_reward"], ax=bx4, palette=["#e74c3c", "#f39c12", "#3498db", "#27ae60"])
    bx4.set_title("Cumulative Objective Reward", fontweight="bold")
    bx4.set_ylabel("Reward (Higher is better)")

    plt.tight_layout()
    fig.savefig(save_dir / "kpi_benchmark_bars.png")
    plt.close(fig)
