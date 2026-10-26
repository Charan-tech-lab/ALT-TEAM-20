"""End-to-End Execution Pipeline for Smart Building RL Framework.

Provides CLI commands for dataset generation, training DQN / DDPG / PPO agents,
benchmarking against conventional Rule-Based Control (RBC), and launching the Streamlit dashboard.
"""

from __future__ import annotations
import argparse
import sys
import subprocess
from pathlib import Path
import numpy as np

# Ensure UTF-8 output encoding across Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Add repo root to sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.preprocessing.data_loader import load_or_generate_building_data
from src.environment.building_env import SmartBuildingEnv
from src.agents.baseline_rbc import RuleBasedController
from src.agents.dqn_agent import DQNAgent
from src.agents.ddpg_agent import DDPGAgent
from src.agents.ppo_agent import PPOAgent
from src.evaluation.evaluate import run_benchmark


def train_dqn(env: SmartBuildingEnv, episodes: int = 10, save_path: str | Path | None = None) -> DQNAgent:
    """Trains a Deep Q-Network agent on discrete building control."""
    save_path = Path(save_path or (REPO_ROOT / "results" / "models" / "dqn_model.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[TRAIN] Training Deep Q-Network (DQN) for {episodes} episodes...")
    agent = DQNAgent(state_dim=12, action_dim=env.action_space.n)

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()

            obs = next_obs
            total_reward += reward
            step_count += 1

        print(f"  [DQN] Episode {ep:2d}/{episodes} | Steps: {step_count} | Reward: {total_reward:8.1f} | Epsilon: {agent.epsilon:.3f}")

    agent.save(str(save_path))
    print(f"  [SAVED] Saved DQN model to {save_path}")
    return agent


def train_ddpg(env: SmartBuildingEnv, episodes: int = 10, save_path: str | Path | None = None) -> DDPGAgent:
    """Trains a Deep Deterministic Policy Gradient agent on continuous building control."""
    save_path = Path(save_path or (REPO_ROOT / "results" / "models" / "ddpg_model.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[TRAIN] Training DDPG for {episodes} episodes...")
    agent = DDPGAgent(state_dim=12, action_dim=4)

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, next_obs, done)
            agent.update()

            obs = next_obs
            total_reward += reward
            step_count += 1

        print(f"  [DDPG] Episode {ep:2d}/{episodes} | Steps: {step_count} | Reward: {total_reward:8.1f} | Noise Std: {agent.noise_std:.3f}")

    agent.save(str(save_path))
    print(f"  [SAVED] Saved DDPG model to {save_path}")
    return agent


def train_ppo(env: SmartBuildingEnv, episodes: int = 10, save_path: str | Path | None = None) -> PPOAgent:
    """Trains a Proximal Policy Optimization agent on continuous building control."""
    save_path = Path(save_path or (REPO_ROOT / "results" / "models" / "ppo_model.pt"))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[TRAIN] Training PPO for {episodes} episodes...")
    agent = PPOAgent(state_dim=12, action_dim=4)

    for ep in range(1, episodes + 1):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        step_count = 0

        while not done:
            action = agent.select_action(obs, evaluate=False)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_outcome(reward, done)
            obs = next_obs
            total_reward += reward
            step_count += 1

        loss_info = agent.update()
        loss_val = loss_info.get("ppo_loss", 0.0)
        print(f"  [PPO] Episode {ep:2d}/{episodes} | Steps: {step_count} | Reward: {total_reward:8.1f} | Loss: {loss_val:.4f}")

    agent.save(str(save_path))
    print(f"  [SAVED] Saved PPO model to {save_path}")
    return agent


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Building RL Pipeline Runner")
    parser.add_argument("--generate-data", action="store_true", help="Generate annual building energy genome dataset")
    parser.add_argument("--train", action="store_true", help="Train DQN, DDPG, and PPO agents")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark evaluation comparing RBC and RL agents")
    parser.add_argument("--dashboard", action="store_true", help="Launch interactive Streamlit dashboard")
    parser.add_argument("--all", action="store_true", help="Run entire pipeline end-to-end")
    parser.add_argument("--episodes", type=int, default=10, help="Training episodes per agent (default: 10)")
    args = parser.parse_args()

    # Default to --all if no specific flags given
    if not any([args.generate_data, args.train, args.benchmark, args.dashboard, args.all]):
        args.all = True

    # 1. Dataset Generation
    if args.generate_data or args.all:
        print("\n" + "="*70)
        print("[STEP 1] Generating Building Energy Genome Dataset (8760 Hours)")
        print("="*70)
        df = load_or_generate_building_data(force_regenerate=True)
        print(f"[OK] Generated dataset with {len(df)} hourly samples.")
        print(df[["outdoor_temp", "solar_irradiance", "occupancy", "base_load_kw", "electricity_price", "solar_pv_kw"]].describe())

    # Load dataset for training and testing
    df = load_or_generate_building_data()

    # Create Environments
    env_cont = SmartBuildingEnv(df=df, discrete_actions=False, episode_length_hours=168)
    env_disc = SmartBuildingEnv(df=df, discrete_actions=True, episode_length_hours=168)

    models_dir = REPO_ROOT / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    dqn_path = models_dir / "dqn_model.pt"
    ddpg_path = models_dir / "ddpg_model.pt"
    ppo_path = models_dir / "ppo_model.pt"

    trained_agents = {}

    # 2. Agent Training
    if args.train or args.all:
        print("\n" + "="*70)
        print(f"[STEP 2] Training RL Agents ({args.episodes} Episodes Each)")
        print("="*70)

        # Train DQN
        dqn_agent = train_dqn(env_disc, episodes=args.episodes, save_path=dqn_path)
        trained_agents["DQN"] = dqn_agent

        # Train DDPG
        ddpg_agent = train_ddpg(env_cont, episodes=args.episodes, save_path=ddpg_path)
        trained_agents["DDPG"] = ddpg_agent

        # Train PPO
        ppo_agent = train_ppo(env_cont, episodes=args.episodes, save_path=ppo_path)
        trained_agents["PPO"] = ppo_agent
    else:
        # Load pre-trained models if they exist
        if dqn_path.exists():
            dqn_agent = DQNAgent(state_dim=12, action_dim=env_disc.action_space.n)
            dqn_agent.load(str(dqn_path))
            trained_agents["DQN"] = dqn_agent
        if ddpg_path.exists():
            ddpg_agent = DDPGAgent(state_dim=12, action_dim=4)
            ddpg_agent.load(str(ddpg_path))
            trained_agents["DDPG"] = ddpg_agent
        if ppo_path.exists():
            ppo_agent = PPOAgent(state_dim=12, action_dim=4)
            ppo_agent.load(str(ppo_path))
            trained_agents["PPO"] = ppo_agent

    # 3. Benchmark Evaluation
    if args.benchmark or args.all:
        print("\n" + "="*70)
        print("[STEP 3] Executing Comprehensive Benchmark Comparison (Summer Peak Week)")
        print("="*70)
        comparison_df = run_benchmark(
            env_continuous=env_cont,
            env_discrete=env_disc,
            trained_agents=trained_agents,
            results_dir=REPO_ROOT / "results",
            eval_start_step=4320,  # Middle of summer (hour 4320 = July 1st)
            episode_length=168,    # 1 full week
        )

    # 4. Streamlit Dashboard
    if args.dashboard:
        print("\n" + "="*70)
        print("[STEP 4] Launching Streamlit Energy Management Dashboard")
        print("="*70)
        dashboard_app = REPO_ROOT / "src" / "dashboard" / "app.py"
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_app)])


if __name__ == "__main__":
    main()
