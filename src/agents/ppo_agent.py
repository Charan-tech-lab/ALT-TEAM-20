"""Proximal Policy Optimization (PPO) Agent for Continuous Smart Building Control.

Implements Actor-Critic neural architectures, Generalized Advantage Estimation (GAE),
clipped surrogate loss, and entropy bonus regularization for robust policy updates.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from .base_agent import BaseAgent


class ActorCritic(nn.Module):
    """Joint Actor and Critic network architecture."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
        )

        # Policy Mean (Actor)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

        # Value Function (Critic)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        features = self.shared(state)
        mean = torch.tanh(self.actor_mean(features))
        std = torch.exp(torch.clamp(self.actor_log_std, -2.0, 0.5))
        dist = Normal(mean, std)
        value = self.critic(features)
        return dist, value


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization agent operating on continuous actions."""

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.20,
        c_val: float = 0.50,
        c_entropy: float = 0.01,
        update_epochs: int = 6,
        mini_batch_size: int = 64,
        device: str = "cpu",
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim, name="PPO")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_epsilon
        self.c_val = c_val
        self.c_entropy = c_entropy
        self.update_epochs = update_epochs
        self.mini_batch_size = mini_batch_size
        self.device = torch.device(device)

        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Trajectory buffer
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Selects action: deterministic mean when evaluating, Gaussian sample during exploration."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist, value = self.policy(state_t)
            if evaluate:
                action = dist.mean
            else:
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)
                self.states.append(state)
                self.actions.append(action.squeeze(0).cpu().numpy())
                self.log_probs.append(float(log_prob.item()))
                self.values.append(float(value.item()))

        action_np = action.squeeze(0).cpu().numpy()
        return np.clip(action_np, -1.0, 1.0)

    def store_outcome(self, reward: float, done: bool) -> None:
        """Stores outcome of last step."""
        self.rewards.append(reward)
        self.dones.append(done)

    def update(self) -> dict[str, float]:
        """Runs PPO policy and value optimization over collected trajectory."""
        if len(self.rewards) == 0:
            return {}

        states_t = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs_t = torch.FloatTensor(np.array(self.log_probs)).to(self.device)

        # Compute GAE advantages and discounted returns
        values = np.array(self.values + [0.0])
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values[:-1]
        returns_t = torch.FloatTensor(returns).to(self.device).unsqueeze(1)
        advantages_t = torch.FloatTensor(advantages).to(self.device).unsqueeze(1)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        # Mini-batch gradient updates
        dataset_size = len(self.rewards)
        indices = np.arange(dataset_size)
        total_loss_accum = 0.0

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                batch_idx = indices[start:end]

                b_states = states_t[batch_idx]
                b_actions = actions_t[batch_idx]
                b_old_log_probs = old_log_probs_t[batch_idx]
                b_returns = returns_t[batch_idx]
                b_advantages = advantages_t[batch_idx]

                dist, value = self.policy(b_states)
                new_log_probs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                # Ratio
                ratio = torch.exp(new_log_probs - b_old_log_probs).unsqueeze(1)

                # Clipped surrogate objective
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.MSELoss()(value, b_returns)

                # Total loss
                loss = policy_loss + self.c_val * value_loss - self.c_entropy * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                total_loss_accum += float(loss.item())

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.dones.clear()
        self.values.clear()

        return {"ppo_loss": total_loss_accum / max(1, self.update_epochs * (dataset_size // self.mini_batch_size + 1))}

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
