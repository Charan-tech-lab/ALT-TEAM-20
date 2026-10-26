"""Deep Deterministic Policy Gradient (DDPG) Agent for Continuous Smart Building Control.

Implements actor-critic architecture, replay memory, target networks with Polyak averaging,
and exploratory Gaussian noise for off-policy deterministic gradient optimization.
"""

from __future__ import annotations
import random
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .base_agent import BaseAgent


class DDPGActor(nn.Module):
    """Deterministic Actor network mapping state to continuous action vector."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bounded in [-1, 1]
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DDPGCritic(nn.Module):
    """Critic Q-Network mapping (state, action) pair to scalar Q-value."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class DDPGAgent(BaseAgent):
    """Deep Deterministic Policy Gradient continuous control agent."""

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 4,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        noise_std: float = 0.15,
        noise_decay: float = 0.998,
        buffer_capacity: int = 30000,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim, name="DDPG")
        self.gamma = gamma
        self.tau = tau
        self.noise_std = noise_std
        self.noise_decay = noise_decay
        self.batch_size = batch_size
        self.device = torch.device(device)

        # Actor networks
        self.actor = DDPGActor(state_dim, action_dim).to(self.device)
        self.actor_target = DDPGActor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks
        self.critic = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target = DDPGCritic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = deque(maxlen=buffer_capacity)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Selects deterministic action + optional Gaussian exploration noise."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()

        if not evaluate:
            noise = np.random.normal(0, self.noise_std, size=self.action_dim)
            action = action + noise

        return np.clip(action, -1.0, 1.0)

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Stores experience tuple in replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self) -> dict[str, float]:
        """Samples mini-batch and performs actor and critic gradient descent."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 1. Update Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            target_q = self.critic_target(next_states_t, next_actions)
            y = rewards_t + (1.0 - dones_t) * self.gamma * target_q

        current_q = self.critic(states_t, actions_t)
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 2. Update Actor (Deterministic Policy Gradient)
        pred_actions = self.actor(states_t)
        actor_loss = -self.critic(states_t, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # 3. Soft updates of target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Decay exploration noise
        self.noise_std = max(0.02, self.noise_std * self.noise_decay)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "noise_std": float(self.noise_std),
        }

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_optimizer.state_dict(),
            "critic_opt": self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor_target"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic_target"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
