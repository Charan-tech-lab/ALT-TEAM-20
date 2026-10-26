"""Deep Q-Network (DQN) Agent for Discrete Smart Building Control.

Implements experience replay, target networks with Polyak averaging,
and an epsilon-greedy exploration schedule to optimize discrete building actions.
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


class QNetwork(nn.Module):
    """Multi-Layer Perceptron Q-Function approximator."""

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
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNAgent(BaseAgent):
    """Deep Q-Network agent with experience replay and soft target updates."""

    def __init__(
        self,
        state_dim: int = 12,
        action_dim: int = 60,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 20000,
        batch_size: int = 64,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        super().__init__(state_dim=state_dim, action_dim=action_dim, name="DQN")
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.device = torch.device(device)

        # Networks
        self.q_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.replay_buffer = deque(maxlen=buffer_capacity)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> int:
        """Selects action using epsilon-greedy policy during training or argmax during evaluation."""
        if not evaluate and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_t)
            action = int(q_values.argmax(dim=-1).item())
        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Appends experience tuple to replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def update(self) -> dict[str, float]:
        """Samples a mini-batch and updates the Q-Network."""
        if len(self.replay_buffer) < self.batch_size:
            return {}

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states_t = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones_t = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q estimates
        current_q = self.q_net(states_t).gather(1, actions_t)

        # Double DQN / Target Q computation
        with torch.no_grad():
            next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=5.0)
        self.optimizer.step()

        # Polyak soft update of target network
        for param, target_param in zip(self.q_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return {"loss": float(loss.item()), "epsilon": float(self.epsilon)}

    def save(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
        }, filepath)

    def load(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)
