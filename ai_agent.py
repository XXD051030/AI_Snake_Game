import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class DQNAgent:
    """
    Deep Q-Network (DQN) agent for playing Snake
    
    This agent uses a neural network to learn the best actions.
    Key features:
    - Experience Replay: Stores past experiences to learn from them
    - Target Network: Uses a separate network for stable learning
    - Epsilon-Greedy: Gradually reduces random actions
    """
    
    def __init__(self, 
                 state_size: int, 
                 action_size: int, 
                 lr: float = 0.0003, 
                 use_gpu: bool = True,
                 epsilon_start: float = 1.0,
                 epsilon_min: float = 0.05,
                 epsilon_decay: float = 0.9995,
                 replay_size: int = 100000,
                 gamma: float = 0.95):
        """
        Initialize the DQN agent
        
        Args:
            state_size: Size of the state vector (game grid flattened)
            action_size: Number of possible actions (4 for snake)
            lr: Learning rate for optimizer (降低到0.0003，更深的网络需要更小的学习率)
            use_gpu: Whether to use GPU if available
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
        print(f"Using device: {self.device}")
        
        # Create main network and target network (for stable learning)
        # 使用更大的网络（256神经元）提高学习能力
        self.q_network = NeuralNetwork(state_size, action_size).to(self.device)
        self.target_network = NeuralNetwork(state_size, action_size).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer to update weights
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Experience replay buffer (stores past game experiences)
        # 保持可配置大小，提供足够多样的经验
        self.memory = deque(maxlen=replay_size)
        
        # Epsilon (exploration rate)
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Discount factor
        self.gamma = gamma
        
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory for later learning
        
        This is called after each step in the game
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """
        Choose an action based on current state
        
        Uses epsilon-greedy strategy:
        - With probability epsilon: choose random action (exploration)
        - Otherwise: choose best action according to Q-network (exploitation)
        """
        if random.random() <= self.epsilon:
            # Explore: choose random action
            return random.randrange(self.action_size)
        
        # Exploit: choose best action predicted by neural network
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():  # No need to compute gradients for prediction
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def replay(self, batch_size=128):
        """
        Train the neural network using random samples from memory
        
        This is the core of DQN learning - it learns from past experiences
        Batch size increased to 128 for more stable gradient updates
        """
        if len(self.memory) < batch_size:
            return
        
        # Sample random batch of experiences
        batch = random.sample(self.memory, batch_size)
        
        # Extract components (convert to numpy arrays first to avoid warnings)
        states = torch.FloatTensor(np.array([e[0] for e in batch])).to(self.device)
        actions = torch.LongTensor(np.array([e[1] for e in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([e[2] for e in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([e[3] for e in batch])).to(self.device)
        dones = torch.BoolTensor(np.array([e[4] for e in batch])).to(self.device)
        
        # Current Q values (what we predict)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: Use main network to SELECT action, target network to EVALUATE it
        # 这减少了Q值过估计的问题
        with torch.no_grad():
            # Use main network to select best actions
            next_actions = self.q_network(next_states).argmax(1)
            # Use target network to evaluate those actions
            next_q = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            # Use configurable gamma for discounted target
            target_q = rewards + (self.gamma * next_q * ~dones)
        
        # Loss function: difference between current and target Q values
        loss = F.mse_loss(current_q.squeeze(), target_q)
        
        # Backpropagation to update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon (reduce exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save(self, filepath):
        """Save the trained model"""
        torch.save(self.q_network.state_dict(), filepath)
    
    def load(self, filepath):
        """Load a trained model"""
        self.q_network.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())


class NeuralNetwork(nn.Module):
    """
    Neural network architecture for DQN
    
    Deeper fully connected network:
    Input -> 512 neurons -> 256 neurons -> 128 neurons -> Output (4 actions)
    更深的网络可以学习更复杂的策略和特征层次
    """
    
    def __init__(self, input_size, output_size):
        super(NeuralNetwork, self).__init__()
        # 更深更宽的网络架构
        self.fc1 = nn.Linear(input_size, 512)  # First hidden layer (更宽)
        self.fc2 = nn.Linear(512, 256)  # Second hidden layer
        self.fc3 = nn.Linear(256, 128)  # Third hidden layer (新增)
        self.fc4 = nn.Linear(128, output_size)  # Output layer
        
        # Dropout for regularization (防止过拟合)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        """Forward pass through the network with dropout"""
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Dropout after first layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Dropout after second layer
        x = F.relu(self.fc3(x))
        x = self.fc4(x)  # No activation on output (Q-values can be any real number)
        return x

