import pygame
import random
from typing import Tuple, List
import numpy as np

class SnakeGame:
    """
    Snake game environment for AI training
    This class handles all game logic including snake movement, food generation, and collision detection
    """
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.cell_size = 20  # Size of each cell in pixels
        self.window_size = grid_size * self.cell_size
        
        # Initialize snake at center
        center = grid_size // 2
        self.snake = [(center, center)]
        self.direction = (1, 0)  # Right
        self.food = None
        self.score = 0
        self.is_game_over = False
        self.snake_length = 1
        self.death_reason = None  # 记录死因
        
        # Generate first food
        self._spawn_food()
    
    def _spawn_food(self):
        """Generate food at a random position that's not on the snake"""
        # 防止无限循环：如果蛇占满了整个地图，就不生成食物
        if len(self.snake) >= self.grid_size * self.grid_size - 1:
            self.food = None
            return
        
        max_attempts = 1000  # 防止死循环
        attempts = 0
        while attempts < max_attempts:
            self.food = (random.randint(0, self.grid_size - 1), 
                        random.randint(0, self.grid_size - 1))
            if self.food not in self.snake:
                break
            attempts += 1
        
        # 如果1000次都找不到空位，说明地图太满，随机选一个
        if attempts >= max_attempts:
            empty_positions = []
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if (x, y) not in self.snake:
                        empty_positions.append((x, y))
            if empty_positions:
                self.food = random.choice(empty_positions)
            else:
                self.food = None  # 没有空位，游戏胜利？
    
    def get_state(self) -> np.ndarray:
        """
        Convert game state to a numpy array for AI to understand
        Returns: Enhanced state with spatial and directional information
        """
        state = np.zeros((4, self.grid_size, self.grid_size))
        
        # Layer 0: Walls (boundaries)
        state[0, :, 0] = 1  # Left wall
        state[0, :, -1] = 1  # Right wall
        state[0, 0, :] = 1  # Top wall
        state[0, -1, :] = 1  # Bottom wall
        
        # Layer 1: Snake body (all segments except head)
        for segment in self.snake[:-1]:
            state[1, segment[1], segment[0]] = 1
        
        # Layer 2: Snake head
        if self.snake:
            head_x, head_y = self.snake[-1]
            state[2, head_y, head_x] = 1
        
        # Layer 3: Food
        if self.food:
            state[3, self.food[1], self.food[0]] = 1
        
        # Flatten spatial state
        spatial_state = state.flatten()
        
        # Add extra features for better decision making
        extra_features = []
        
        if self.snake and self.food:
            head_x, head_y = self.snake[-1]
            
            # 1. Normalized distance to food (Manhattan distance)
            dist_x = (self.food[0] - head_x) / self.grid_size
            dist_y = (self.food[1] - head_y) / self.grid_size
            extra_features.extend([dist_x, dist_y])
            
            # 2. Snake length (normalized)
            extra_features.append(len(self.snake) / (self.grid_size * self.grid_size))
            
            # 3. Danger in 4 directions (1 = danger, 0 = safe)
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
            for dx, dy in directions:
                next_x, next_y = head_x + dx, head_y + dy
                is_danger = (next_x < 0 or next_x >= self.grid_size or 
                           next_y < 0 or next_y >= self.grid_size or 
                           (next_x, next_y) in self.snake)
                extra_features.append(1.0 if is_danger else 0.0)
            
            # 4. Safe space count (normalized)
            safe_count = self._count_safe_space(head_x, head_y)
            extra_features.append(safe_count / 4.0)
        else:
            # Default values if no snake or food
            extra_features = [0.0] * 10
        
        # Combine spatial state with extra features
        return np.concatenate([spatial_state, extra_features])
    
    def move(self, direction: int) -> Tuple[int, bool, bool]:
        """
        Move snake based on action
        Args:
            direction: 0=up, 1=right, 2=down, 3=left
        Returns:
            (reward, done, ate_food): reward value, game over flag, ate food flag
        """
        if self.is_game_over:
            return (-10, True, False)
        
        # Convert action to direction vector
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # up, right, down, left
        new_direction = directions[direction]
        
        # Get current head position
        head_x, head_y = self.snake[-1]
        
        # Calculate distance to food BEFORE moving
        old_dist_to_food = 0
        if self.food:
            old_dist_to_food = abs(head_x - self.food[0]) + abs(head_y - self.food[1])
        
        # Calculate new head position
        new_x = head_x + new_direction[0]
        new_y = head_y + new_direction[1]
        
        # Check collision with walls
        if new_x < 0 or new_x >= self.grid_size or new_y < 0 or new_y >= self.grid_size:
            self.is_game_over = True
            self.death_reason = "wall"  # 记录死因
            return (-10, True, False)
        
        # Check collision with self
        if (new_x, new_y) in self.snake:
            self.is_game_over = True
            self.death_reason = "self"  # 记录死因
            return (-10, True, False)
        
        # Move snake
        self.snake.append((new_x, new_y))
        
        # Check if food is eaten
        ate_food = False
        if self.food and (new_x, new_y) == self.food:
            self.score += 1
            self.snake_length += 1
            ate_food = True
            # 连续吃食物的递增奖励：鼓励吃多个
            # 第1个：15分，第2个：19分，第3个：24分...（加速增长）
            # 使用平方增长鼓励长蛇：bonus = 4 * (n-1) + (n-1)^2
            n = self.snake_length - 1  # 食物数量
            consecutive_bonus = 4 * (n - 1) + (n - 1) ** 2
            base_reward = 15 + consecutive_bonus
            self._spawn_food()
            return (base_reward, False, True)  # 连续吃食物获得递增奖励
        else:
            # Remove tail
            self.snake.pop(0)
        
        # Calculate IMPROVED reward shaping with balanced safety awareness
        reward = 0.0
        
        # 1. Distance to food reward (encourage moving closer to food)
        # 用途：引导AI朝食物方向移动，而不是漫无目的游走
        if self.food:
            new_dist_to_food = abs(new_x - self.food[0]) + abs(new_y - self.food[1])
            # 如果距离变近，给正奖励；变远，给负奖励
            if new_dist_to_food < old_dist_to_food:
                reward += 3.0  # 靠近食物：+3
            elif new_dist_to_food == old_dist_to_food:
                reward -= 0.5  # 既不靠近也不远离：轻微惩罚（防止原地打转）
            else:
                reward -= 1.0  # 远离食物：-1（从-2降低到-1，避免吃完后无所适从）
        
        # 2. Safety/Space reward (防止吃豆后撞自己，但对长蛇更宽容)
        # 用途：只惩罚真正危险的情况，鼓励继续追食物
        safe_space = self._count_safe_space(new_x, new_y)
        if safe_space == 0:
            reward -= 10.0  # 死路！严重惩罚
        elif safe_space == 1:
            # 只有一条路：只有在蛇很长时才轻微惩罚，且惩罚逐渐降低
            if len(self.snake) > 8:  # 从>5改为>8，更宽容
                reward -= 0.3  # 从-0.5降低到-0.3
            elif len(self.snake) > 5:
                reward -= 0.1  # 中等长度时只轻微惩罚
        # safe_space >= 2: 不惩罚，鼓励AI继续追食物
        
        # 3. Danger awareness (只在极度危险时才惩罚)
        # 用途：检测周围完全被包围的情况
        danger_level = self._check_danger_ahead(new_x, new_y)
        if danger_level == 4:  # 只有四面都危险时才惩罚（从>=3改为==4）
            reward -= 5.0  # 四面楚歌，严重惩罚
        
        # 4. Survival penalty (encourage efficiency, don't waste time)
        # 用途：防止AI原地打转或无限游走
        # 蛇越长，惩罚反而越轻（避免长蛇过于急躁）
        if len(self.snake) <= 5:
            survival_penalty = 0.01
        else:
            # 长蛇时降低存活惩罚，给更多时间规划
            survival_penalty = 0.01 / (1 + (len(self.snake) - 5) * 0.1)
        reward -= survival_penalty
        
        # 5. Wall proximity penalty (只在非常接近墙时才惩罚)
        # 用途：避免贴墙走，但不阻止正常移动
        dist_to_walls = min(new_x, self.grid_size - 1 - new_x, 
                           new_y, self.grid_size - 1 - new_y)
        if dist_to_walls == 0:
            reward -= 3.0  # 紧贴墙：-3（这不应该发生，因为会撞墙）
        elif dist_to_walls == 1:
            reward -= 0.5  # 距离墙1格：-0.5（轻微惩罚）
        # dist_to_walls >= 2: 不惩罚
        
        return (reward, False, False)
    
    def _count_safe_space(self, x: int, y: int) -> int:
        """
        计算当前位置周围有多少安全方向可以走
        用途：避免AI把自己困在死路
        """
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 上右下左
        safe_count = 0
        
        for dx, dy in directions:
            next_x, next_y = x + dx, y + dy
            # 检查是否安全：不撞墙，不撞蛇身
            if (0 <= next_x < self.grid_size and 
                0 <= next_y < self.grid_size and 
                (next_x, next_y) not in self.snake):
                safe_count += 1
        
        return safe_count
    
    def _check_danger_ahead(self, x: int, y: int) -> int:
        """
        检查当前位置周围有多少危险方向（会撞到东西）
        返回危险方向的数量
        """
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # 上右下左
        danger_count = 0
        
        for dx, dy in directions:
            next_x, next_y = x + dx, y + dy
            # 检查是否危险：撞墙或撞蛇身
            if (next_x < 0 or next_x >= self.grid_size or 
                next_y < 0 or next_y >= self.grid_size or 
                (next_x, next_y) in self.snake):
                danger_count += 1
        
        return danger_count
    
    def reset(self):
        """Reset game to initial state"""
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = (1, 0)
        self.score = 0
        self.is_game_over = False
        self.snake_length = 1
        self.death_reason = None  # 记录死因
        self._spawn_food()
    
    def render(self):
        """
        Render game using pygame (for visualization during training/testing)
        Note: This will be used when playing/testing, not during actual training
        """
        screen = pygame.display.set_mode((self.window_size, self.window_size))
        screen.fill((0, 0, 0))
        
        # Draw snake
        for i, segment in enumerate(self.snake):
            x = segment[0] * self.cell_size
            y = segment[1] * self.cell_size
            color = (0, 255, 0) if i == len(self.snake) - 1 else (0, 200, 0)  # Head slightly brighter
            pygame.draw.rect(screen, color, 
                           (x, y, self.cell_size - 1, self.cell_size - 1))
        
        # Draw food
        if self.food:
            fx = self.food[0] * self.cell_size
            fy = self.food[1] * self.cell_size
            pygame.draw.rect(screen, (255, 0, 0), 
                           (fx, fy, self.cell_size - 1, self.cell_size - 1))
        
        pygame.display.flip()

