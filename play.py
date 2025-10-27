import pygame
import numpy as np
from game import SnakeGame
from ai_agent import DQNAgent
import sys
import argparse

def play_with_ai(model_path="models/snake_model_final.pth", fps=10, show_grid=True):
    """
    Test the trained AI by watching it play Snake
    
    This will load the trained model and let it play the game while you watch
    """
    # Initialize pygame with proper font
    pygame.init()
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    
    # Initialize game and agent
    game = SnakeGame(grid_size=20)
    # Get actual state size from game
    sample_state = game.get_state()
    state_size = len(sample_state)
    action_size = 4
    agent = DQNAgent(state_size, action_size, use_gpu=False)  # GPU not needed for playing
    
    # Track stats
    best_score = 0
    total_games = 0
    
    # Load trained model
    try:
        agent.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first using train.py")
        sys.exit(1)
    
    # Set epsilon to 0 (no random actions, use best predictions)
    agent.epsilon = 0
    
    clock = pygame.time.Clock()
    running = True
    
    print("AI is playing Snake!")
    print("Controls:")
    print("  ESC/Q - Quit")
    print("  SPACE - Pause/Resume")
    print("  +/-   - Adjust Speed")
    print("  R     - Force Restart Game")
    print(f"Current FPS: {fps} (adjust with +/-)")
    
    paused = False
    
    # Get screen size
    screen_size = game.window_size
    info_width = 200
    total_width = screen_size + info_width
    
    # Create screen with info panel
    screen = pygame.display.set_mode((total_width, screen_size))
    pygame.display.set_caption(f"Snake AI - Model: {model_path}")
    
    # Loop detection (检测原地打转)
    position_history = []  # 记录最近的位置
    action_history = []    # 记录最近的动作
    loop_detected = False
    stuck_counter = 0
    max_stuck_steps = 100  # 100步内没进展 = 卡住
    last_score = 0
    
    # Play until game over
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:  # R key - force restart
                    game.reset()
                    position_history.clear()
                    action_history.clear()
                    loop_detected = False
                    stuck_counter = 0
                    last_score = 0
                    print("Game restarted manually")
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:  # + key
                    fps = min(fps + 5, 60)
                    print(f"Speed increased to {fps} FPS")
                elif event.key == pygame.K_MINUS:  # - key
                    fps = max(fps - 5, 1)
                    print(f"Speed decreased to {fps} FPS")
        
        if not paused and not game.is_game_over:
            # Get current state
            state = game.get_state()
            
            # AI chooses action
            action = agent.act(state)
            
            # Debug: 显示AI的Q值预测（可选）
            if game.snake_length <= 3:  # 只在蛇短时显示
                import torch
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    q_values = agent.q_network(state_tensor)
                action_names = ['UP', 'RIGHT', 'DOWN', 'LEFT']
                if game.snake_length == 2:  # 刚吃完第一个
                    print(f"Q-values after eating 1st food: {action_names[action]}={q_values[0][action].item():.2f}")
            
            # Execute action
            reward, done, ate_food = game.move(action)
            
            # Loop detection - 检测是否原地打转
            current_head = game.snake[-1] if game.snake else (0, 0)
            position_history.append(current_head)
            action_history.append(action)
            
            # 只保留最近50步的历史
            if len(position_history) > 50:
                position_history.pop(0)
                action_history.pop(0)
            
            # 检测循环：如果最近20步内位置重复超过10次
            if len(position_history) >= 20:
                recent_positions = position_history[-20:]
                unique_positions = len(set(recent_positions))
                if unique_positions <= 5:  # 20步内只访问了5个位置或更少
                    loop_detected = True
                else:
                    loop_detected = False
            
            # 检测卡住：长时间没有进展
            current_score = game.snake_length - 1
            if current_score == last_score:
                stuck_counter += 1
            else:
                stuck_counter = 0
                last_score = current_score
            
            # 如果卡住超过100步，自动重启
            if stuck_counter >= max_stuck_steps:
                print(f"AI stuck for {max_stuck_steps} steps! Auto-restarting...")
                game.reset()
                position_history.clear()
                action_history.clear()
                loop_detected = False
                stuck_counter = 0
                last_score = 0
                total_games += 1
        
        # Render game
        screen.fill((20, 20, 20))  # Dark background
        
        # Draw game area
        game_surface = pygame.Surface((screen_size, screen_size))
        game_surface.fill((0, 0, 0))
        
        # Draw grid if enabled
        if show_grid:
            for i in range(game.grid_size + 1):
                x = i * (screen_size // game.grid_size)
                pygame.draw.line(game_surface, (40, 40, 40), (x, 0), (x, screen_size))
                pygame.draw.line(game_surface, (40, 40, 40), (0, x), (screen_size, x))
        
        # Draw snake
        for i, segment in enumerate(game.snake):
            x = segment[0] * (screen_size // game.grid_size)
            y = segment[1] * (screen_size // game.grid_size)
            size = screen_size // game.grid_size
            color = (0, 255, 0) if i == len(game.snake) - 1 else (0, 200, 0)
            pygame.draw.rect(game_surface, color, (x + 1, y + 1, size - 2, size - 2))
        
        # Draw food
        if game.food:
            fx = game.food[0] * (screen_size // game.grid_size)
            fy = game.food[1] * (screen_size // game.grid_size)
            size = screen_size // game.grid_size
            pygame.draw.rect(game_surface, (255, 0, 0), (fx + 1, fy + 1, size - 2, size - 2))
        
        # Blit game surface to screen
        screen.blit(game_surface, (0, 0))
        
        # Draw info panel
        info_x = screen_size + 10
        y_pos = 20
        
        # Score info
        score_text = font.render(f"Score: {game.snake_length - 1}", True, (255, 255, 255))
        screen.blit(score_text, (info_x, y_pos))
        y_pos += 30
        
        best_score = max(best_score, game.snake_length - 1)
        best_text = font.render(f"Best: {best_score}", True, (255, 215, 0))
        screen.blit(best_text, (info_x, y_pos))
        y_pos += 30
        
        length_text = font.render(f"Length: {game.snake_length}", True, (200, 200, 200))
        screen.blit(length_text, (info_x, y_pos))
        y_pos += 50
        
        # Stats
        total_games += 1 if game.is_game_over else 0
        games_text = font.render(f"Games: {total_games}", True, (150, 150, 150))
        screen.blit(games_text, (info_x, y_pos))
        y_pos += 30
        
        fps_text = font.render(f"FPS: {fps}", True, (150, 150, 150))
        screen.blit(fps_text, (info_x, y_pos))
        y_pos += 50
        
        # Loop/Stuck detection display
        if loop_detected:
            loop_text = font.render("LOOPING!", True, (255, 100, 0))
            screen.blit(loop_text, (info_x, y_pos))
            y_pos += 25
        
        if stuck_counter > 20:  # 显示卡住警告
            stuck_pct = min(100, int(stuck_counter / max_stuck_steps * 100))
            stuck_text = small_font.render(f"Stuck: {stuck_pct}%", True, (255, 150, 0))
            screen.blit(stuck_text, (info_x, y_pos))
            y_pos += 25
        
        # Controls
        small_font = pygame.font.Font(None, 18)
        controls = [
            "ESC/Q - Quit",
            "SPACE - Pause",
            "+/- - Speed",
            "R - Restart"
        ]
        for ctrl in controls:
            ctrl_text = small_font.render(ctrl, True, (100, 100, 100))
            screen.blit(ctrl_text, (info_x, y_pos))
            y_pos += 20
        
        # Pause indicator
        if paused:
            pause_text = font.render("PAUSED", True, (255, 255, 0))
            text_rect = pause_text.get_rect(center=(total_width // 2, 30))
            screen.blit(pause_text, text_rect)
        
        # Loop warning overlay
        if loop_detected:
            warning_text = font.render("AI IS LOOPING!", True, (255, 100, 0))
            text_rect = warning_text.get_rect(center=(total_width // 2, screen_size - 30))
            # Draw background box
            padding = 10
            box_rect = pygame.Rect(
                text_rect.left - padding,
                text_rect.top - padding,
                text_rect.width + padding * 2,
                text_rect.height + padding * 2
            )
            pygame.draw.rect(screen, (50, 20, 0), box_rect)
            pygame.draw.rect(screen, (255, 100, 0), box_rect, 2)
            screen.blit(warning_text, text_rect)
        
        pygame.display.flip()
        clock.tick(fps)
        
        # Handle game over
        if game.is_game_over:
            # 显示游戏结束原因
            final_score = game.snake_length - 1
            death_reason = getattr(game, 'death_reason', 'unknown')
            reason_text = {
                'wall': '撞墙',
                'self': '撞到自己',
                'unknown': '未知原因'
            }.get(death_reason, '未知原因')
            
            print(f"Game Over! Score: {final_score}, Reason: {reason_text}")
            if final_score > 0:
                print(f"  Snake ate {final_score} food(s) before dying")
            
            pygame.time.wait(2000)
            game.reset()
            position_history.clear()
            action_history.clear()
            loop_detected = False
            stuck_counter = 0
            last_score = 0
            total_games += 1
    
    pygame.quit()
    print(f"\nFinal stats:")
    print(f"  Best score: {best_score}")
    print(f"  Total games: {total_games}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Watch AI play Snake')
    parser.add_argument('model', nargs='?', default='models/snake_model_final.pth',
                       help='Path to the trained model')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second (default: 10)')
    parser.add_argument('--no-grid', action='store_true',
                       help='Disable grid lines')
    
    args = parser.parse_args()
    
    play_with_ai(args.model, fps=args.fps, show_grid=not args.no_grid)

