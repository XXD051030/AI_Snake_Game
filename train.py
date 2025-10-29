import pygame
import numpy as np
from game import SnakeGame
from ai_agent import DQNAgent
import matplotlib.pyplot as plt
from collections import deque
import os
from tqdm import tqdm
import time
import json
from datetime import datetime
import keyboard
import signal
import sys
import argparse
import torch
import random
 
import platform
import subprocess
try:
    import psutil  # type: ignore
except Exception:  # psutil optional; degrade gracefully
    psutil = None


 

def train_snake_ai(
    episodes=100000,
    visualize=False,
    mode: str = "new",
    model_path: str | None = None,
    reset_epsilon: float | None = None,
    batch_size: int = 128,
    lr: float = 0.0003,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.9995,
    target_update: int = 50,
    replay_size: int = 100000,
    checkpoint_interval: int = 1000,
    log_interval: int = 100,
    device: str = "auto",  # auto|cuda|cpu
    seed: int | None = None,
    grid_size: int = 20,
    max_steps_per_episode: int = 1000,
    save_prefix: str = "snake_model_",
):
    """
    Train the AI agent to play Snake
    
    Args:
        episodes: Number of training games to play
        visualize: Whether to show game during training (slows down significantly)
    """
    # Global flag for graceful shutdown
    global should_stop_training
    should_stop_training = False
    
    def signal_handler(sig, frame):
        """Handle Ctrl+C"""
        global should_stop_training
        print("\n\nReceived interrupt signal. Finishing current episode and saving...")
        should_stop_training = True
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Setup keyboard interrupt (Delete key to stop)
    print("\n" + "="*60)
    print("Training Controls:")
    print("  DELETE key - Stop training gracefully (saves model and graphs)")
    print("  Ctrl+C     - Stop training gracefully")
    print("="*60 + "\n")
    
    try:
        # Register Delete key handler
        keyboard.add_hotkey('delete', lambda: setattr(
            globals(), 'should_stop_training', True
        ))
    except Exception as e:
        print(f"Warning: Could not register Delete key ({e})")
        print("You can still use Ctrl+C to stop")
    
    # Initialize pygame for rendering (even if not visualized, needed for initialization)
    pygame.init()
    
    # Reproducibility (optional)
    if seed is not None:
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass

    # Initialize game and agent
    game = SnakeGame(grid_size=grid_size)
    # Get actual state size from game (handles extra features automatically)
    sample_state = game.get_state()
    state_size = len(sample_state)  # Should be 1610 (1600 spatial + 10 extra)
    action_size = 4  # Up, Right, Down, Left
    # Device selection
    if device == "cpu":
        use_gpu = False
    elif device == "cuda":
        if not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
        use_gpu = True
    else:
        use_gpu = True  # auto

    agent = DQNAgent(
        state_size,
        action_size,
        lr=lr,
        use_gpu=use_gpu,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        replay_size=replay_size,
        gamma=gamma,
    )

    # Resume from existing model if requested
    if mode == "resume":
        if not model_path:
            print("Error: --mode resume requires --model PATH")
            sys.exit(1)
        try:
            agent.load(model_path)
            print(f"Resumed from model: {model_path}")
        except FileNotFoundError:
            print(f"Error: model not found at {model_path}")
            sys.exit(1)
        if reset_epsilon is not None:
            agent.epsilon = float(reset_epsilon)
            print(f"Epsilon reset to {agent.epsilon}")
    
    print(f"State size: {state_size}")
    print(f"Network architecture: {state_size} -> 512 -> 256 -> 128 -> {action_size}")
    
    # Track training progress
    scores = []
    max_scores = deque(maxlen=100)  # Track last 100 max scores
    all_scores = []
    
    print("Starting training...")
    print(f"Target episodes: {episodes}")
    print(f"Device: {agent.device}")
    # GPU hardware information (if CUDA is used)
    if hasattr(agent, 'device') and getattr(agent.device, 'type', '') == 'cuda':
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            capability = f"{props.major}.{props.minor}"
            torch_ver = torch.__version__
            cuda_ver = torch.version.cuda or "unknown"
            print(f"GPU: {gpu_name}")
            print(f"CUDA: {cuda_ver} | Capability: {capability}")
            print(f"VRAM: {vram_gb:.1f} GB | PyTorch: {torch_ver}")
        except Exception as e:
            print(f"GPU info unavailable: {e}")
    
    # Create log directory and file
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/training_{timestamp}.log"
    
    # Collect system info for logs
    def _get_git_info():
        branch = commit = None
        try:
            branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
        try:
            commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            pass
        return branch, commit

    def _bytes_gb(x):
        try:
            return float(x) / (1024 ** 3)
        except Exception:
            return None

    cpu_logical = os.cpu_count() or 1
    cpu_physical = None
    cpu_freq_cur = cpu_freq_max = None
    cpu_name = platform.processor() or None
    mem_total_gb = mem_avail_gb = None
    affinity_cores = None
    if psutil is not None:
        try:
            cpu_physical = psutil.cpu_count(logical=False)
        except Exception:
            pass
        try:
            f = psutil.cpu_freq()
            if f:
                cpu_freq_cur = f.current
                cpu_freq_max = f.max
        except Exception:
            pass
        try:
            vm = psutil.virtual_memory()
            mem_total_gb = _bytes_gb(vm.total)
            mem_avail_gb = _bytes_gb(vm.available)
        except Exception:
            pass
        try:
            affinity_cores = len(psutil.Process().cpu_affinity())
        except Exception:
            pass

    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    gpu_name = None
    gpu_vram_gb = None
    gpu_capability = None
    cuda_ver = torch.version.cuda or None
    torch_ver = torch.__version__
    if hasattr(agent, 'device') and getattr(agent.device, 'type', '') == 'cuda':
        try:
            gpu_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram_gb = _bytes_gb(props.total_memory)
            gpu_capability = f"{props.major}.{props.minor}"
        except Exception:
            pass

    git_branch, git_commit = _get_git_info()

    # Capture run configuration
    run_config = {
        "mode": mode,
        "model_path": model_path,
        "reset_epsilon": reset_epsilon,
        "episodes": episodes,
        "visualize": visualize,
        "batch_size": batch_size,
        "lr": lr,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "target_update": target_update,
        "replay_size": replay_size,
        "checkpoint_interval": checkpoint_interval,
        "log_interval": log_interval,
        "device_arg": device,
        "seed": seed,
        "grid_size": grid_size,
        "max_steps_per_episode": max_steps_per_episode,
        "save_prefix": save_prefix,
    }

    system_info = {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "python": platform.python_version(),
        "git": {"branch": git_branch, "commit": git_commit},
        "cpu": {
            "name": cpu_name,
            "logical_cores": cpu_logical,
            "physical_cores": cpu_physical,
            "freq_current_mhz": cpu_freq_cur,
            "freq_max_mhz": cpu_freq_max,
            "affinity_cores": affinity_cores,
            "torch_num_threads": torch.get_num_threads(),
        },
        "memory": {
            "total_gb": mem_total_gb,
            "available_gb": mem_avail_gb,
        },
        "gpu": {
            "count": gpu_count,
            "name": gpu_name,
            "vram_gb": gpu_vram_gb,
            "capability": gpu_capability,
            "cuda": cuda_ver,
            "pytorch": torch_ver,
        },
        "parallel": None
    }

    # Print consolidated system info to console (after variables are defined)
    print("-" * 50)
    print(f"OS: {system_info['os']['system']} {system_info['os']['release']}")
    print(f"Python: {system_info['python']} | Git: {git_branch or 'n/a'}@{git_commit or 'n/a'}")
    print(f"CPU: {cpu_name or 'n/a'} | logical={cpu_logical} physical={cpu_physical or 'n/a'}"
          f" | freq={cpu_freq_cur or 'n/a'}MHz max={cpu_freq_max or 'n/a'}MHz")
    if mem_total_gb and mem_avail_gb:
        print(f"Memory: total={mem_total_gb:.2f}GB available={mem_avail_gb:.2f}GB")
    if gpu_count:
        print(f"GPU: {gpu_name or 'n/a'} | vram={gpu_vram_gb:.1f}GB | capability={gpu_capability or 'n/a'} | CUDA={cuda_ver or 'n/a'} | torch={torch_ver}")
    print(f"Execution: single-process | torch_threads={torch.get_num_threads()} | affinity={affinity_cores or 'n/a'}")

    # Initialize log data
    log_data = {
        "start_time": timestamp,
        "episodes": episodes,
        "device": str(agent.device),
        "training_history": [],
        "system_info": system_info,
        "config": run_config,
    }
    
    # Write initial log
    with open(log_file, 'w') as f:
        f.write(f"# Training started at {timestamp}\n")
        f.write(f"# Target episodes: {episodes}\n")
        f.write(f"# Device: {agent.device}\n")
        # System info header
        f.write(f"# OS: {system_info['os']['system']} {system_info['os']['release']}\n")
        f.write(f"# Python: {system_info['python']} | Git: {git_branch or 'n/a'}@{git_commit or 'n/a'}\n")
        f.write(f"# CPU: {cpu_name or 'n/a'} | logical={cpu_logical} physical={cpu_physical or 'n/a'}"
                f" | freq={cpu_freq_cur or 'n/a'}MHz max={cpu_freq_max or 'n/a'}MHz"
                f" | torch_threads={torch.get_num_threads()} | affinity={affinity_cores or 'n/a'}\n")
        f.write(f"# Memory: total={mem_total_gb:.2f}GB available={mem_avail_gb:.2f}GB\n" if (mem_total_gb and mem_avail_gb) else "")
        if gpu_count:
            f.write(f"# GPU: {gpu_name or 'n/a'} | vram={gpu_vram_gb:.1f}GB | capability={gpu_capability or 'n/a'} | CUDA={cuda_ver or 'n/a'} | torch={torch_ver}\n")
        # Config block
        f.write("# Config:\n")
        for k in [
            "mode","model_path","reset_epsilon","episodes","visualize","batch_size","lr","gamma",
            "epsilon_start","epsilon_min","epsilon_decay","target_update","replay_size",
            "checkpoint_interval","log_interval","device_arg","seed","grid_size",
            "max_steps_per_episode","save_prefix"
        ]:
            f.write(f"#   {k}={run_config[k]}\n")
        f.write("# Format: episode, avg_score, max_score, epsilon, memory_size, time_last_100, time_total, time_remaining, eps_per_s, steps_per_s\n")
        f.write("# episode,avg_score,max_score,epsilon,memory_size,time_last_100,time_total,time_remaining,eps_per_s,steps_per_s\n")
    
    # Single-process execution (no rollout workers)

    # Track time
    start_time = time.time()
    last_checkpoint_time = start_time  # last log time
    # Throughput tracking
    total_steps = 0
    last_logged_episode = -1
    last_logged_steps = 0
    eps_per_s_recent = 0.0
    steps_per_s_recent = 0.0
    
    # Create main progress bar
    pbar = tqdm(total=episodes, desc="Training", unit="episode", ncols=100, miniters=10, maxinterval=1)
    
    # Create nested progress bar for current chunk
    nested_pbar = None
    
    for episode in range(episodes):
        # Check if training should stop
        if should_stop_training:
            print(f"\nStopping training at episode {episode}/{episodes}")
            # Let current episode finish
            break
        
        game.reset()
        state = game.get_state()
        total_reward = 0
        steps = 0
        done = False
        
        # Play one episode (with timeout to prevent stuck episodes)
        steps_in_episode = 0
        
        # Play one episode
        while not done and steps_in_episode < max_steps_per_episode:
            steps_in_episode += 1
            # Choose action
            action = agent.act(state)
            
            # Take action and get result
            reward, done, ate_food = game.move(action)
            total_reward += reward
            
            if not done:
                next_state = game.get_state()
            else:
                next_state = np.zeros_like(state)
            
            # Store experience in memory
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            steps += 1
        
        # Force end if stuck
        if steps_in_episode >= max_steps_per_episode:
            game.is_game_over = True
            
            # Optional: render game (very slow, only for debugging)
            if visualize:
                game.render()
                pygame.time.delay(50)
        
        # Train the agent
        if episode > batch_size:  # 等待足够的经验（至少一个完整batch）
            loss = agent.replay(batch_size)
        
        # Update target network per configured interval
        if episode % target_update == 0:
            agent.update_target_network()
        
        # Track progress
        scores.append(game.snake_length - 1)  # Score = snake length - 1
        max_scores.append(scores[-1])
        all_scores.append(scores[-1])
        total_steps += steps
        
        # No parallel experience ingestion in single-process mode

        # Update nested progress bar (for current chunk)
        if episode % log_interval == 0:
            # Close previous nested bar if exists
            if nested_pbar is not None:
                nested_pbar.close()
            # Create new nested bar for next 100 episodes
            nested_pbar = tqdm(
                total=min(log_interval, episodes - episode),
                desc=f"Ep {episode}-{min(episode+log_interval-1, episodes-1)}",
                unit="ep",
                ncols=80,
                leave=False,
                position=1,
                disable=False,
                miniters=1,  # Update every step
                maxinterval=5  # But throttle to every 5 seconds max
            )
        
        # Update nested bar less frequently to reduce overhead
        if nested_pbar is not None and episode % 5 == 0:
            nested_pbar.update(5)
        
        # Update progress bar
        current_avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        current_max = max(scores[-100:]) if len(scores) >= 100 else max(scores) if scores else 0
        pbar.update(1)
        # Only update postfix every 10 episodes to reduce overhead
        if episode % 10 == 0:
            pbar.set_postfix({
                'avg': f'{current_avg:.2f}',
                'max': current_max,
                'eps': f'{agent.epsilon:.3f}',
                'eps/s': f'{eps_per_s_recent:.1f}',
                'steps/s': f'{steps_per_s_recent:.0f}'
            })
        
            # Print progress
            if episode % log_interval == 0:
                avg_score = np.mean(scores[-100:])
                max_score = max(scores[-100:])
                avg_max_score = np.mean(max_scores)
                epsilon = agent.epsilon
                
                # Calculate time statistics
                current_time = time.time()
                time_since_start = current_time - start_time
                time_since_checkpoint = current_time - last_checkpoint_time
                estimated_total_time = time_since_start * episodes / (episode + 1)
                remaining_time = estimated_total_time - time_since_start
                # Throughput since last log
                episodes_since_last = 0 if last_logged_episode < 0 else (episode - last_logged_episode)
                steps_since_last = total_steps - last_logged_steps
                eps_per_s = (episodes_since_last / time_since_checkpoint) if time_since_checkpoint > 0 else 0.0
                steps_per_s = (steps_since_last / time_since_checkpoint) if time_since_checkpoint > 0 else 0.0
                eps_per_s_recent = eps_per_s
                steps_per_s_recent = steps_per_s
                
                # Prepare log entry
                log_entry = {
                    "episode": episode,
                    "avg_score": float(avg_score),
                    "max_score": int(max_score),
                    "avg_max_score": float(avg_max_score),
                    "epsilon": float(epsilon),
                    "memory_size": len(agent.memory),
                    "time_last_100": time_since_checkpoint,
                    "time_total": time_since_start,
                    "time_remaining": remaining_time,
                    "eps_per_s": float(eps_per_s),
                    "steps_per_s": float(steps_per_s)
                }
                log_data["training_history"].append(log_entry)
                
                # Write to log file (one line per update)
                with open(log_file, 'a') as f:
                    # Format: episode, avg_score, max_score, epsilon, memory_size, time_last_100, time_total, time_remaining, eps_per_s, steps_per_s
                    f.write(f"{episode},{avg_score:.2f},{max_score},{epsilon:.4f},{len(agent.memory)},{time_since_checkpoint:.2f},{time_since_start:.2f},{remaining_time:.2f},{eps_per_s:.2f},{steps_per_s:.0f}\n")
                
                print(f"Episode: {episode}/{episodes}")
                print(f"  Average Score: {avg_score:.2f}")
                print(f"  Max Score (last 100): {max_score}")
                print(f"  Avg Max Score (last 100): {avg_max_score:.2f}")
                print(f"  Epsilon: {epsilon:.4f}")
                print(f"  Memory size: {len(agent.memory)}")
                print(f"  Time (last 100): {time_since_checkpoint:.2f}s")
                print(f"  Time (total): {time_since_start:.2f}s")
                print(f"  Estimated remaining: {remaining_time:.2f}s")
                print(f"  Episodes/s: {eps_per_s:.2f}")
                print(f"  Steps/s: {steps_per_s:.0f}")
                print(f"  Log saved to: {log_file}")
                print("-" * 50)
                
                # Update last-log trackers
                last_checkpoint_time = current_time
                last_logged_episode = episode
                last_logged_steps = total_steps
        
        # Save model every checkpoint_interval episodes
        if episode % checkpoint_interval == 0 and episode > 0:
            os.makedirs("models", exist_ok=True)
            agent.save(f"models/{save_prefix}{episode}.pth")
            print(f"Model saved at episode {episode} -> models/{save_prefix}{episode}.pth")
    
    # Close progress bars
    if nested_pbar is not None:
        nested_pbar.close()
    pbar.close()
    
    # No workers to stop in single-process mode

    # Final save
    os.makedirs("models", exist_ok=True)
    
    # Check if training was stopped early
    if should_stop_training:
        print("\n" + "="*60)
        print("Training stopped by user (Delete key or Ctrl+C)")
        print(f"Completed {len(scores)} episodes out of {episodes}")
        agent.save("models/snake_model_interrupted.pth")
        print("Model saved as snake_model_interrupted.pth")
    else:
        agent.save("models/snake_model_final.pth")
        print("Training complete! Final model saved.")
    print("="*60)
    
    # Finalize log data
    final_time = time.time()
    total_time = final_time - start_time
    log_data["end_time"] = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_data["total_time"] = total_time
    log_data["final_scores"] = {
        "avg_last_100": float(np.mean(scores[-100:])),
        "max_all": int(max(scores)),
        "all_scores": [int(s) for s in all_scores]
    }
    
    # Save JSON log
    json_log_file = log_file.replace('.log', '.json')
    with open(json_log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # Write final summary to log
    with open(log_file, 'a') as f:
        f.write(f"# Training Complete! Total time: {total_time/60:.2f} minutes, Avg score: {np.mean(scores[-100:]):.2f}, Max score: {max(scores)}\n")
    
    print(f"Training log saved to: {log_file}")
    print(f"JSON data saved to: {json_log_file}")
    
    # Plot training results
    plot_training_progress(all_scores, episodes)
    
    pygame.quit()
    
    # Cleanup keyboard listener
    try:
        keyboard.unhook_all()
    except:
        pass


def plot_training_progress(scores, episodes):
    """
    Plot training progress as a graph
    
    Shows how the AI's performance improves over time
    """
    # Create graphs directory
    os.makedirs("graphs", exist_ok=True)
    
    # Plot 1: All scores
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores, alpha=0.6, color='blue')
    plt.title('Training Progress - All Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Moving average
    plt.subplot(1, 2, 2)
    window = 100
    moving_avg = []
    for i in range(window, len(scores)):
        moving_avg.append(np.mean(scores[i-window:i]))
    
    plt.plot(moving_avg, color='red', linewidth=2)
    plt.title(f'Moving Average (window={window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graphs/training_progress.png', dpi=300, bbox_inches='tight')
    print("Training graph saved to graphs/training_progress.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Snake AI (DQN / Double DQN)")
    # Mode / resume
    parser.add_argument("--mode", choices=["new", "resume"], default="new", help="Start new training or resume from a model")
    parser.add_argument("--model", type=str, default=None, help="Path to model to resume from (required if --mode resume)")
    parser.add_argument("--reset-epsilon", type=float, default=None, help="Reset epsilon when resuming (e.g., 0.1)")
    
    # Core training
    parser.add_argument("--episodes", type=int, default=100000, help="Total training episodes")
    parser.add_argument("--visualize", action="store_true", help="Render during training (slow)")
    parser.add_argument("--batch-size", type=int, default=128, help="Replay batch size")
    parser.add_argument("--lr", type=float, default=0.0003, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum epsilon")
    parser.add_argument("--epsilon-decay", type=float, default=0.9995, help="Epsilon decay factor")
    parser.add_argument("--target-update", type=int, default=50, help="Target network update interval (episodes)")
    parser.add_argument("--replay-size", type=int, default=100000, help="Replay buffer size")
    
    
    # Logging / saving
    parser.add_argument("--checkpoint-interval", type=int, default=1000, help="Save checkpoint every N episodes")
    parser.add_argument("--log-interval", type=int, default=100, help="Print/log interval (episodes)")
    parser.add_argument("--save-prefix", type=str, default="snake_model_", help="Checkpoint filename prefix")
    
    # Env / system
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Device selection")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--grid-size", type=int, default=20, help="Game grid size")
    parser.add_argument("--max-steps-per-episode", type=int, default=1000, help="Per-episode step cap")
    
    args = parser.parse_args()
    
    train_snake_ai(
        episodes=args.episodes,
        visualize=args.visualize,
        mode=args.mode,
        model_path=args.model,
        reset_epsilon=args.reset_epsilon,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        replay_size=args.replay_size,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        device=args.device,
        seed=args.seed,
        grid_size=args.grid_size,
        max_steps_per_episode=args.max_steps_per_episode,
        save_prefix=args.save_prefix,
    )

