# Snake AI - DQN with Major Upgrades (English) / 贪吃蛇AI - 强化学习重大升级（中文）

## 🎮 Project Overview (English)

A Snake AI trained with Deep Q-Networks (DQN), upgraded for stability and long-snake performance:
- Double DQN (main net selects actions; target net evaluates) to reduce Q overestimation
- Deeper network (512→256→128→4) with Dropout(0.2)
- Enhanced state: 20×20×4 grid (1600) + 10 extra features (food dx/dy, normalized snake length, 4-direction danger flags, normalized safe-space count)
- Reward shaping for multi-food runs (accelerating bonus) and safety awareness
- One-key graceful stop (Delete/Ctrl+C) that still saves model, logs and graphs

## 🧠 How It Works

### Algorithm
- DQN with Experience Replay and Target Network
- Double DQN action selection/evaluation for stability

### Neural Network (Current)
```
Input (1610) → 512 → 256 → 128 → Output (4 actions)
Dropout(0.2) after first two hidden layers
```

### State Representation (Current)
- 4-layer grid: walls, snake body, snake head, food (shape 4×20×20 = 1600)
- Extra 10 features:
  - Relative food dx, dy (normalized by grid size)
  - Normalized snake length
  - Danger flags for [up, right, down, left] (1 = danger, 0 = safe)
  - Normalized safe-space count in 4 directions

### Reward System (Current)
- Eat food: +15 base; accelerating consecutive bonus with a quadratic term
  - Example: 1st +15; 2nd +19; 3rd +24; 5th +35; 10th +87
- Move relative to food: closer +3; unchanged −0.5; farther −1
- Safety: only penalize dead-ends (−10) and full surround (−5)
- Wall proximity: light penalty only when 1-cell from wall
- Survival penalty: adaptive — decreases as snake grows (longer snake gets more planning time)
- Collisions: −10 for wall or self

## ⚙️ Training Configuration (Current)
- Episodes: 100,000
- Batch size: 128
- Learning rate: 0.0003
- Discount factor (Gamma): 0.95
- Epsilon: start 1.0 → min 0.05, slow decay (0.9995)
- Target network update: every 50 episodes
- Checkpoints: every 1,000 episodes

## 📋 Requirements
- Python 3.11 recommended
- See `requirements.txt` (includes `keyboard` for one-key stop)

## 🚀 Installation
```bash
pip install -r requirements.txt
```

## 🎯 Usage

### Train
```bash
python train.py
```
Controls during training:
- Delete: graceful stop (saves `models/snake_model_interrupted.pth`, logs, JSON, and graphs)
- Ctrl+C: graceful stop as well

Outputs:
- Models: checkpoints per 1,000 episodes, final or interrupted model
- Logs: `logs/training_*.log` (CSV-like), `logs/training_*.json`
- Graphs: `graphs/training_progress.png`

### Resume training
```bash
python train.py --mode resume --model models/snake_model_20000.pth --reset-epsilon 0.1
```
Notes:
- `--mode resume` loads an existing model and continues training
- `--reset-epsilon` optionally resets exploration rate (e.g., 0.1)

### Advanced configuration (examples)
```bash
# New training with custom save/log cadence
python train.py --mode new \
  --episodes 100000 \
  --checkpoint-interval 500 \
  --log-interval 50 \
  --save-prefix expA_

# Heavier training setup
python train.py \
  --episodes 150000 \
  --batch-size 256 \
  --lr 0.0002 \
  --gamma 0.97 \
  --epsilon-start 1.0 --epsilon-min 0.05 --epsilon-decay 0.9995 \
  --target-update 50 \
  --replay-size 200000 \
  --device auto \
  --seed 123 \
  --grid-size 20 \
  --max-steps-per-episode 1200
```

### All CLI options (summary)
- Mode / resume
  - `--mode {new|resume}` (default new)
  - `--model PATH` (required for resume)
  - `--reset-epsilon FLOAT` (optional)
- Core training
  - `--episodes INT` (default 100000)
  - `--visualize` (render during training)
  - `--batch-size INT` (default 128)
  - `--lr FLOAT` (default 0.0003)
  - `--gamma FLOAT` (default 0.95)
  - `--epsilon-start FLOAT` (default 1.0)
  - `--epsilon-min FLOAT` (default 0.05)
  - `--epsilon-decay FLOAT` (default 0.9995)
  - `--target-update INT` (default 50)
  - `--replay-size INT` (default 100000)
- Logging / saving
  - `--checkpoint-interval INT` (default 1000)
  - `--log-interval INT` (default 100)
  - `--save-prefix STR` (default snake_model_)
- Env / system
  - `--device {auto|cuda|cpu}` (default auto)
  - `--seed INT`
  - `--grid-size INT` (default 20)
  - `--max-steps-per-episode INT` (default 1000)

### Train options — Beginner Guide

- `--mode {new|resume}` (default: new)
  - What it does: Choose to start fresh or continue from a saved model.
  - When to change: Use resume if you already have a promising checkpoint and want to keep training.
  - Pitfall: Resuming requires the same board size as the model was trained with (same `--grid-size`). If you changed grid-size, start new.
  - Example:
    ```bash
    python train.py --mode resume --model models/snake_model_20000.pth
    ```

- `--model PATH` (resume only)
  - What it does: The checkpoint file to load (a .pth saved by this project).
  - When to change: Required whenever `--mode=resume`.
  - Pitfall: Using a model from a different code/version may cause shape mismatch.

- `--reset-epsilon FLOAT` (resume only)
  - What it does: Reset the exploration rate when resuming (e.g., 0.1).
  - When to change: If training got “stuck” in a local pattern, bump exploration to escape it.
  - Typical values: 0.05–0.2. Higher = more exploration, slower short-term reward, but may learn better strategies.

- `--episodes INT` (default: 100000)
  - What it does: How many games to train in total.
  - Trade-off: More episodes = more stable learning, but longer time.

- `--visualize` (flag)
  - What it does: Show the game window during training.
  - When to use: Debug only. It is 10–100× slower than headless training.
  - Tip: Keep it off for long runs.

- `--batch-size INT` (default: 128)
  - What it does: Number of experiences used per learning step.
  - Trade-off: Larger = smoother updates but needs more GPU memory. With 12GB GPU, 128–256 is a good range.
  - Tip: If you see out-of-memory, lower this first.

- `--lr FLOAT` (default: 0.0003)
  - What it does: Learning rate for the optimizer.
  - Guidance: 2e-4 to 5e-4 is typical for this network. Too high may diverge; too low learns slowly.

- `--gamma FLOAT` (default: 0.95)
  - What it does: How much the agent values future rewards.
  - Guidance: 0.95 is balanced. 0.97–0.99 focuses more on long-term but can slow learning.

- `--epsilon-start FLOAT`, `--epsilon-min FLOAT`, `--epsilon-decay FLOAT` (defaults: 1.0, 0.05, 0.9995)
  - What they do: Control the balance between exploration and exploitation over time.
  - Easy mental model:
    - start = how curious at the very beginning (keep at 1.0).
    - min = curiosity floor later on (0.03–0.1 common).
    - decay = how fast curiosity drops each step (0.995 faster, 0.999–0.9997 slower).
  - If you see “moving back-and-forth in place”, slow the decay and/or raise epsilon-min.

- `--target-update INT` (default: 50)
  - What it does: How often to copy the main network to the target network (in episodes).
  - Guidance: 50–200 is typical. Too frequent reduces stability; too infrequent slows adoption of new behavior.

- `--replay-size INT` (default: 100000)
  - What it does: How many past experiences to keep in the replay buffer.
  - Trade-off: Larger buffers improve stability but use more memory and sampling time. 10k–100k is typical for Snake.
  - Note: Resuming loads weights, not the old buffer.

- `--checkpoint-interval INT` (default: 1000)
  - What it does: Save a model checkpoint every N episodes.
  - Guidance: 500–5000. Smaller = more files, more I/O.

- `--log-interval INT` (default: 100)
  - What it does: How often to print/write logs and update the nested progress bar window.
  - Guidance: Smaller gives more frequent feedback, slightly more overhead.

- `--save-prefix STR` (default: snake_model_)
  - What it does: Prefix for checkpoint filenames under `models/`.
  - Note: Final filenames are fixed: `snake_model_final.pth` or `snake_model_interrupted.pth`. The prefix affects periodic checkpoints only.
  - Example:
    ```bash
    python train.py --save-prefix expA_ --checkpoint-interval 500
    ```

- `--device {auto|cuda|cpu}` (default: auto)
  - What it does: Where to run the neural network.
  - Guidance: auto uses GPU if available; cpu forces CPU (slow); cuda forces GPU (falls back with a warning if unavailable).

- `--seed INT` (default: None)
  - What it does: Set random seeds for fairer comparisons between runs.
  - Note: Exact reproducibility is not guaranteed due to GPU and environment randomness.

- `--grid-size INT` (default: 20)
  - What it does: Board size; affects state dimension dramatically (4×grid² + 10).
  - Pitfall: Changing this makes models incompatible; you cannot resume across different grid sizes.

- `--max-steps-per-episode INT` (default: 1000)
  - What it does: Hard cap to end episodes that loop without progress.
  - Guidance: 500–2000 is a good range. Too small may cut off good plans; too large wastes time.

Common pitfalls & tips
- Resuming with a different `--grid-size` will fail. Start new instead.
- If learning stalls or loops in place: slow `--epsilon-decay` and/or increase `--epsilon-min`; when resuming, try `--reset-epsilon 0.1`.
- For instability (loss spikes): try larger `--batch-size` or larger `--target-update` (less frequent sync).
- For GPU memory issues: reduce `--batch-size` first.

Quick recipes
- Fresh run (safe defaults):
  ```bash
  python train.py --episodes 100000 --batch-size 128 --target-update 50
  ```
- Resume and add exploration:
  ```bash
  python train.py --mode resume --model models/snake_model_20000.pth --reset-epsilon 0.1
  ```
- Faster checkpoints and logs to monitor closely:
  ```bash
  python train.py --checkpoint-interval 500 --log-interval 50 --save-prefix expA_
  ```

### Play options — Beginner Guide

- model (positional; default: `models/snake_model_final.pth`)
  - What it does: Which trained model to visualize.
  - Pitfall: Must match the grid-size used during its training.

- `--fps INT` (default: 10)
  - What it does: Playback speed (frames per second).
  - Guidance: 5–20 for debugging; 30–60 for smoother viewing.

- `--no-grid` (flag)
  - What it does: Hide grid lines for a cleaner view and slight speedup.

During play
- ESC/Q: quit | SPACE: pause/resume | +/-: change FPS | R: restart current game
- Right info panel shows score/best/length/FPS, loop/stuck detection with auto-restart, and death reason.

### Watch the AI play
```bash
python play.py                # use final
python play.py models/snake_model_20000.pth
```
Controls:
- ESC / Q: quit; SPACE: pause; +/-: speed; R: restart
- Info panel shows score/best/length/FPS; loop/stuck detection and auto-restart

## 📊 Expected Results (Indicative)
- Early (≤10k): avg 1–3, max 5–15
- Mid (10–50k): avg 3–7, max 15–40
- Late (50–100k): avg 6–12+, max 40–100+
(Exact values vary with randomness and hardware)

## 📁 Project Structure
```
snake/
├── game.py              # Environment, state, rewards, rendering
├── ai_agent.py          # DQN + Double DQN, network, replay, training
├── train.py             # Training loop, logging, one-key stop, graphing
├── play.py              # Visualize the agent with controls & diagnostics
├── requirements.txt     # Dependencies (incl. keyboard)
├── models/              # Saved models (checkpoints, final, interrupted)
├── logs/                # .log (human) + .json (structured)
└── graphs/              # training_progress.png
```

## 🐛 Troubleshooting
- Shape mismatch (mat1/mat2): ensure state size matches code (now auto-derived from `game.get_state()`)
- Delete key not working: use Ctrl+C to stop
- Slow training: ensure CUDA is used; reduce batch size if needed

## 📚 References
- Mnih et al. 2015 (DQN), Experience Replay, Target Networks
- Double DQN, Dueling DQN, Prioritized Replay (future work)

---

## 🎮 项目概览（中文）

本项目基于 DQN 训练贪吃蛇 AI，并针对稳定性与长蛇能力做了重大升级：
- Double DQN（主网选动作、目标网评估）降低 Q 值过估计
- 更深网络：512→256→128→4，并加入 Dropout(0.2)
- 增强状态：20×20×4 栅格（1600）+ 10 个额外特征（食物相对距离、蛇长归一化、四方向危险标记、可行动安全方向数）
- 奖励塑形：鼓励连续吃多颗、兼顾安全与效率
- 一键优雅停止（Delete/Ctrl+C），仍会保存模型、日志与曲线

## 🧠 工作原理

### 算法
- DQN + 经验回放 + 目标网络
- Double DQN：主网络选择，目标网络评估

### 神经网络（当前）
```
输入(1610) → 512 → 256 → 128 → 输出(4动作)
前两层后各有 Dropout(0.2)
```

### 状态表示（当前）
- 4 层网格：墙、蛇身、蛇头、食物（4×20×20=1600）
- 额外 10 维：食物相对 dx/dy、蛇长归一化、四方向危险标记、安全方向数量（归一化）

### 奖励系统（当前）
- 吃食物：基础 +15；随连续个数按加速方式递增（含平方项）
  - 例：第1个 +15；第2个 +19；第3个 +24；第5个 +35；第10个 +87
- 相对食物位移：更近 +3；不变 −0.5；更远 −1
- 安全：仅在死路（−10）与四面楚歌（−5）时重罚
- 贴墙：仅 1 格距离时轻微惩罚
- 存活惩罚：随长度降低（蛇越长越宽松，便于规划）
- 撞墙/撞自己：−10

## ⚙️ 训练配置（当前）
- 训练轮数：100,000
- 批大小：128
- 学习率：0.0003
- 折扣因子（Gamma）：0.95
- Epsilon：起始 1.0，最小 0.05，衰减 0.9995（更慢）
- 目标网络更新：每 50 轮
- Checkpoint：每 1,000 轮保存一次

## 📋 环境依赖
- 建议 Python 3.11
- 详见 `requirements.txt`（含 `keyboard`，用于一键停止）

## 🚀 安装
```bash
pip install -r requirements.txt
```

## 🎯 使用方法

### 训练
```bash
python train.py
```
训练过程中可用：
- Delete：优雅停止（保存 `models/snake_model_interrupted.pth`、日志、JSON、曲线图）
- Ctrl+C：同样优雅停止

输出内容：
- 模型：每 1000 轮 checkpoint，最终/中断模型
- 日志：`logs/training_*.log`（人类可读）、`logs/training_*.json`（结构化）
- 图表：`graphs/training_progress.png`

### 继续训练
```bash
python train.py --mode resume --model models/snake_model_20000.pth --reset-epsilon 0.1
```
说明：
- `--mode resume` 载入已有模型继续训练
- `--reset-epsilon` 可选，重置探索率（例如 0.1）

### 高级配置（示例）
```bash
# 自定义保存/日志频率的新训练
python train.py --mode new \
  --episodes 100000 \
  --checkpoint-interval 500 \
  --log-interval 50 \
  --save-prefix expA_

# 更重配置
python train.py \
  --episodes 150000 \
  --batch-size 256 \
  --lr 0.0002 \
  --gamma 0.97 \
  --epsilon-start 1.0 --epsilon-min 0.05 --epsilon-decay 0.9995 \
  --target-update 50 \
  --replay-size 200000 \
  --device auto \
  --seed 123 \
  --grid-size 20 \
  --max-steps-per-episode 1200
```

### 全部命令行选项（摘要）
- 新/续训
  - `--mode {new|resume}`（默认 new）
  - `--model 路径`（resume 必填）
  - `--reset-epsilon 浮点数`（可选）
- 核心训练
  - `--episodes 整数`（默认 100000）
  - `--visualize`（训练期渲染）
  - `--batch-size 整数`（默认 128）
  - `--lr 浮点数`（默认 0.0003）
  - `--gamma 浮点数`（默认 0.95）
  - `--epsilon-start 浮点数`（默认 1.0）
  - `--epsilon-min 浮点数`（默认 0.05）
  - `--epsilon-decay 浮点数`（默认 0.9995）
  - `--target-update 整数`（默认 50）
  - `--replay-size 整数`（默认 100000）
- 日志/保存
  - `--checkpoint-interval 整数`（默认 1000）
  - `--log-interval 整数`（默认 100）
  - `--save-prefix 字符串`（默认 snake_model_）
- 环境/系统
  - `--device {auto|cuda|cpu}`（默认 auto）
  - `--seed 整数`
  - `--grid-size 整数`（默认 20）
  - `--max-steps-per-episode 整数`（默认 1000）

### 训练选项·新手友好版（中文）

- `--mode {new|resume}`（默认：new）
  - 作用：选择“新训练”或“在已有模型上继续训练”。
  - 什么时候改：已有不错的模型想进一步提升时，选择 resume。
  - 易错点：resume 需要与模型训练时相同的棋盘大小（`--grid-size`）。若改了 grid-size，请用 new 重新训练。
  - 示例：
    ```bash
    python train.py --mode resume --model models/snake_model_20000.pth
    ```

- `--model 路径`（仅 resume）
  - 作用：要读取的 .pth 检查点。
  - 什么时候改：`--mode=resume` 时必填。
  - 易错点：用其它项目生成的模型可能形状不匹配。

- `--reset-epsilon 浮点数`（仅 resume）
  - 作用：续训时重置探索率（例如 0.1）。
  - 什么时候改：训练陷入“套路循环”时，提高探索帮助跳出局部最优。
  - 常用范围：0.05–0.2。越大探索越多、短期分数可能变慢，但长期更好。

- `--episodes 整数`（默认：100000）
  - 作用：总训练回合数。
  - 权衡：越多越稳，但耗时更长。

- `--visualize`（开关）
  - 作用：训练时显示游戏画面。
  - 什么时候用：仅调试用。会比不显示慢 10–100 倍。
  - 建议：长时间训练关闭它。

- `--batch-size 整数`（默认：128）
  - 作用：每次学习使用的样本数。
  - 权衡：更大更稳但更占显存。12GB 显卡常用 128–256。
  - 提示：显存不够就先降这里。

- `--lr 浮点数`（默认：0.0003）
  - 作用：学习率。
  - 建议：2e-4～5e-4 比较合适。过大会发散、过小会很慢。

- `--gamma 浮点数`（默认：0.95）
  - 作用：对“未来奖励”的重视程度。
  - 建议：0.95 比较均衡；0.97～0.99 更看重长期但可能学得更慢。

- `--epsilon-start / --epsilon-min / --epsilon-decay`（默认：1.0 / 0.05 / 0.9995）
  - 作用：控制“探索 vs 利用”的节奏。
  - 简单理解：
    - start：一开始有多“好奇”（建议 1.0）。
    - min：后期最低“好奇度”（0.03～0.1 常见）。
    - decay：每步降低“好奇”的速度（0.995 快、0.999～0.9997 慢更稳）。
  - 如果出现“原地来回走”，放慢 decay 或把 min 提高一些。

- `--target-update 整数`（默认：50）
  - 作用：每隔多少回合同步一次目标网络。
  - 建议：50～200。太频繁会降低稳定性，太稀疏又学得慢。

- `--replay-size 整数`（默认：100000）
  - 作用：经验回放池容量。
  - 权衡：越大越稳但占内存与采样时间。贪吃蛇常用 1～10 万。
  - 注意：resume 只加载权重，不会加载旧的回放缓存。

- `--checkpoint-interval 整数`（默认：1000）
  - 作用：每 N 回合保存一次模型。
  - 建议：500～5000。太小文件多、I/O 多。

- `--log-interval 整数`（默认：100）
  - 作用：打印/写日志频率，也会决定嵌套进度条的窗口。
  - 建议：越小越密集，略有性能开销。

- `--save-prefix 字符串`（默认：snake_model_）
  - 作用：`models/` 下周期性保存的检查点文件名前缀。
  - 注意：最终文件名固定（`snake_model_final.pth` 或 `snake_model_interrupted.pth`），前缀只影响中间的 checkpoint。
  - 示例：
    ```bash
    python train.py --save-prefix expA_ --checkpoint-interval 500
    ```

- `--device {auto|cuda|cpu}`（默认：auto）
  - 作用：选择运行设备。
  - 建议：auto 会自动用 GPU（有就用），cpu 强制走 CPU（较慢），cuda 强制 GPU（如果不可用会回退并提示）。

- `--seed 整数`（默认：None）
  - 作用：设置随机种子，便于公平对比不同实验。
  - 注意：GPU 与环境的随机性使结果无法完全复现。

- `--grid-size 整数`（默认：20）
  - 作用：棋盘大小；会显著影响状态维度（4×grid² + 10）。
  - 易错点：改了 grid-size 的模型与旧模型不兼容，不能直接 resume。

- `--max-steps-per-episode 整数`（默认：1000）
  - 作用：每回合最大步数上限，防止无意义循环。
  - 建议：500～2000。过小可能截断好策略，过大浪费时间。

常见误区与建议
- 续训时改了 `--grid-size` 会因维度不符而失败；请新训。
- 若出现“卡在原地”，放慢 `--epsilon-decay`、提高 `--epsilon-min`；续训时也可用 `--reset-epsilon 0.1`。
- 训练不稳：尝试增大 `--batch-size` 或增大 `--target-update`（降低同步频率）。
- 显存不足：优先降低 `--batch-size`。

新手快捷方案
- 从零开始（稳健默认）：
  ```bash
  python train.py --episodes 100000 --batch-size 128 --target-update 50
  ```
- 续训并提升探索：
  ```bash
  python train.py --mode resume --model models/snake_model_20000.pth --reset-epsilon 0.1
  ```
- 更密集的保存与日志，便于观察：
  ```bash
  python train.py --checkpoint-interval 500 --log-interval 50 --save-prefix expA_
  ```

### 可视化选项·新手友好版（中文）

- model（位置参数；默认：`models/snake_model_final.pth`）
  - 作用：要展示的已训练模型。
  - 易错点：需与其训练时的 `--grid-size` 保持一致。

- `--fps 整数`（默认：10）
  - 作用：播放速度（帧率）。
  - 建议：调试 5～20；演示 30～60。

- `--no-grid`（开关）
  - 作用：隐藏网格线，界面更简洁，略快。

运行时按键
- ESC/Q：退出 | SPACE：暂停/继续 | +/-：调节速度 | R：重开当前对局
- 右侧信息面板会显示分数/最佳/长度/FPS、循环/卡住检测与自动重启、以及死亡原因

### 观看 AI 对局
```bash
python play.py
python play.py models/snake_model_20000.pth
```
操作：
- ESC / Q：退出；SPACE：暂停；+/-：调速；R：重开
- 右侧信息面板：分数/最佳/长度/FPS；循环/卡住检测与自动重启

## 📊 预期结果（示意）
- 早期（≤10k）：平均 1–3，最高 5–15
- 中期（10–50k）：平均 3–7，最高 15–40
- 后期（50–100k）：平均 6–12+，最高 40–100+
（受随机性与硬件影响，波动属正常）

## 📁 项目结构
```
snake/
├── game.py              # 环境、状态、奖励、渲染
├── ai_agent.py          # DQN + Double DQN，网络、回放、训练
├── train.py             # 训练主循环、日志、一键停止、绘图
├── play.py              # 可视化与诊断（信息面板、循环/卡住检测）
├── requirements.txt     # 依赖（含 keyboard）
├── models/              # 模型（checkpoint、final、interrupted）
├── logs/                # .log（人类可读）+ .json（结构化）
└── graphs/              # training_progress.png
```

## 🐛 故障排查
- 形状不匹配（mat1/mat2）：确保状态维度与代码一致（现默认从 `game.get_state()` 自动推断）
- Delete 无效：请使用 Ctrl+C 停止
- 训练变慢：确认 CUDA 在使用；必要时降低 batch size

## 📚 参考资料
- Mnih 等（2015）：DQN、经验回放、目标网络
- Double DQN、Dueling DQN、优先级回放（后续可选）

## 🎉 祝玩得开心！

从零开始训练你的贪吃蛇 AI，观察它逐步学会吃更多食物与更稳健的生存策略！

