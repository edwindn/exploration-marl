import gymnasium as gym
import numpy as np
from PIL import Image as PILImage
import yaml
from pathlib import Path


class NavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 render_mode=None,
                 config_path=None,
                 max_episode_length=100,
        ):
        # Load configuration from YAML
        if config_path is None:
            config_path = Path(__file__).parent / "env_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        env_config = config['env']
        action_type = env_config['action_type']
        size = env_config['size']
        agent_window_size = env_config['agent_window_size']
        agent_step_size = env_config['agent_step_size']
        terminal_radius = env_config['terminal_radius']
        reward_type = env_config['reward_type']
        background = env_config['background']

        assert action_type in ["discrete", "continuous"]
        assert reward_type in ["dense", "sparse"]

        # Handle "auto" size option
        if size == "auto":
            if background is None:
                raise ValueError("Cannot use size='auto' when background is set to null. "
                               "Auto-sizing requires a background PNG image.")
            # Construct background path from string
            bg_path = Path(__file__).parent / f"{background}.png"
            if not bg_path.exists():
                raise FileNotFoundError(f"Background image not found: {bg_path}")
            # Load the image to get its dimensions
            bg_image = PILImage.open(bg_path).convert("RGB")
            width, height = bg_image.size
            if width != height:
                raise ValueError(f"Background image must be square for auto-sizing. "
                               f"Got dimensions: {width}x{height}")
            size = width

        self.size = size

        # Parse agent_window_size (can be integer pixels or percentage string like "25%")
        if isinstance(agent_window_size, str):
            if agent_window_size.endswith('%'):
                try:
                    percentage = float(agent_window_size[:-1])
                    if percentage <= 0 or percentage > 100:
                        raise ValueError(f"Percentage must be between 0 and 100, got {percentage}%")
                    obs_size = int(size * percentage / 100)
                except ValueError as e:
                    raise ValueError(f"Invalid percentage format for agent_window_size: '{agent_window_size}'. "
                                   f"Expected format like '25%'. Error: {e}")
            else:
                raise ValueError(f"Invalid string format for agent_window_size: '{agent_window_size}'. "
                               f"Expected an integer or percentage string like '25%'.")
        elif isinstance(agent_window_size, (int, float)):
            obs_size = int(agent_window_size)
            if obs_size > size:
                raise ValueError(f"agent_window_size ({obs_size} pixels) cannot be larger than "
                               f"environment size ({size} pixels)")
            if obs_size <= 0:
                raise ValueError(f"agent_window_size must be positive, got {obs_size}")
        else:
            raise ValueError(f"agent_window_size must be an integer or percentage string, "
                           f"got type {type(agent_window_size)}")

        self._obs_size = obs_size

        # Parse agent_step_size (can be integer pixels or percentage string like "5%")
        if isinstance(agent_step_size, str):
            if agent_step_size.endswith('%'):
                try:
                    percentage = float(agent_step_size[:-1])
                    if percentage <= 0 or percentage > 100:
                        raise ValueError(f"Percentage must be between 0 and 100, got {percentage}%")
                    step_size = int(size * percentage / 100)
                except ValueError as e:
                    raise ValueError(f"Invalid percentage format for agent_step_size: '{agent_step_size}'. "
                                   f"Expected format like '5%'. Error: {e}")
            else:
                raise ValueError(f"Invalid string format for agent_step_size: '{agent_step_size}'. "
                               f"Expected an integer or percentage string like '5%'.")
        elif isinstance(agent_step_size, (int, float)):
            step_size = int(agent_step_size)
            if step_size <= 0:
                raise ValueError(f"agent_step_size must be positive, got {step_size}")
        else:
            raise ValueError(f"agent_step_size must be an integer or percentage string, "
                           f"got type {type(agent_step_size)}")

        self._step_size = step_size
        self.render_mode = render_mode
        self.action_type = action_type
        self.terminal_radius = terminal_radius
        self.reward_type = reward_type
        self.max_episode_length = max_episode_length
        self.step_count = 0

        if background is None:
            # White background when background is null
            self._background = np.full((size, size, 3), 255, dtype=np.uint8)
        else:
            # Load background image from string name
            bg_path = Path(__file__).parent / f"{background}.png"
            if not bg_path.exists():
                raise FileNotFoundError(f"Background image not found: {bg_path}")
            bg = PILImage.open(bg_path).convert("RGB").resize((size, size))
            self._background = np.array(bg, dtype=np.uint8)  # shape (size, size, 3)

        # Observation: crop of the frame centred on the agent
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self._obs_size, self._obs_size, 3), dtype=np.uint8
        )
        if action_type == "discrete":
            # Actions: 0=stay, 1=up, 2=down, 3=left, 4=right
            self.action_space = gym.spaces.Discrete(5)
        else:
            # Actions: [dx, dy] - direct x and y displacements
            self.action_space = gym.spaces.Box(
                low=np.array([-10.0, -10.0], dtype=np.float32),
                high=np.array([10.0, 10.0], dtype=np.float32),
            )

        self._goal_pos = np.array([size * 0.5, size * 0.5], dtype=np.float32)
        self._agent_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        half = self._obs_size // 2
        self._agent_pos = self.np_random.integers(half, self.size - half, size=2).astype(np.float32)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        half = self._obs_size // 2
        lo, hi = half, self.size - 1 - half
        if self.action_type == "discrete":
            delta = {0: (0, 0), 1: (0, -self._step_size), 2: (0, self._step_size), 3: (-self._step_size, 0), 4: (self._step_size, 0)}[action]
            self._agent_pos = np.clip(
                self._agent_pos + np.array(delta, dtype=np.float32),
                lo, hi
            )
        else:
            dx, dy = float(action[0]), float(action[1])
            self._agent_pos = np.clip(
                self._agent_pos + np.array([dx, dy], dtype=np.float32),
                lo, hi
            )

        reward = self._get_reward()
        terminated = False
        truncated = False
        #truncated = self.step_count >= self.max_episode_length

        if self.render_mode == "human":
            self._render_frame()

        info = {
            "reward": reward,
            "step": self.step_count,
            "done": terminated or truncated
        }

        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        frame = self._build_frame()
        x0, y0 = self._obs_window_origin()
        s = self._obs_size
        return frame[y0:y0 + s, x0:x0 + s]

    def _obs_window_origin(self):
        """Return (x0, y0) top-left of the observation window centred on the agent."""
        half = self._obs_size // 2
        cx, cy = int(round(self._agent_pos[0])), int(round(self._agent_pos[1]))
        return cx - half, cy - half

    def _get_reward(self):
        dist = self._agent_pos - self._goal_pos
        dist = np.sqrt(dist[0]**2 + dist[1]**2)
        if self.reward_type == "sparse":
            reward = 1 if dist <= self.terminal_radius else 0
        else:
            reward = max(0, 1 - dist / self.size)
        return float(reward)

    def _build_frame(self):
        frame = self._background.copy()
        self._draw_circle(frame, self._goal_pos, radius=5, color=(255, 0, 0))
        self._draw_circle(frame, self._agent_pos, radius=5, color=(0, 0, 0))
        return frame

    def _draw_circle(self, frame, pos, radius, color):
        cx, cy = int(round(pos[0])), int(round(pos[1]))
        h, w = frame.shape[:2]
        ys, xs = np.mgrid[
            max(0, cy - radius):min(h, cy + radius + 1),
            max(0, cx - radius):min(w, cx + radius + 1),
        ]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        frame[ys[mask], xs[mask]] = color

    def _render_frame(self):
        frame = self._build_frame()

        # Draw black box showing the agent's observation window
        x0, y0 = self._obs_window_origin()
        s = self._obs_size
        for row in (y0, y0 + s - 1):
            frame[row, x0:x0 + s] = (0, 0, 0)
        for col in (x0, x0 + s - 1):
            frame[y0:y0 + s, col] = (0, 0, 0)

        return frame

    def render(self):
        return self._render_frame()

    def close(self):
        pass


if __name__ == "__main__":
    import os
    from PIL import Image

    os.makedirs("frames", exist_ok=True)

    env = NavEnv(size=200)
    env.reset()

    # Save frame 0 (initial state)
    frame = env.render()
    Image.fromarray(frame).save("frames/frame_000.png")

    for i in range(1, 51):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        Image.fromarray(frame).save(f"frames/frame_{i:03d}.png")
        if terminated:
            env.reset()

    env.close()
    print("Saved 51 frames to frames/")
