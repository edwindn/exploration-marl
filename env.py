import gymnasium as gym
import numpy as np
from PIL import Image as PILImage
import yaml


class NavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self,
                 render_mode=None,
                 config_path="env_config.yaml",
                 max_episode_length=100,
        ):
        # Load configuration from YAML
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        env_config = config['env']
        action_type = env_config['action_type']
        size = env_config['size']
        terminal_radius = env_config['terminal_radius']
        reward_type = env_config['reward_type']
        background = env_config.get('background', True)

        assert action_type in ["discrete", "continuous"]
        assert reward_type in ["dense", "sparse"]
        self.size = size
        self.render_mode = render_mode
        self.action_type = action_type
        self.terminal_radius = terminal_radius
        self.reward_type = reward_type
        self.max_episode_length = max_episode_length
        self.step_count = 0

        if background:
            bg = PILImage.open("background.png").convert("RGB").resize((size, size))
            self._background = np.array(bg, dtype=np.uint8)  # shape (size, size, 3)
        else:
            # White background
            self._background = np.full((size, size, 3), 255, dtype=np.uint8)

        self._obs_size = 50  # cropped observation window side length

        # Observation: 30x30 crop of the frame centred on the agent
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
            delta = {0: (0, 0), 1: (0, -10), 2: (0, 10), 3: (-10, 0), 4: (10, 0)}[action]
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
        truncated = self.step_count >= self.max_episode_length

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
        """Return (x0, y0) top-left of the 50x50 observation window centred on the agent."""
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
