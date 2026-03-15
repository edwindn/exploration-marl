import gymnasium as gym
import numpy as np
import pygame
from PIL import Image as PILImage


class NavEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 render_mode=None,
                 action_type="discrete",
                 size=200,
                 reward_radius=5,         
        ):
        assert action_type in ("discrete", "continuous"), "action_type must be 'discrete' or 'continuous'"
        self.size = size
        self.render_mode = render_mode
        self.action_type = action_type
        self.reward_radius = reward_radius

        bg = PILImage.open("background.png").convert("RGB").resize((size, size))
        self._background = np.array(bg, dtype=np.uint8)  # shape (size, size, 3)

        self._obs_size = 50  # cropped observation window side length

        # Observation: 30x30 crop of the frame centred on the agent
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self._obs_size, self._obs_size, 3), dtype=np.uint8
        )
        if action_type == "discrete":
            # Actions: 0=up, 1=down, 2=left, 3=right
            self.action_space = gym.spaces.Discrete(4)
        else:
            # Actions: [angle (0-360 degrees clockwise from right), radius (0-10)]
            self.action_space = gym.spaces.Box(
                low=np.array([0.0, 0.0], dtype=np.float32),
                high=np.array([360.0, 10.0], dtype=np.float32),
            )

        self._goal_pos = np.array([size * 0.8, size * 0.2], dtype=np.float32)
        self._agent_pos = None

        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        half = self._obs_size // 2
        self._agent_pos = self.np_random.integers(half, self.size - half, size=2).astype(np.float32)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        half = self._obs_size // 2
        lo, hi = half, self.size - 1 - half
        if self.action_type == "discrete":
            delta = {0: (0, -10), 1: (0, 10), 2: (-10, 0), 3: (10, 0)}[action]
            self._agent_pos = np.clip(
                self._agent_pos + np.array(delta, dtype=np.float32),
                lo, hi
            )
        else:
            angle_deg, radius = float(action[0]), float(action[1])
            # Clockwise from right: x increases rightward, y increases downward in screen coords
            angle_rad = np.deg2rad(angle_deg)
            dx = radius * np.cos(angle_rad)
            dy = radius * np.sin(angle_rad)  # positive angle -> downward in screen space
            self._agent_pos = np.clip(
                self._agent_pos + np.array([dx, dy], dtype=np.float32),
                lo, hi
            )

        reward = self._get_reward()
        terminated = reward == 1.0

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, False, {}

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
        return 1 if np.sqrt(dist[0]**2 + dist[1]**2) <= self.reward_radius else 0

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
        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.size, self.size))
            self.clock = pygame.time.Clock()

        frame = self._build_frame()

        # Draw black box showing the agent's observation window
        x0, y0 = self._obs_window_origin()
        s = self._obs_size
        for row in (y0, y0 + s - 1):
            frame[row, x0:x0 + s] = (0, 0, 0)
        for col in (x0, x0 + s - 1):
            frame[y0:y0 + s, col] = (0, 0, 0)

        # pygame surfarray expects (width, height, 3), numpy frame is (height, width, 3)
        surface = pygame.surfarray.make_surface(frame.transpose(1, 0, 2))

        if self.render_mode == "human":
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return frame

    def render(self):
        return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None


if __name__ == "__main__":
    import os
    from PIL import Image

    os.makedirs("frames", exist_ok=True)

    pygame.init()
    env = NavEnv(size=200, render_mode="rgb_array")
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
    pygame.quit()
    print("Saved 51 frames to frames/")
