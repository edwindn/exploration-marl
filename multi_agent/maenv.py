import gymnasium as gym
import numpy as np
from PIL import Image as PILImage
import yaml
from pathlib import Path


class GridEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=None, config_path=None):
        # Load configuration from YAML
        if config_path is None:
            config_path = Path(__file__).parent / "maenv_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        env_config = config['env']
        grid_size = env_config['grid_size']  # Number of grid cells (e.g., 10x10)
        cell_size = env_config['cell_size']  # Size of each cell in pixels
        obs_size = env_config['obs_size']    # Number of grid cells visible around agent
        obs_pixels = env_config['obs_pixels']  # Downsampled observation size in pixels
        wall_fraction = env_config['wall_fraction']  # Fraction of edges to be walls

        self.grid_size = grid_size
        self.cell_size = cell_size
        self.obs_size = obs_size
        self.obs_pixels = obs_pixels
        self.wall_fraction = wall_fraction
        self.render_mode = render_mode

        # Total environment size in pixels
        self.pixel_size = grid_size * cell_size

        # Wall storage: sets of (x, y, direction) where direction is 'h' or 'v'
        # 'h' means horizontal edge below cell (x, y)
        # 'v' means vertical edge to the right of cell (x, y)
        self._walls = set()

        # Goal position (grid coordinates)
        self._goal_pos = None

        # Observation space: downsampled to obs_pixels x obs_pixels
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(obs_pixels, obs_pixels, 3), dtype=np.uint8
        )

        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_space = gym.spaces.Discrete(5)

        self._agent_grid_pos = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate random walls
        self._generate_walls()

        # Place goal randomly in the middle of the grid
        mid_start = self.grid_size // 4
        mid_end = 3 * self.grid_size // 4
        goal_x = self.np_random.integers(mid_start, mid_end)
        goal_y = self.np_random.integers(mid_start, mid_end)
        self._goal_pos = np.array([goal_x, goal_y], dtype=np.int32)

        # Initialize agent randomly on the edge of the grid
        edge = self.np_random.integers(0, 4)  # 0=top, 1=bottom, 2=left, 3=right

        if edge == 0:  # Top edge
            x = self.np_random.integers(0, self.grid_size)
            y = 0
        elif edge == 1:  # Bottom edge
            x = self.np_random.integers(0, self.grid_size)
            y = self.grid_size - 1
        elif edge == 2:  # Left edge
            x = 0
            y = self.np_random.integers(0, self.grid_size)
        else:  # Right edge
            x = self.grid_size - 1
            y = self.np_random.integers(0, self.grid_size)

        self._agent_grid_pos = np.array([x, y], dtype=np.int32)

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), {}

    def step(self, action):
        # Action deltas: 0=stay, 1=up, 2=down, 3=left, 4=right
        grid_delta = {0: (0, 0), 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action]

        # Calculate new position with boundary checks
        new_x = np.clip(self._agent_grid_pos[0] + grid_delta[0], 0, self.grid_size - 1)
        new_y = np.clip(self._agent_grid_pos[1] + grid_delta[1], 0, self.grid_size - 1)

        # Check if movement crosses a wall
        wall_collision = self._check_wall_collision(
            self._agent_grid_pos[0], self._agent_grid_pos[1],
            new_x, new_y
        )

        if wall_collision:
            # Movement blocked by wall, give -1 reward and don't move
            reward = -1.0
        else:
            # Valid movement
            self._agent_grid_pos = np.array([new_x, new_y], dtype=np.int32)
            reward = self._get_reward()

        terminated = False
        truncated = False

        if self.render_mode == "human":
            self._render_frame()

        info = {}

        return self._get_obs(), reward, terminated, truncated, info

    def rand_act(self):
        return self.action_space.sample()

    def _get_obs(self):
        """Get agent's observation: a local view of the grid centered on the agent."""
        # Create observation canvas with white padding at full resolution
        full_obs_pixels = (2 * self.obs_size + 1) * self.cell_size
        obs = np.full((full_obs_pixels, full_obs_pixels, 3), 255, dtype=np.uint8)

        agent_x, agent_y = self._agent_grid_pos

        # Determine which grid cells are visible
        for dy in range(-self.obs_size, self.obs_size + 1):
            for dx in range(-self.obs_size, self.obs_size + 1):
                grid_x = agent_x + dx
                grid_y = agent_y + dy

                # Check if this grid cell is within bounds
                if 0 <= grid_x < self.grid_size and 0 <= grid_y < self.grid_size:
                    # Calculate position in observation
                    obs_cell_x = dx + self.obs_size
                    obs_cell_y = dy + self.obs_size

                    # Draw grid cell (white with black borders)
                    x0 = obs_cell_x * self.cell_size
                    y0 = obs_cell_y * self.cell_size
                    x1 = x0 + self.cell_size
                    y1 = y0 + self.cell_size

                    # Draw black borders (top and left edges)
                    obs[y0, x0:x1] = 0  # Top border
                    obs[y0:y1, x0] = 0  # Left border

                    # Add right and bottom borders for edge cells
                    if grid_x == self.grid_size - 1:
                        obs[y0:y1, x1 - 1] = 0  # Right border
                    if grid_y == self.grid_size - 1:
                        obs[y1 - 1, x0:x1] = 0  # Bottom border

                    # Draw walls in bright red (3 pixels thick)
                    # Check horizontal wall below this cell
                    if (grid_x, grid_y, 'h') in self._walls:
                        obs[max(0, y1 - 2):y1 + 1, x0:x1] = (255, 0, 0)
                    # Check vertical wall to the right of this cell
                    if (grid_x, grid_y, 'v') in self._walls:
                        obs[y0:y1, max(0, x1 - 2):x1 + 1] = (255, 0, 0)

                    # Draw goal as yellow circle if visible
                    if self._goal_pos is not None and grid_x == self._goal_pos[0] and grid_y == self._goal_pos[1]:
                        goal_px_x = obs_cell_x * self.cell_size + self.cell_size // 2
                        goal_px_y = obs_cell_y * self.cell_size + self.cell_size // 2
                        self._draw_circle(obs, (goal_px_x, goal_px_y), radius=self.cell_size // 3, color=(255, 255, 0))

        # Draw agent as blue dot in center of its cell
        center_cell_x = self.obs_size
        center_cell_y = self.obs_size
        center_px_x = center_cell_x * self.cell_size + self.cell_size // 2
        center_px_y = center_cell_y * self.cell_size + self.cell_size // 2

        self._draw_circle(obs, (center_px_x, center_px_y), radius=3, color=(0, 0, 255))

        # Downsample observation to configured size using PIL
        if full_obs_pixels != self.obs_pixels:
            pil_image = PILImage.fromarray(obs)
            pil_image = pil_image.resize((self.obs_pixels, self.obs_pixels), PILImage.LANCZOS)
            obs = np.array(pil_image)

        return obs

    def _generate_walls(self):
        """Generate random walls as a fraction of interior edges, avoiding loops."""
        self._walls.clear()

        # Count all interior edges (excluding outer border)
        num_horizontal = self.grid_size * (self.grid_size - 1)
        num_vertical = (self.grid_size - 1) * self.grid_size
        total_interior_edges = num_horizontal + num_vertical

        # Number of walls to place
        num_walls = int(total_interior_edges * self.wall_fraction)

        # Generate all possible interior edges
        all_edges = []

        # Horizontal edges (below each cell, excluding bottom border)
        for y in range(self.grid_size - 1):
            for x in range(self.grid_size):
                all_edges.append((x, y, 'h'))

        # Vertical edges (to the right of each cell, excluding right border)
        for y in range(self.grid_size):
            for x in range(self.grid_size - 1):
                all_edges.append((x, y, 'v'))

        # Shuffle edges for random selection
        shuffled_edges = all_edges.copy()
        self.np_random.shuffle(shuffled_edges)

        # Track endpoints of walls
        # An endpoint is a grid vertex (corner of cells)
        # For each vertex, count how many walls connect to it
        vertex_degree = {}  # (vertex_x, vertex_y) -> count

        # Helper to check if a vertex is on the boundary
        def is_boundary_vertex(vx, vy):
            return vx == 0 or vx == self.grid_size or vy == 0 or vy == self.grid_size

        # Helper to get the two vertices of an edge
        def get_edge_vertices(x, y, direction):
            if direction == 'h':
                # Horizontal edge below cell (x, y)
                # Connects vertices (x, y+1) and (x+1, y+1)
                return (x, y + 1), (x + 1, y + 1)
            else:  # 'v'
                # Vertical edge to right of cell (x, y)
                # Connects vertices (x+1, y) and (x+1, y+1)
                return (x + 1, y), (x + 1, y + 1)

        # Try to add walls sequentially, checking for loop formation
        walls_added = 0
        for edge in shuffled_edges:
            if walls_added >= num_walls:
                break

            x, y, direction = edge
            v1, v2 = get_edge_vertices(x, y, direction)

            # Get current degrees
            deg1 = vertex_degree.get(v1, 0)
            deg2 = vertex_degree.get(v2, 0)

            # Check if both vertices already have connections
            # If both endpoints would have degree >= 1, this creates a loop or extends a path
            # We allow:
            # - Adding to two unconnected vertices (creates new segment)
            # - Adding to one connected and one unconnected (extends a path)
            # We reject:
            # - Both vertices already connected AND at least one is on boundary (closes loop to boundary)
            # - Both vertices already connected AND both are interior (creates cycle)

            # Check if adding this wall would create a problematic connection
            v1_is_boundary = is_boundary_vertex(*v1)
            v2_is_boundary = is_boundary_vertex(*v2)

            # Reject if both endpoints are already connected
            if deg1 >= 1 and deg2 >= 1:
                # This would create a loop or connect two existing paths
                continue

            # Reject if one endpoint is on boundary and already has a connection
            if (v1_is_boundary and deg1 >= 1) or (v2_is_boundary and deg2 >= 1):
                continue

            # Add this wall
            self._walls.add(edge)
            vertex_degree[v1] = deg1 + 1
            vertex_degree[v2] = deg2 + 1
            walls_added += 1

    def _check_wall_collision(self, old_x, old_y, new_x, new_y):
        """Check if moving from (old_x, old_y) to (new_x, new_y) crosses a wall."""
        # If no movement, no collision
        if old_x == new_x and old_y == new_y:
            return False

        # Moving right: check vertical wall to the right of old cell
        if new_x > old_x:
            if (old_x, old_y, 'v') in self._walls:
                return True

        # Moving left: check vertical wall to the left of old cell (right of new cell)
        if new_x < old_x:
            if (new_x, new_y, 'v') in self._walls:
                return True

        # Moving down: check horizontal wall below old cell
        if new_y > old_y:
            if (old_x, old_y, 'h') in self._walls:
                return True

        # Moving up: check horizontal wall above old cell (below new cell)
        if new_y < old_y:
            if (new_x, new_y, 'h') in self._walls:
                return True

        return False

    def _get_reward(self):
        """Return +1 if agent is on the goal, 0 otherwise."""
        if np.array_equal(self._agent_grid_pos, self._goal_pos):
            return 1.0
        return 0.0

    def _build_frame(self):
        """Build the full environment frame for rendering."""
        # Create white background
        frame = np.full((self.pixel_size, self.pixel_size, 3), 255, dtype=np.uint8)

        # Draw grid lines (black)
        for i in range(self.grid_size + 1):
            pos = i * self.cell_size
            # Horizontal line
            if pos < self.pixel_size:
                frame[pos, :] = 0
            # Vertical line
            if pos < self.pixel_size:
                frame[:, pos] = 0

        # Draw walls in bright red (3 pixels thick)
        wall_thickness = 3
        for x, y, direction in self._walls:
            if direction == 'h':
                # Horizontal wall below cell (x, y)
                pos_y = (y + 1) * self.cell_size
                pos_x_start = x * self.cell_size
                pos_x_end = (x + 1) * self.cell_size
                if pos_y < self.pixel_size:
                    y_start = max(0, pos_y - wall_thickness // 2)
                    y_end = min(self.pixel_size, pos_y + wall_thickness // 2 + 1)
                    frame[y_start:y_end, pos_x_start:pos_x_end] = (255, 0, 0)
            else:  # 'v'
                # Vertical wall to the right of cell (x, y)
                pos_x = (x + 1) * self.cell_size
                pos_y_start = y * self.cell_size
                pos_y_end = (y + 1) * self.cell_size
                if pos_x < self.pixel_size:
                    x_start = max(0, pos_x - wall_thickness // 2)
                    x_end = min(self.pixel_size, pos_x + wall_thickness // 2 + 1)
                    frame[pos_y_start:pos_y_end, x_start:x_end] = (255, 0, 0)

        # Draw goal as large yellow circle
        if self._goal_pos is not None:
            goal_px_x = self._goal_pos[0] * self.cell_size + self.cell_size // 2
            goal_px_y = self._goal_pos[1] * self.cell_size + self.cell_size // 2
            self._draw_circle(frame, (goal_px_x, goal_px_y), radius=self.cell_size // 3, color=(255, 255, 0))

        # Draw agent as blue dot
        agent_px_x = self._agent_grid_pos[0] * self.cell_size + self.cell_size // 2
        agent_px_y = self._agent_grid_pos[1] * self.cell_size + self.cell_size // 2

        self._draw_circle(frame, (agent_px_x, agent_px_y), radius=3, color=(0, 0, 255))

        return frame

    def _draw_circle(self, frame, pos, radius, color):
        """Draw a filled circle on the frame."""
        cx, cy = int(round(pos[0])), int(round(pos[1]))
        h, w = frame.shape[:2]
        ys, xs = np.mgrid[
            max(0, cy - radius):min(h, cy + radius + 1),
            max(0, cx - radius):min(w, cx + radius + 1),
        ]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius ** 2
        frame[ys[mask], xs[mask]] = color

    def _render_frame(self):
        """Render the full environment frame."""
        return self._build_frame()

    def render(self):
        return self._render_frame()

    def close(self):
        pass


if __name__ == "__main__":
    import os
    from PIL import Image

    os.makedirs("frames", exist_ok=True)

    env = GridEnv()
    env.reset()

    # Save frame 0 (initial state)
    frame = env.render()
    Image.fromarray(frame).save("frames/frame_000.png")

    for i in range(1, 51):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        frame = env.render()
        Image.fromarray(frame).save(f"frames/frame_{i:03d}.png")

        # Also save agent's observation
        Image.fromarray(obs).save(f"frames/obs_{i:03d}.png")

        if terminated:
            env.reset()

    env.close()
    print("Saved 51 frames to frames/")
