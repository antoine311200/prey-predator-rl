import gymnasium as gym
import numpy as np
import pygame


class PredatorPreyEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str = "human"):
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # pygame setup
        self.screen_size = (1280, 720)
        self.window = None
        self.clock = None
        self.dt = 60 / 1000

        # Predator config:
        self.predator_size = 40
        self.prey_size = 40

    def _get_obs(self):
        return np.array(
            [
                self.predator_pos.x / self.screen_size[0],
                self.predator_pos.y / self.screen_size[1],
            ],
            dtype=np.float32,
        )

    def _get_info(self):
        return {}

    def check_collision(self):
        if (
            self.predator_pos.distance_to(self.prey_pos)
            < self.predator_size + self.prey_size
        ):
            return True
        return False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Predator physics
        self.predator_pos = pygame.Vector2(
            self.screen_size[0] / 2, self.screen_size[1] / 2
        )
        # Prey physics
        self.prey_pos = pygame.Vector2(
            (0.75) * self.screen_size[0], (0.75) * self.screen_size[1]
        )

        if self.render_mode == "human":
            self._render_frame()

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def _update_pos(self, action):
        """Update the predator's position based on the action."""
        if action == 0:
            self.predator_pos.y -= 300 * self.dt
        elif action == 1:
            self.predator_pos.y += 300 * self.dt
        elif action == 2:
            self.predator_pos.x -= 300 * self.dt
        elif action == 3:
            self.predator_pos.x += 300 * self.dt

    def step(self, action):
        self._update_pos(action)
        self._check_oob()
        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self._render_frame()
        # If the preydator touches the prey, the game is over
        if self.check_collision():
            return observation, 1, True, False, info

        return observation, 0, False, False, info

    def _check_oob(self):
        """Check if the predator is out of bounds and wrap it around if it is."""
        if self.predator_pos.x < 0 or self.predator_pos.x > self.screen_size[0]:
            self.predator_pos.x = self.predator_pos.x % self.screen_size[0]
        if self.predator_pos.y < 0 or self.predator_pos.y > self.screen_size[1]:
            self.predator_pos.y = self.predator_pos.y % self.screen_size[1]

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.screen_size)
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface(self.screen_size)
        canvas.fill((255, 255, 255))

        pygame.draw.circle(canvas, "red", self.predator_pos, self.predator_size)
        pygame.draw.circle(canvas, "green", self.prey_pos, self.prey_size)

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


class PredatorPreyContinuousEnv(PredatorPreyEnv):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode: str = "human"):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(2,), dtype=np.float32
        )
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # pygame setup
        self.screen_size = (1280, 720)
        self.window = None
        self.clock = None
        self.dt = 60 / 1000

        # Predator config:
        self.predator_size = 40
        self.prey_size = 40

    def _update_pos(self, action):
        """Update the predator's position based on the action."""
        self.predator_pos.x += 300 * action[0] * self.dt
        self.predator_pos.y += 300 * action[1] * self.dt
