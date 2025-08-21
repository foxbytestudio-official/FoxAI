import numpy as np
import random

class SimpleGameEnv:
    def __init__(self, config):
        self.size = config.get("size", 5)
        self.target_pos = tuple(config.get("target_pos", [self.size-1, self.size-1]))
        self.obstacles = [tuple(obs) for obs in config.get("obstacles", [[2,2]])]
        self.reset()

    def reset(self):
        self.agent_pos = (0, 0)
        return self.get_state()

    def get_state(self):
        return self.agent_pos

    def step(self, action):
        x, y = self.agent_pos
        if action == 0:    # up
            y = max(0, y-1)
        elif action == 1:  # down
            y = min(self.size-1, y+1)
        elif action == 2:  # left
            x = max(0, x-1)
        elif action == 3:  # right
            x = min(self.size-1, x+1)
        next_pos = (x, y)
        reward = -1
        done = False
        if next_pos in self.obstacles:
            reward = -5
        elif next_pos == self.target_pos:
            reward = 50
            done = True
        self.agent_pos = next_pos
        return self.agent_pos, reward, done
