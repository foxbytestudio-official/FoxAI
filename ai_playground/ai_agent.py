import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, config):
        self.env = env
        self.size = env.size
        self.actions = [0,1,2,3]  # up, down, left, right
        self.q_table = {}  # {(state): [q for each action]}
        self.alpha = config.get("alpha", 0.1)
        self.gamma = config.get("gamma", 0.9)
        self.epsilon = config.get("epsilon", 0.2)
        self.train_steps = 0
        self.episode_rewards = []
        self.last_question = ""
        self.last_action = None

    def get_qs(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0 for _ in self.actions]
        return self.q_table[state]

    def select_action(self, state):
        qs = self.get_qs(state)
        max_q = max(qs)
        min_q = min(qs)
        # If all Qs are similar (low confidence), ask for advice
        if max_q - min_q < 0.2:
            self.last_question = f"I'm at {state}. My options are: up, down, left, right. What should I do?"
            self.last_action = None
            return None  # Will request advice
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
            self.last_question = ""
            self.last_action = action
            return action
        action = int(np.argmax(qs))
        self.last_question = ""
        self.last_action = action
        return action

    def train_episode(self, max_steps=100, user_feedback=None):
        state = self.env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = self.select_action(state)
            # If the agent is asking for help and user_feedback is provided:
            if action is None and user_feedback is not None:
                # Map feedback to action
                action_map = {"up": 0, "down": 1, "left": 2, "right": 3}
                action = action_map.get(user_feedback.lower(), random.choice(self.actions))
                self.last_action = action
                # Optional: Give a bonus for following user advice
                qs = self.get_qs(state)
                qs[action] += self.alpha * (10 - qs[action])
            next_state, reward, done = self.env.step(action)
            qs = self.get_qs(state)
            next_qs = self.get_qs(next_state)
            # Q-learning update
            qs[action] += self.alpha * (reward + self.gamma * max(next_qs) - qs[action])
            state = next_state
            total_reward += reward
            if done:
                break
        self.train_steps += 1
        self.episode_rewards.append(total_reward)
        return total_reward

    def get_stats(self):
        return {
            "episodes": self.train_steps,
            "last_reward": self.episode_rewards[-1] if self.episode_rewards else 0,
            "avg_reward": np.mean(self.episode_rewards[-50:]) if self.episode_rewards else 0
        }

    def get_message(self):
        return self.last_question

    def incorporate_feedback(self, state, user_feedback):
        # Use user feedback to update Q-table at current state
        action_map = {"up": 0, "down": 1, "left": 2, "right": 3}
        action = action_map.get(user_feedback.lower())
        if action is not None:
            qs = self.get_qs(state)
            qs[action] += self.alpha * (10 - qs[action])  # Boost Q-value for suggested action
            self.last_action = action
