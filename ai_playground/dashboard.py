import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import yaml

from environment import SimpleGameEnv
from ai_agent import QLearningAgent

st.title("Autonomous AI Playground with Communication")

with open("ai_playground/config.yaml", "r") as f:
    config = yaml.safe_load(f)

if "env" not in st.session_state or "agent" not in st.session_state:
    st.session_state.env = SimpleGameEnv(config)
    st.session_state.agent = QLearningAgent(st.session_state.env, config)
    st.session_state.chat_history = []

env = st.session_state.env
agent = st.session_state.agent

st.subheader("Training Controls")
train_episodes = st.number_input("Episodes to train", min_value=1, max_value=1000, value=10)
user_feedback = None

if st.button("Train"):
    for _ in range(train_episodes):
        state = env.get_state()
        # Try to get feedback from chat history if available
        last_feedback = None
        if st.session_state.chat_history:
            # Find last user feedback message
            for sender, msg in reversed(st.session_state.chat_history):
                if sender == "User":
                    last_feedback = msg
                    break
        reward = agent.train_episode(max_steps=100, user_feedback=last_feedback)
        msg = agent.get_message()
        if msg:
            st.session_state.chat_history.append(("AI", msg))
            st.info(msg)
            user_feedback = st.text_input("Your answer (up/down/left/right):", key=f"feedback_{agent.train_steps}")
            if user_feedback:
                agent.incorporate_feedback(state, user_feedback)
                st.session_state.chat_history.append(("User", user_feedback))
                st.success(f"Your advice '{user_feedback}' was sent to the AI.")

    st.success(f"Trained for {train_episodes} episodes.")

if st.button("Reset Agent & Env"):
    st.session_state.env = SimpleGameEnv(config)
    st.session_state.agent = QLearningAgent(st.session_state.env, config)
    st.session_state.chat_history = []
    st.success("Environment and agent reset.")

st.subheader("Agent Statistics")
stats = agent.get_stats()
st.write(f"Total Episodes: {stats['episodes']}")
st.write(f"Last Episode Reward: {stats['last_reward']}")
st.write(f"Average Reward (last 50): {stats['avg_reward']:.2f}")

if agent.episode_rewards:
    fig, ax = plt.subplots()
    ax.plot(agent.episode_rewards, label='Episode Reward')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.legend()
    st.pyplot(fig)

st.subheader("Chat with the AI")
for sender, msg in st.session_state.chat_history:
    if sender == "AI":
        st.markdown(f"<span style='color:blue'><b>AI:</b> {msg}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:green'><b>You:</b> {msg}</span>", unsafe_allow_html=True)

st.subheader("Environment Visualization")
state = env.get_state()
target = env.target_pos
obstacles = env.obstacles

fig, ax = plt.subplots(figsize=(5,5))
ax.set_xlim(-0.5, env.size-0.5)
ax.set_ylim(-0.5, env.size-0.5)
ax.grid(True)
ax.add_patch(plt.Circle((state[0], state[1]), 0.3, color="blue", label="Agent"))
ax.add_patch(plt.Circle((target[0], target[1]), 0.3, color="green", label="Target"))
for obs in obstacles:
    ax.add_patch(plt.Circle((obs[0], obs[1]), 0.3, color="red"))
ax.set_aspect('equal')
st.pyplot(fig)