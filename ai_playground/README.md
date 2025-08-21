# FoxAI RL Playground

This directory contains an interactive RL agent and dashboard for human-AI two-way communication.

## Files

- `ai_agent.py`: Q-learning agent that can ask for user help when uncertain.
- `environment.py`: Simple grid world environment with obstacles.
- `dashboard.py`: Streamlit dashboard for training, chatting, and visualization.
- `config.yaml`: Environment and agent configuration.

## Usage

1. Install requirements:
   ```
   pip install streamlit matplotlib pyyaml
   ```

2. Run the dashboard:
   ```
   streamlit run ai_playground/dashboard.py
   ```

3. Use the controls to train the AI. When the agent is uncertain, it asks for advice. You can reply in the chat box, and your advice will help the agent learn.

## Communication Loop

- The agent will ask questions when unsure what to do.
- Your answers directly shape its learning and policy.

## Customization

- Change `config.yaml` to adjust the environment or agent hyperparameters.
- Add more obstacles, change the target, or tweak learning rates.

---
