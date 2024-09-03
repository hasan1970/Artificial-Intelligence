
# Blackjack with Q-Learning

This project implements a Blackjack game using Q-Learning, a reinforcement learning algorithm. The goal of this project is to create an AI agent that can learn to play Blackjack by optimizing its strategy through trial and error, improving its performance over time.

## Project Overview

Blackjack is a popular card game where the objective is to have a hand value as close to 21 as possible without exceeding it. This project simulates a Blackjack environment where the AI agent learns to make decisions (hit, stand, etc.) based on its current hand, the dealer's visible card, and the state of the game.

## Features

- **Q-Learning Implementation:** The AI agent uses Q-Learning to update its strategy over multiple episodes of gameplay.
- **State Representation:** The state is represented by the player's hand value, the dealer's visible card, and whether the player has a usable ace.
- **Reward System:** Rewards are given based on the outcome of each game (win, lose, or draw), guiding the AI's learning process.
- **Training and Evaluation:** The model is trained over numerous episodes, and its performance is evaluated by tracking the win rate over time.

## Files in this Project

- `blackjack.py`: The main Python script that contains the implementation of the Blackjack game and the Q-Learning algorithm.
- `play.py`: Script to train the AI and play the Blackjack game against it.
- `README.md`: This document, providing an overview of the project.

## How It Works

1. **Initialize Q-Table:** The Q-Table is initialized with zeros, representing the expected reward for each action in each state.
2. **Gameplay Loop:** The AI agent plays a large number of games, selecting actions based on its current policy (initially random) and updating the Q-Table based on the rewards received.
3. **Policy Update:** Over time, the policy improves as the Q-Table values are updated to reflect the expected rewards for different actions.
4. **Evaluation:** The trained policy is evaluated by observing the AI's performance in unseen games.

## Running the Project

1. Clone this repository:
   ```bash
   git clone https://github.com/hasan1970/Artificial-Intelligence.git
   cd Artificial-Intelligence/Blackjack
   ```
2. Run the play script:
   ```bash
   python play.py
   ```

