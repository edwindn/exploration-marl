import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from dreamer import Dreamer
from utils import loadConfig, seedEverything
from environment.env import NavEnv

# Configuration
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENV_CONFIG_PATH = "../environment/env_config.yaml"
DREAMER_CONFIG_PATH = "dreamer_config.yaml"

# Training hyperparameters
NUM_SEED_EPISODES = 5
NUM_TRAINING_ITERATIONS = 1000
EPISODES_PER_ITERATION = 1
GRADIENT_STEPS_PER_ITERATION = 100

def main():
    # Seed for reproducibility
    seedEverything(SEED)

    # Load configurations
    dreamerConfig = loadConfig(DREAMER_CONFIG_PATH)

    # Create environment
    env = NavEnv(config_path=ENV_CONFIG_PATH)

    # Get environment properties
    observationShape = env.observation_space.shape
    actionSize = env.action_space.n if hasattr(env.action_space, 'n') else env.action_space.shape[0]

    # For discrete action spaces, we need to convert to continuous
    if hasattr(env.action_space, 'n'):
        actionLow = [0.0] * actionSize
        actionHigh = [1.0] * actionSize
    else:
        actionLow = env.action_space.low.tolist()
        actionHigh = env.action_space.high.tolist()

    print(f"Observation shape: {observationShape}")
    print(f"Action size: {actionSize}")
    print(f"Device: {DEVICE}")

    # Create Dreamer agent
    agent = Dreamer(
        observationShape=observationShape,
        actionSize=actionSize,
        actionLow=actionLow,
        actionHigh=actionHigh,
        device=DEVICE,
        config=dreamerConfig
    )

    print("\n=== Starting seed episodes ===")
    # Seed episodes to populate replay buffer
    avgScore = agent.environmentInteraction(env, NUM_SEED_EPISODES, seed=SEED)
    print(f"Seed episodes average score: {avgScore:.2f}")

    print("\n=== Starting training ===")
    # Training loop
    for iteration in range(NUM_TRAINING_ITERATIONS):
        # Collect experience
        avgScore = agent.environmentInteraction(env, EPISODES_PER_ITERATION, seed=SEED)

        # Train on collected experience
        for _ in range(GRADIENT_STEPS_PER_ITERATION):
            # Sample batch from replay buffer
            data = agent.buffer.sample(dreamerConfig.batchSize, dreamerConfig.batchLength)

            # Train world model
            fullStates, worldModelMetrics = agent.worldModelTraining(data)

            # Train behavior (actor and critic)
            behaviorMetrics = agent.behaviorTraining(fullStates)

            agent.totalGradientSteps += 1

        # Logging
        if (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}/{NUM_TRAINING_ITERATIONS}")
            print(f"  Avg Score: {avgScore:.2f}")
            print(f"  Episodes: {agent.totalEpisodes}")
            print(f"  Env Steps: {agent.totalEnvSteps}")
            print(f"  Gradient Steps: {agent.totalGradientSteps}")

    print("\n=== Training complete ===")
    env.close()

if __name__ == "__main__":
    main()
