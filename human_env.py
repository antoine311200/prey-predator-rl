import gymnasium as gym
import keyboard

env = gym.make("predator_prey.envs:PredatorPrey-v0")
env.action_space.seed(42)

observation, info = env.reset(seed=42)
action = 0
while True:
    # To stay in place if no key is pressed
    if action == 0:
        action = 1
    else:
        action = 0
    if keyboard.is_pressed("up"):
        action = 0
    elif keyboard.is_pressed("down"):
        action = 1
    elif keyboard.is_pressed("left"):
        action = 2
    elif keyboard.is_pressed("right"):
        action = 3
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
    env.render()
    # Check if the env should be closed
    if keyboard.is_pressed("esc"):
        break

env.close()
