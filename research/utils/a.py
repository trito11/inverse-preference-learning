import gym
import d4rl  # D4RL tự động đăng ký môi trường vào Gym

# Tạo môi trường
env = gym.make("hopper-medium-replay-v2")

# Reset để lấy trạng thái ban đầu
state = env.reset()

# Kiểm tra action space và state space
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
env.unwrapped.state = state

# Lấy dữ liệu offline
dataset = env.get_dataset()
print(f"Keys in dataset: {dataset.keys()}")

import numpy as np

# Load .npz file
data = np.load("file.npz")

# Liệt kê các mảng trong file
print("Keys:", data.files)

# Đọc vài dòng đầu của từng mảng
for key in data.files:
    print(f"\n{key}:")
    print(data[key][:5])  # Đọc 5 dòng đầu tiên
