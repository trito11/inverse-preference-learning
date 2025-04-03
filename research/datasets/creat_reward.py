
import gym
import d4rl
import os
env = gym.make("hopper-medium-replay-v2")
dataset = d4rl.qlearning_dataset(env)
print(dataset.keys())  # ['observations', 'actions', 'rewards', 'terminals', 'next_observations']

# Example: Check the shape of observations
print(dataset["observations"].shape) 

import h5py
import os
import pickle
import numpy as np


with open(os.path.join("/home/tri/offline_rl/PreferenceTransformer/human_label/hopper-medium-replay-v2/indices_num500"), "rb") as fp:   # Unpickling
    human_indices1 = pickle.load(fp)
    # print(human_indices1)


with open(os.path.join("/home/tri/offline_rl/PreferenceTransformer/human_label/hopper-medium-replay-v2/indices_2_num500"), "rb") as fp:   # Unpickling
    human_indices2 = pickle.load(fp)
# from datasets import load_dataset

# ds = load_dataset("edbeeching/decision_transformer_gym_replay", "halfcheetah-medium-v2")
# print(ds.keys())


input_file = "/home/tri/offline_rl/inverse-preference-learning/datasets/preference_transformer/hopper-medium-replay-v2/num500_human_train.npz"
output_file = input_file.replace(".npz", "_reward.npz")
# Load dữ liệu gốc
data = np.load(output_file)
obs1=data["obs_1"][1][99]
obs1=np.array(obs1)

for i in range(500):
    print(data["label"][i],data["sum_reward_2"][i]/data["sum_reward_1"][i], data["sum_reward_1"][i])

# reward_1 = np.zeros((data["obs_1"].shape[:2]))
# reward_2 = np.zeros((data["obs_1"].shape[:2]))
# sum_reward_1 = np.zeros((data["label"].shape))
# sum_reward_2 = np.zeros((data["label"].shape))
# for k in range(data["obs_1"].shape[0]):  # Lặp qua tất cả các sequence
#     sum1, sum2 = 0, 0
#     idx1, idx2 = human_indices1[k], human_indices2[k]

#     print(k)
#     for i in range(data["obs_1"].shape[1]):
#         reward1 = dataset["rewards"][idx1+i]*1000/(3192.925002515316+1.4400692265480757)
#         reward2 = dataset["rewards"][idx2+i]*1000/(3192.925002515316+1.4400692265480757)



#         sum1 += reward1
#         sum2 += reward2
#         reward_1[k][i] = reward1
#         reward_2[k][i] = reward2


#     sum_reward_1[k] = sum1
#     sum_reward_2[k] = sum2

# # Chuyển reward mới thành numpy arrays


# # Lưu tất cả dữ liệu (bao gồm reward mới) vào file mới
# np.savez(output_file, **data, reward_1=reward_1, reward_2=reward_2, sum_reward_1=sum_reward_1, sum_reward_2=sum_reward_2)

