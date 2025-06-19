# %%
import numpy as np

num_agents = 8
num_instances = 50
np.random.seed(0)
index_set = np.zeros((num_instances, num_agents), dtype=int)
for i in range(num_instances):
    index_set[i, :] = np.random.permutation(num_agents)
    # # check if there is any repeated permutation
    # if i > 0:
    #     for j in range(i):
    #         if np.all(index_set[i, :] == index_set[j, :]):
    #             print('Repeated permutation found')
    #             print(index_set[i, :])
    #             print(index_set[j, :])
    #             break
# save index_set
np.save('index_set.npy', index_set)
