import torch
import numpy as np
import matplotlib.pyplot as plt
feature_map = ["enemy00", "enemy01", "enemy02", "enemy03", "enemy04", "ally01", "ally02",
               "ally03", "ally04", "own"]
enemy_ally_dim_map = ["distance", "relative x", "relative y", "health", "shield", "unit_type"]
own_dim_map = ["health", "shield", "unit_type"]

action_number = 0

action_targe = action_number -6

def draw_shutter(data):
    number = len(data)
    plt.bar(np.arange(number), np.array(data)+0.05)
    plt.show()


if __name__ == "__main__":
    datas = torch.load("./results/replays/trajectory.pkl")

    for data in datas:
        store_input_grads = data['store_input_grad']
        store_obs = data['store_obs']
        store_action = data['store_action']
        for grad, obs, action in zip(store_input_grads, store_obs, store_action):
            for id in range(5):
                grad_tmp = torch.squeeze(grad[:,id,4:78]).detach().cpu().numpy()
                obs_tmp = torch.squeeze(obs[:,id,4:78]).detach().cpu().numpy()
                action_tmp = action.detach().cpu().numpy()

                grad_list = []
                for index in range(9):
                    grad_list.append(np.mean(grad_tmp[index*8:index*8+2]))
                    grad_list.append(grad_tmp[index*8+2])
                    grad_list.append(grad_tmp[index*8+3])
                    grad_list.append(np.mean(grad_tmp[index * 8+4:index * 8 + 6]))
                    grad_list.append(grad_tmp[index * 8 + 6])
                    grad_list.append(grad_tmp[index * 8 + 7])
                grad_list.append(np.mean(grad_tmp[70:72]))
                grad_list.append(grad_tmp[72])
                grad_list.append(grad_tmp[73])
                draw_shutter(grad_list)
                exit()
