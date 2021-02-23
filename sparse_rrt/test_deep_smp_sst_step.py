import sys
sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/deps/sparse_rrt-1')
sys.path.append('/media/arclabdl1/HD1/Linjun/mpc-mpnet-py')

from sparse_rrt import _deep_smp_module
from sparse_rrt.systems import standard_cpp_systems

import pickle
from mpnet.sst_envs.utils import load_data

import numpy as np
import time


model = "acrobot_obs"
env_id = 0
traj_id = 10
filepath = "/media/arclabdl1/HD1/Linjun/data/kinodynamic/{model}/{filetype}_{env_id}.pkl".format(model=model, env_id=env_id, filetype="obs")

data = load_data(model, env_id, traj_id)
path = data['path']
# print(path)
obs_list = pickle.load(open(filepath, "rb")).reshape(-1, 2)
width = 6

env_vox = np.load("/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/sst_envs/acrobot_obs_env_vox.npy")
obc = env_vox[env_id, 0]
planner = _deep_smp_module.DSSTMPCWrapper(
            start_state=np.array(path[0]),
            goal_state=np.array(path[-1]),
            goal_radius=10,
            random_seed=0,
            sst_delta_near=10,
            sst_delta_drain=1e-1,
            obs_list=obs_list,
            width=width,
            verbose=False,
            mpnet_weight_path="/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet//exported/output/mpnet5000.pt", 
            cost_predictor_weight_path="/media/arclabdl1/HD1/Linjun/mpc-mpnet-py/mpnet/exported/output/costnet5000.pt", 
            num_sample=3,
            ns=32, nt=5, ne=4, max_it=10,
            converge_r=0.1, mu_u=0, std_u=4, mu_t=0.1, 
            std_t=0.5, t_max=0.5, step_size=0.75, integration_step=2e-2)
min_time_steps, max_time_steps = 10, 50
number_of_iterations = 1000
integration_step = 2e-2

start_time = time.time()



for iteration in range(number_of_iterations):
    # planner.step(min_time_steps, max_time_steps, integration_step)
    planner.neural_step(obc.reshape(-1))
#         planner.neural_step(obc.reshape(-1))
    solution = planner.get_solution()
    if iteration % 100 == 0:
        if solution is None:
            solution_cost = None
        else:
            print(solution[0])
            solution_cost = np.sum(solution[2])

        print("Time: %.2fs, Iterations: %d, Nodes: %d, Solution Quality: %s" %
                (time.time() - start_time, iteration, planner.get_number_of_nodes(), solution_cost))
