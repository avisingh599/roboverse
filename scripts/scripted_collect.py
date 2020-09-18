import numpy as np
import time
import os
import os.path as osp
import roboverse
from roboverse.envs.env_policy_list import instantiate_policy_class
import argparse
from tqdm import tqdm

from roboverse.utils import get_timestamp
EPSILON = 0.1

# TODO(avi): Clean this up
NFS_PATH = '/nfs/kun1/users/avi/imitation_datasets/'


def add_transition(traj, observation, action, reward, info, agent_info, done,
                   next_observation, img_dim):
    observation["image"] = np.reshape(np.uint8(observation["image"] * 255.),
                                      (img_dim, img_dim, 3))
    traj["observations"].append(observation)
    next_observation["image"] = np.reshape(
        np.uint8(next_observation["image"] * 255.), (img_dim, img_dim, 3))
    traj["next_observations"].append(next_observation)
    traj["actions"].append(action)
    traj["rewards"].append(reward)
    traj["terminals"].append(done)
    traj["agent_infos"].append(agent_info)
    traj["env_infos"].append(info)
    return traj


def main(args):

    timestamp = get_timestamp()
    if osp.exists(NFS_PATH):
        data_save_path = osp.join(NFS_PATH, args.save_directory)
    else:
        data_save_path = osp.join(__file__, "../..", "data", args.save_directory)
    data_save_path = osp.abspath(data_save_path)
    if not osp.exists(data_save_path):
        os.makedirs(data_save_path)

    env = roboverse.make(args.env_name,
                         gui=args.gui,
                         transpose_image=False)
    img_dim = env.observation_img_dim

    env_action_dim = env.action_space.shape[0]

    data = []
    policy = instantiate_policy_class(args.env_name, env)
    num_success = 0

    for i in tqdm(range(args.num_trajectories)):

        rewards = []

        env.reset()
        policy.reset()
        time.sleep(1)
        traj = dict(
            observations=[],
            actions=[],
            rewards=[],
            next_observations=[],
            terminals=[],
            agent_infos=[],
            env_infos=[],
        )
        num_steps = -1
        for j in range(args.num_timesteps):

            action, agent_info = policy.get_action()

            # In case we need to pad actions by 1 for easier realNVP modelling
            if env_action_dim - action.shape[0] == 1:
                action = np.append(action, 0)
            action += np.random.normal(scale=args.noise, size=(env_action_dim,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            observation = env.get_observation()
            next_observation, reward, done, info = env.step(action)
            add_transition(traj, observation,  action, reward, info, agent_info,
                           done, next_observation, img_dim)
            print("reward", reward)
            if reward and num_steps < 0:
                num_steps = j

            rewards.append(reward)
            if done:
                break

        if rewards[-1] > 0.:
            if args.gui:
                print("num_timesteps: ", num_steps)
            data.append(traj)
            num_success += 1

        if args.gui:
            print("success rate: {}".format(num_success/(i+1)))

    print("number of successful trajectories: ", len(data))
    path = osp.join(data_save_path, "scripted_{}_{}.npy".format(
        args.env_name, timestamp))
    print(path)
    np.save(path, data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("--gui", action='store_true', default=False)
    parser.add_argument("-o", "--target-object", type=str)
    parser.add_argument("-d", "--save-directory", type=str, default="")
    parser.add_argument("--noise", type=float, default=0.1)
    args = parser.parse_args()

    main(args)
