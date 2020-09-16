import numpy as np
import time
import os
import roboverse
import roboverse.bullet as bullet
import argparse
import datetime
from tqdm import tqdm

EPSILON = 0.1


def accept_traj(env_info, env_reward_type):
    if env_reward_type == 'grasping':
        return env_info["grasp_success"]
    elif env_reward_type == 'pick_place':
        return env_info["place_success_target"]
    else:
        assert NotImplementedError  # TODO: will add more


def timestamp(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider, dtd=datetime_divider))


class PickPlace:

    def __init__(self, env, pick_height_thresh=-0.31, non_zero_gripper=False):
        self.env = env
        self.gripper_open_value = 0.7 if non_zero_gripper else 0
        self.gripper_close_value = -0.7 if non_zero_gripper else 0
        self.pick_height_thresh_noisy = pick_height_thresh \
                                            + np.random.normal(scale=0.01)
        self.place_attempted = False
        self.xyz_action_scale = 7.0

        self.dist_thresh = 0.07 + np.random.normal(scale=0.01)

    def get_action(self):
        ee_pos, _ = bullet.get_link_state(env.robot_id, env.end_effector_index)
        object_pos, _ = bullet.get_object_position(env.objects[target_object])
        object_lifted = object_pos[2] > self.pick_height_thresh_noisy
        object_gripper_dist = np.linalg.norm(object_pos - ee_pos)

        container_pos = self.env.container_position
        target_pos = np.append(container_pos[:2], container_pos[2] + 0.15)
        target_pos = target_pos + np.random.normal(scale=0.01)
        gripper_target_dist = np.linalg.norm(target_pos - ee_pos)
        gripper_target_threshold = 0.03

        if self.place_attempted:
            # Avoid pick and place the object again after one attempt
            print('place done')
            action_xyz = [0., 0., 0.]
            action_angles = [0., 0., 0.]
            action_gripper = [self.gripper_open_value]
        elif object_gripper_dist > self.dist_thresh and env.is_gripper_open:
            print('moving')
            # move near the object
            action_xyz = (object_pos - ee_pos) * self.xyz_action_scale
            xy_diff = np.linalg.norm(action_xyz[:2] / self.xyz_action_scale)
            # if xy_diff > 0.03:
            #     action_xyz[2] = 0.0
            action_angles = [0., 0., 0.]
            action_gripper = [0.0]
        elif env.is_gripper_open:
            print('grasping')
            # near the object enough, performs grasping action
            action_xyz = (object_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [-0.7]
        elif not object_lifted:
            print('lifting')
            # lifting objects above the height threshold for picking
            action_xyz = (env.ee_pos_init - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [self.gripper_close_value]
        elif gripper_target_dist > gripper_target_threshold:
            # lifted, now need to move towards the container
            action_xyz = (target_pos - ee_pos) * self.xyz_action_scale
            action_angles = [0., 0., 0.]
            action_gripper = [self.gripper_close_value]
        else:
            # already moved above the container; drop object
            action_xyz = (0., 0., 0.)
            action_angles = [0., 0., 0.]
            action_gripper = [0.7]
            self.place_attempted = True

        return np.concatenate((action_xyz, action_angles, action_gripper))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-trajectories", type=int, required=True)
    parser.add_argument("-t", "--num-timesteps", type=int, required=True)
    parser.add_argument("-e", "--env-name", type=str, required=True)
    parser.add_argument("-a", "--action-dim", type=int, required=True,
                        choices=(4, 7, 8))
    parser.add_argument("-r", "--reward-type", type=str, required=True,
                        choices=('grasping', 'pick_place',))  # TODO: will add more
    parser.add_argument("-k", "--task-name", type=str, required=True)
    parser.add_argument("--gui", action='store_true', default=False)
    parser.add_argument("-o", "--target-object", type=str)
    parser.add_argument("--non-zero-gripper", action='store_true', default=False)
    parser.add_argument("-d", "--save-directory", type=str, default="")
    args = parser.parse_args()

    ACTION_DIM = args.action_dim
    reward_type = args.reward_type

    timestamp = timestamp()
    data_save_path = os.path.join(__file__, "../..", 'data', args.save_directory)
    data_save_path = os.path.abspath(data_save_path)
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)

    env = roboverse.make(args.env_name,
                         gui=args.gui,
                         transpose_image=False)

    img_dim = env.observation_img_dim

    grasping_height_thresh = -0.20
    pick_height_thresh = -0.31

    data = []

    for i in tqdm(range(args.num_trajectories)):
        # obs = env.reset()

        rewards = []

        o = env.reset()
        time.sleep(1)
        images = []
        accept = False
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

        target_object = env.target_object
        if "Two" in args.env_name:
            target_object = args.target_object

        # place_attempted = False
        policy = PickPlace(env)
        for j in range(args.num_timesteps):

            action = policy.get_action()

            if ACTION_DIM == 8:
                action = np.append(action, 0)
            # action += np.random.normal(scale=0.1, size=(ACTION_DIM,))
            print(action)
            action[3:6] = 0.0
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            observation = env.get_observation()
            observation["image"] = np.reshape(np.uint8(observation["image"] * 255.), (img_dim, img_dim, 3))
            traj["observations"].append(observation)
            next_state, reward, done, info = env.step(action)
            next_state["image"] = np.reshape(np.uint8(next_state["image"] * 255.), (img_dim, img_dim, 3))
            traj["next_observations"].append(next_state)
            traj["actions"].append(action)
            traj["rewards"].append(reward)
            traj["terminals"].append(done)
            traj["agent_infos"].append(info)
            traj["env_infos"].append(info)

            if accept_traj(info, reward_type) and num_steps < 0:
                num_steps = j

            time.sleep(0.03)
            rewards.append(reward)

        print("success: ", accept_traj(info, reward_type))
        if accept_traj(info, reward_type):
            print("num_timesteps: ", num_steps)
            data.append(traj)

    print("number of successful trajectories: ", len(data))
    path = os.path.join(data_save_path, "scripted_{}_{}_{}.npy".format(args.env_name, args.task_name, timestamp))
    print(path)
    np.save(path, data)