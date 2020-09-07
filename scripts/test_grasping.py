import numpy as np
import time

from roboverse.envs.widow250 import Widow250Env
import roboverse.bullet as bullet

EPSILON = 0.1

if __name__ == "__main__":
    env = Widow250Env(gui=True,
                      control_mode='discrete_gripper',
                      target_object='beer_bottle')
    height_thresh = -0.20

    scripted_traj_len = 30

    for _ in range(50):
        obs = env.reset()
        # object_pos[2] = -0.30

        dist_thresh = 0.06 + np.random.normal(scale=0.01)
        rewards = []

        for j in range(scripted_traj_len):

            ee_pos, _ = bullet.get_link_state(env.robot_id, env.end_effector_index)
            object_pos, _ = bullet.get_object_position(env.objects[env.target_object])

            object_lifted = object_pos[2] > height_thresh

            object_gripper_dist = np.linalg.norm(object_pos - ee_pos)
            theta_action = 0.
            # theta_action = np.random.uniform()
            # print(object_gripper_dist)
            if object_gripper_dist > dist_thresh and env.is_gripper_open:
                print('approaching', object_gripper_dist)
                action = (object_pos - ee_pos) * 5.0
                xy_diff = np.linalg.norm(action[:2]/7.0)
                if xy_diff > 0.02:
                    action[2] = 0.0
                action = np.concatenate((action, np.asarray([0, 0., 0., 0.])))
            elif env.is_gripper_open:
                print('gripper closing')
                action = (object_pos - ee_pos) * 7.0
                action = np.concatenate((action, np.asarray([0, 0., 0., -0.7])))

            elif not object_lifted:
                print('raise object upward')
                action = np.asarray([0., 0., 0.7])
                action = np.concatenate(
                    (action, np.asarray([0., 0., 0., 0.])))
            else:
                tray_pos, _ = bullet.get_object_position(env.tray_id)
                action = (tray_pos - ee_pos)[:2]
                action = np.concatenate(
                    (action, np.asarray([0.1, 0., 0., 0., 0.])))
                # action = np.zeros((6,))

            action[:3] += np.random.normal(scale=0.05, size=(3,))
            action = np.clip(action, -1 + EPSILON, 1 - EPSILON)
            obs, rew, done, info = env.step(action)
            time.sleep(0.1)
            print(rew)
            rewards.append(rew)