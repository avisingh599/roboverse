import roboverse
from roboverse.policies import policies
from scripted_collect import collect_one_traj
import skvideo.io
import cv2
import os
import argparse
import numpy as np

ROBOT_VIEW_HEIGHT = 100
ROBOT_VIEW_WIDTH = 100
ROBOT_VIEW_CROP_X = 30


class BulletVideoLogger:
    def __init__(self, env_name, scripted_policy_name,
                 num_timesteps_per_traj, accept_trajectory_key,
                 video_save_dir, success_only,
                 add_robot_view, noise=0.2):
        self.env_name = env_name
        self.num_timesteps_per_traj = num_timesteps_per_traj
        self.accept_trajectory_key = accept_trajectory_key
        self.noise = noise
        self.video_save_dir = video_save_dir
        self.success_only = success_only
        self.image_size = 512
        self.add_robot_view = add_robot_view

        if not os.path.exists(self.video_save_dir):
            os.makedirs(self.video_save_dir)
        # camera settings
        self.camera_target_pos = [0.6, 0.2, -0.2]
        self.camera_roll = 0.0
        self.camera_pitch = -40
        self.camera_yaw = 180
        self.camera_distance = 0.5
        self.view_matrix_args = dict(target_pos=self.camera_target_pos,
                                     distance=self.camera_distance,
                                     yaw=self.camera_yaw,
                                     pitch=self.camera_pitch,
                                     roll=self.camera_roll,
                                     up_axis_index=2)
        self.view_matrix = roboverse.bullet.get_view_matrix(
            **self.view_matrix_args)
        self.projection_matrix = roboverse.bullet.get_projection_matrix(
            self.image_size, self.image_size)
        # end camera settings
        self.env = self.instantiate_env()
        assert scripted_policy_name in policies.keys()
        policy_class = policies[scripted_policy_name]
        self.scripted_policy_class = policy_class(self.env)
        self.trajectories_collected = 0

    def instantiate_env(self):
        env = roboverse.make(self.env_name, gui=False,
                             transpose_image=False)
        return env

    def get_traj_and_success(self):
        self.trajectories_collected += 1
        print("trajectories collected", self.trajectories_collected)
        policy = self.scripted_policy_class
        traj, success, _ = collect_one_traj(
            self.env, policy, self.num_timesteps_per_traj,
            self.noise, self.accept_trajectory_key)
        return traj, success

    def get_single_traj(self):
        if self.success_only:
            success = False
            while not success:
                traj, success = self.get_traj_and_success()
            print("collected success", traj, success)
        else:
            traj, _ = self.get_traj_and_success()
        return traj

    def add_robot_view_to_video(self, images):
        image_x, image_y, image_c = images[0].shape
        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(len(images)):
            robot_view_margin = 5
            robot_view = cv2.resize(images[i],
                                    (ROBOT_VIEW_HEIGHT, ROBOT_VIEW_WIDTH))
            robot_view = robot_view[ROBOT_VIEW_CROP_X:, :, :]
            image_new = np.copy(images[i])
            x_offset = ROBOT_VIEW_HEIGHT-ROBOT_VIEW_CROP_X
            y_offset = image_y - ROBOT_VIEW_WIDTH

            # Draw a background black rectangle
            image_new = cv2.rectangle(image_new, (self.image_size, 0),
                                      (y_offset - 2 * robot_view_margin,
                                      x_offset + 25 + robot_view_margin),
                                      (0, 0, 0), -1)

            image_new[robot_view_margin:x_offset + robot_view_margin,
                      y_offset - robot_view_margin:-robot_view_margin,
                      :] = robot_view
            image_new = cv2.putText(image_new, 'Robot View',
                                    (y_offset - robot_view_margin,
                                     x_offset + 18 + robot_view_margin),
                                    font, 0.55, (255, 255, 255), 1,
                                    cv2.LINE_AA)
            images[i] = image_new

        return images

    def save_video_from_traj(self, traj, path_idx):
        actions = traj['actions']
        print("len(actions)", len(actions))
        images = []
        self.env.reset()
        for t in range(self.num_timesteps_per_traj):
            img, depth, segmentation = roboverse.bullet.render(
                self.image_size, self.image_size,
                self.view_matrix, self.projection_matrix)
            images.append(img)
            obs, rew, done, info = self.env.step(actions[t])  # step_slow
            # if len(imgs) > 0:
            #     images.extend(imgs)

        # Save Video
        save_path = "{}/{}_scripted_{}_reward_{}.mp4".format(
            self.video_save_dir, self.env_name, path_idx, int(rew))
        if self.add_robot_view:
            dot_idx = save_path.index(".")
            save_path = save_path[:dot_idx] + "_with_robot_view" + \
                save_path[dot_idx:]
        inputdict = {'-r': str(12)}
        outputdict = {'-vcodec': 'libx264', '-pix_fmt': 'yuv420p'}
        writer = skvideo.io.FFmpegWriter(
            save_path, inputdict=inputdict, outputdict=outputdict)

        if self.add_robot_view:
            self.add_robot_view_to_video(images)
        for i in range(len(images)):
            writer.writeFrame(images[i])
        writer.close()

    def save_videos(self, num_videos):
        for i in range(num_videos):
            traj = self.get_single_traj()
            self.save_video_from_traj(traj, i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str)
    parser.add_argument("--policy-name", type=str, required=True)
    parser.add_argument("--accept-trajectory-key", type=str, required=True)
    parser.add_argument("--num-timesteps", type=int, required=True)
    parser.add_argument("--video-save-dir", type=str, default="scripted_rollouts")
    parser.add_argument("--num-videos", type=int, default=1)
    parser.add_argument("--add-robot-view", action="store_true", default=False)
    parser.add_argument("--success-only", action="store_true", default=False)
    # Currently, success-only collects only successful trajectories,
    # but these trajectories do not always succeed again due to
    # randomized initial conditions
    args = parser.parse_args()

    vid_log = BulletVideoLogger(
        args.env, args.policy_name, args.num_timesteps,
        args.accept_trajectory_key, args.video_save_dir, args.success_only,
        args.add_robot_view)
    vid_log.save_videos(args.num_videos)
