import roboverse
import os
import argparse
import numpy as np
from PIL import Image
import roboverse.bullet as bullet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, required=True)
    args = parser.parse_args()
    camera_target_pos = (0.7, 0.2, -0.22)  # (0.7, 0.2, -0.28)
    camera_distance = 0.25 # .125
    camera_yaw, camera_pitch, camera_roll = 90, 0, 0
    img_h, img_w = 512, 512
    env = roboverse.make('Widow250PutOnMarbleCubeTestRL4-v0',
                         gui=True, transpose_image=False, in_vr_replay=True,
                         deter_container_position=(0.5, -0.5, -0.3),
                         deter_objects_positions=((0.65, 0.2, -0.35),
                                                  (0.65, 0.28, -0.35)))
    obs, _, _, _ = env.step(np.array([0]*7))
    view_matrix_args = dict(target_pos=camera_target_pos,
                            distance=camera_distance,
                            yaw=camera_yaw,
                            pitch=camera_pitch,
                            roll=camera_roll,
                            up_axis_index=2)
    view_matrix_obs = bullet.get_view_matrix(**view_matrix_args)
    projection_matrix_obs = bullet.get_projection_matrix(
        img_h, img_w)
    img, _, _ = bullet.render(
        img_h, img_w, view_matrix_obs,
        projection_matrix_obs, shadow=0)
    # print("img", img)
    im = Image.fromarray(img)
    im.save(os.path.join(args.save_path, '{}.png'.format(0)))
