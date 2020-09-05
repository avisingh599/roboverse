import numpy as np
import pybullet as p
import pybullet_data

GRAVITY = -10


def connect_headless(gui=False):
    if gui:
        cid = p.connect(p.SHARED_MEMORY)
        if cid < 0:
            p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.resetDebugVisualizerCamera(cameraDistance=0.8,
                                 cameraYaw=180,
                                 cameraPitch=-40,
                                 cameraTargetPosition=[0.6, 0, -0.4])
    p.setGravity(0, 0, GRAVITY)


def get_view_matrix(target_pos=[.75, -.2, 0], distance=0.9,
                    yaw=180, pitch=-20, roll=0, up_axis_index=2):
    view_matrix = p.computeViewMatrixFromYawPitchRoll(
        target_pos, distance, yaw, pitch, roll, up_axis_index)
    return view_matrix


def get_projection_matrix(height, width, fov=60, near_plane=0.1, far_plane=2):
    aspect = width / height
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near_plane,
                                                     far_plane)
    return projection_matrix


def get_joint_states(body_id, joint_indices):
    all_joint_states = p.getJointStates(body_id, joint_indices)
    joint_positions, joint_velocities = [], []
    for state in all_joint_states:
        joint_positions.append(state[0])
        joint_velocities.append(state[1])

    return np.asarray(joint_positions), np.asarray(joint_velocities)


def get_movable_joints(body_id):
    num_joints = p.getNumJoints(body_id)
    movable_joints = []
    for i in range(num_joints):
        if p.getJointInfo(body_id, i)[2] != p.JOINT_FIXED:
            movable_joints.append(i)
    return movable_joints


def get_link_state(body_id, link_index):
    position, orientation, _, _, _, _ = p.getLinkState(body_id, link_index)
    return position, orientation


def apply_action_ik(target_ee_pos, target_ee_quat, target_gripper_state,
                    robot_id, end_effector_index, movable_joints,
                    num_sim_steps=5):
    joint_poses = p.calculateInverseKinematics(robot_id,
                                               end_effector_index,
                                               target_ee_pos,
                                               target_ee_quat,
                                               jointDamping=[0.001] * len(
                                                   movable_joints),
                                               solver=0,
                                               maxNumIterations=100,
                                               residualThreshold=.01)

    p.setJointMotorControlArray(robot_id,
                                movable_joints,
                                controlMode=p.POSITION_CONTROL,
                                targetPositions=joint_poses,
                                # targetVelocity=0,
                                forces=[500] * len(movable_joints),
                                positionGains=[0.03] * len(movable_joints),
                                # velocityGain=1
                                )
    # set gripper action
    p.setJointMotorControl2(robot_id,
                            movable_joints[-2],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[0],
                            force=500,
                            positionGain=0.03)
    p.setJointMotorControl2(robot_id,
                            movable_joints[-1],
                            controlMode=p.POSITION_CONTROL,
                            targetPosition=target_gripper_state[1],
                            force=500,
                            positionGain=0.03)

    for _ in range(num_sim_steps):
        p.stepSimulation()


def reset_robot(robot_id, reset_joint_indices, reset_joint_values):
    assert len(reset_joint_indices) == len(reset_joint_values)
    for i, value in zip(reset_joint_indices, reset_joint_values):
        p.resetJointState(robot_id, i, value)


def quat_to_deg(quat):
    euler_rad = p.getEulerFromQuaternion(quat)
    euler_deg = rad_to_deg(euler_rad)
    return euler_deg


def deg_to_quat(deg):
    rad = deg_to_rad(deg)
    quat = p.getQuaternionFromEuler(rad)
    return quat


def deg_to_rad(deg):
    return np.array([d * np.pi / 180. for d in deg])


def rad_to_deg(rad):
    return np.array([r * 180. / np.pi for r in rad])


def render(height, width, view_matrix, projection_matrix,
           shadow=1, light_direction=[1, 1, 1],
           renderer=p.ER_BULLET_HARDWARE_OPENGL):
    #  ER_BULLET_HARDWARE_OPENGL
    img_tuple = p.getCameraImage(width,
                                 height,
                                 view_matrix,
                                 projection_matrix,
                                 shadow=shadow,
                                 lightDirection=light_direction,
                                 renderer=renderer)
    _, _, img, depth, segmentation = img_tuple
    # import ipdb; ipdb.set_trace()
    # Here, if I do len(img), I get 9216.
    # img = np.reshape(np.array(img), (48, 48, 4))

    img = img[:, :, :-1]
    return img, depth, segmentation
