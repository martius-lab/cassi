from solo_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from solo_gym import LEGGED_GYM_ROOT_DIR


class Solo8FlatCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_observations = 40 # 34 + cla_num_classes
        num_actions = 8

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = "plane"
        curriculum = False
        measure_heights = False
        terrain_proportions = [0.0, 1.0]
        num_rows = 5
        max_init_terrain_level = 4

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            "FL_HFE": 1.0,
            "HL_HFE": -1.0,
            "FR_HFE": 1.0,
            "HR_HFE": -1.0,

            "FL_KFE": -2.0,
            "HL_KFE": 2.0,
            "FR_KFE": -2.0,
            "HR_KFE": 2.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        stiffness = {'HFE': 5.0, 'KFE': 5.0} # [N*m/rad]
        damping = {'HFE': 0.1, 'KFE': 0.1} # [N*m*s/rad]
        torque_limit = 2.5

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/solo8/urdf/solo8.urdf'
        foot_name = "FOOT"
        terminate_after_contacts_on = ["base", "UPPER"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.85
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.24
        max_contact_force = 350.0
        only_positive_rewards = True
        class scales(LeggedRobotCfg.rewards.scales):
            orientation = -0.0
            torques = -0.000025
            feet_air_time = 0.5
            collision = -0.0
            lin_vel_z = -0.0
            ang_vel_xy = -0.0
            stand_still = -0.02
            base_height = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.0
            ang_vel_x = -0.02
            lin_vel_y = -0.02
            ang_vel_z = -0.02
            cassi_style = 1.0
            dis_skill = 1.0
            dis_disdain = 10.0

    class commands(LeggedRobotCfg.commands):
        num_commands = 1
        curriculum = False
        max_curriculum = 5.0
        resampling_time = 5.0
        heading_command = False
        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [0.0, 1.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        push_robots = True
        max_push_vel_xy = 0.5
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.0]
        reference_state_initialization = True
        reference_state_initialization_prob = 0.85

    class motion_loader:
        reference_motion_file = LEGGED_GYM_ROOT_DIR + "/resources/robots/solo8/datasets/motion_data.pt"
        corruption_level = 0.0
        reference_observation_horizon = 2
        test_mode = False
        test_observation_dim = None # observation_dim of reference motion

    class discriminator_ensemble:
        state_idx_dict = {
            "base_pos": [0, 3],
            "base_quat": [3, 7],
            "base_lin_vel": [7, 10],
            "base_ang_vel": [10, 13],
            "projected_gravity": [13, 16],
            "base_height": [16, 17],
            "dof_pos": [17, 25],
            "dof_vel": [25, 33],
        }
        num_classes = 6
        observation_start_dim = 7
        observation_horizon = 8

class Solo8FlatCfgPPO(LeggedRobotCfgPPO):
    runner_class_name = "CASSIOnPolicyRunner"
    class policy(LeggedRobotCfgPPO.policy):
        actor_hidden_dims = [128, 128, 128]
        critic_hidden_dims = [128, 128, 128]
        init_noise_std = 1.0

    class discriminator:
        style_reward_function = "quad_mapping"  # log_mapping, quad_mapping, wasserstein_mapping
        shape = [512, 256]

    class discriminator_ensemble:
        shape = [256, 256]
        ensemble_size = 5
        incremental_input = False

    class algorithm(LeggedRobotCfgPPO.algorithm):
        cassi_replay_buffer_size = 100000
        discriminator_ensemble_replay_buffer_size = 100000
        policy_learning_rate = 1e-3
        discriminator_learning_rate = 1e-4
        discriminator_momentum = 0.5
        discriminator_weight_decay = 1e-4
        discriminator_gradient_penalty_coef = 5
        discriminator_loss_function = "MSELoss"  # MSELoss, BCEWithLogitsLoss, WassersteinLoss
        discriminator_num_mini_batches = 80
        discriminator_ensemble_learning_rate = 1e-4
        discriminator_ensemble_weight_decay = 0.0005
        discriminator_ensemble_num_mini_batches = 80

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'cassi'
        experiment_name = 'flat_solo8'
        algorithm_class_name = "CASSI"
        policy_class_name = "ActorCritic"
        load_run = -1
        max_iterations = 5000
        normalize_style_reward = True
        master_classifier_file = LEGGED_GYM_ROOT_DIR + "/resources/robots/solo8/master_classifier/model.pt"
