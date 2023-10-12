# python
import torch

# solo-gym
from solo_gym.envs import LeggedRobot
from .solo8_config import Solo8FlatCfg
from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    torch_rand_float,
    quat_rotate,
    quat_rotate_inverse,
)
from learning.datasets.motion_loader import MotionLoader
from typing import Dict
from solo_gym.utils.keyboard_controller import KeyboardAction, Delta

class Solo8(LeggedRobot):
    cfg: Solo8FlatCfg

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # load AMP components
        self.reference_motion_file = self.cfg.motion_loader.reference_motion_file
        self.test_mode = self.cfg.motion_loader.test_mode
        self.test_observation_dim = self.cfg.motion_loader.test_observation_dim
        self.reference_observation_horizon = self.cfg.motion_loader.reference_observation_horizon
        self.motion_loader = MotionLoader(
            device=self.device,
            motion_file=self.reference_motion_file,
            corruption_level=self.cfg.motion_loader.corruption_level,
            reference_observation_horizon=self.reference_observation_horizon,
            test_mode=self.test_mode,
            test_observation_dim=self.test_observation_dim
        )
        self.reference_state_idx_dict = self.motion_loader.state_idx_dict
        self.reference_full_dim = sum([ids[1] - ids[0] for ids in self.reference_state_idx_dict.values()])
        self.reference_observation_dim = sum([ids[1] - ids[0] for state, ids in self.reference_state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
        self.cassi_states = torch.zeros(
            self.num_envs, self.reference_full_dim, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.discriminator = None # assigned in runner
        self.cassi_observation_buf = torch.zeros(
            self.num_envs, self.reference_observation_horizon, self.reference_observation_dim, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.cassi_observation_buf[:, -1] = self.get_cassi_observations()

        # load DISDAIN components
        self.dis_observation_horizon = self.cfg.discriminator_ensemble.observation_horizon
        self.dis_state_idx_dict = self.cfg.discriminator_ensemble.state_idx_dict
        self.dis_full_dim = sum([ids[1] - ids[0] for ids in self.dis_state_idx_dict.values()])
        self.dis_observation_dim = sum([ids[1] - ids[0] for state, ids in self.dis_state_idx_dict.items() if ((state != "base_pos") and (state != "base_quat"))])
        self.dis_observation_start_dim = self.cfg.discriminator_ensemble.observation_start_dim
        self.dis_num_classes = self.cfg.discriminator_ensemble.num_classes
        self.dis_states = torch.zeros(
            self.num_envs, self.dis_full_dim, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.dis_observation_buf = torch.zeros(
            self.num_envs, self.dis_observation_horizon, self.dis_observation_dim, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.dis_observation_buf[:, -1] = self.get_dis_observations()
        self.discriminator_ensemble = None

    def post_physics_step(self):
        """check terminations, compute observations and rewards
        calls self._post_physics_step_callback() for common computations
        calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        if self.cfg.asset.enable_joint_force_sensors:
            self.gym.refresh_dof_force_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.base_height[:] = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1, keepdim=True)
        self.base_lin_vel_x[:] = self.base_lin_vel[:, :1]
        self.base_pos_x[:] = self.root_states[:, :1] - self.env_origins[:, :1]

        self._post_physics_step_callback()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.cassi_record_states()
        self.dis_record_states()
        self.next_cassi_observations = self.get_cassi_observations()
        self.next_dis_observations = self.get_dis_observations()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()  # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.update_cassi_observation_buf()
        self.update_dis_observation_buf()

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def update_cassi_observation_buf(self):
        self.cassi_observation_buf[:, :-1] = self.cassi_observation_buf[:, 1:].clone()
        self.cassi_observation_buf[:, -1] = self.next_cassi_observations.clone()

    def get_cassi_observation_buf(self):
        return self.cassi_observation_buf.clone()

    def update_dis_observation_buf(self):
        self.dis_observation_buf[:, :-1] = self.dis_observation_buf[:, 1:].clone()
        self.dis_observation_buf[:, -1] = self.next_dis_observations.clone()

    def get_dis_observation_buf(self):
        return self.dis_observation_buf.clone()

    def compute_observations(self):
        """Computes observations"""
        self.obs_buf = torch.cat(
            (
                self.base_lin_vel * self.obs_scales.lin_vel,
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                self.commands * self.commands_scale,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.actions,
            ),
            dim=-1,
        )
        style_selector_one_hot = torch.nn.functional.one_hot(self.style_selector, self.dis_num_classes)
        self.obs_buf = torch.cat((self.obs_buf, style_selector_one_hot), dim=-1)
        # add noise if needed
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _init_buffers(self):
        super()._init_buffers()
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel], device=self.device, requires_grad=False)
        self.hip_indices = torch.tensor([i for i in range(self.num_dof) if "KFE" not in self.dof_names[i]], device=self.device, requires_grad=False)
        self.knee_indices = torch.tensor([i for i in range(self.num_dof) if "KFE" in self.dof_names[i]], device=self.device, requires_grad=False)
        self.desired_torques = torch.zeros_like(self.torques)
        self.max_torque = torch.zeros_like(self.torques)
        self.min_torque = torch.zeros_like(self.torques)
        self.dof_vel_limits = torch.zeros_like(self.dof_vel)
        self.max_torque[:] = self.cfg.control.torque_limit
        self.min_torque[:] = -self.cfg.control.torque_limit
        self.dof_vel_limits[:, self.hip_indices] = 14.0
        self.dof_vel_limits[:, self.knee_indices] = 5.0
        self.base_height = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.style_selector = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.base_lin_vel_x = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_pos_x = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)

    def _resample_commands(self, env_ids):
        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],
            self.command_ranges["lin_vel_x"][1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :1] *= (torch.norm(self.commands[env_ids, :1], dim=1) > 0.2).unsqueeze(1)

    def _resample_style_selector(self, env_ids):
        self.style_selector[env_ids] = torch.randint(self.dis_num_classes, (len(env_ids),), device=self.device, requires_grad=False)

    def _get_keyboard_events(self) -> Dict[str, KeyboardAction]:
        """Simple keyboard controller for linear and angular velocity."""

        def print_command():
            print("[LeggedRobot]: Environment 0 command: ", self.commands[0])
            print("[LeggedRobot]: Environment 0 style selector: ", self.style_selector[0])

        key_board_events = {
            "u": Delta("lin_vel_x", amount=0.1, variable_reference=self.commands[:, 0], callback=print_command),
            "j": Delta("lin_vel_x", amount=-0.1, variable_reference=self.commands[:, 0], callback=print_command),
            "h": Delta("style_selector", amount=1, variable_reference=self.style_selector, callback=print_command),
            "k": Delta("style_selector", amount=-1, variable_reference=self.style_selector, callback=print_command),
        }
        return key_board_events

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:10] = 0.0  # commands
        noise_vec[10:18] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[18:26] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[26:34] = 0.0  # previous actions
        return noise_vec

    def reset_idx(self, env_ids):
        """Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length == 0):
            self.update_command_curriculum(env_ids)

        # reset robot states
        if self.cfg.domain_rand.reference_state_initialization:
            frames = self.motion_loader.get_frames(len(env_ids))
            env_ids_mask = torch.rand(len(env_ids), device=self.device, requires_grad=False) <= self.cfg.domain_rand.reference_state_initialization_prob
        else:
            frames = None
            env_ids_mask = None
        self._reset_dofs(env_ids, frames, env_ids_mask)
        self._reset_root_states(env_ids, frames, env_ids_mask)

        self._resample_commands(env_ids)

        # reset buffers
        self.last_actions[env_ids] = 0.0
        self.last_dof_vel[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            )
            self.episode_sums[key][env_ids] = 0.0
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
        self._resample_style_selector(env_ids)

    def get_cassi_observations(self):
        if self.test_mode:
            cassi_obs = torch.zeros(self.num_envs, self.test_observation_dim, device=self.device, requires_grad=False)
        else:
            cassi_obs = self.cassi_states[:, self.motion_loader.observation_start_dim:].clone()
        return cassi_obs

    def cassi_record_states(self):
        for key, value in self.reference_state_idx_dict.items():
            if key == "base_pos":
                self.cassi_states[:, value[0]: value[1]] = self._get_base_pos()
            elif key == "feet_pos":
                self.cassi_states[:, value[0]: value[1]] = self._get_feet_pos()
            else:
                self.cassi_states[:, value[0]: value[1]] = getattr(self, key)

    def get_dis_observations(self):
        dis_obs = self.dis_states[:, self.dis_observation_start_dim:].clone()
        return dis_obs

    def get_style_selector(self):
        style_selector = self.style_selector.clone()
        return style_selector

    def dis_record_states(self):
        for key, value in self.dis_state_idx_dict.items():
            if key == "base_pos":
                self.dis_states[:, value[0]: value[1]] = self._get_base_pos()
            elif key == "feet_pos":
                self.dis_states[:, value[0]: value[1]] = self._get_feet_pos()
            else:
                self.dis_states[:, value[0]: value[1]] = getattr(self, key)

    def _get_base_pos(self):
        return self.root_states[:, :3] - self.env_origins[:, :3]

    def _get_feet_pos(self):
        feet_pos_global = self.rigid_body_pos[:, self.feet_indices, :3]
        feet_pos_local = torch.zeros_like(feet_pos_global)
        for i in range(len(self.feet_indices)):
            feet_pos_local[:, i] = quat_rotate_inverse(
                self.base_quat,
                feet_pos_global[:, i]
            )
        return feet_pos_local.flatten(1, 2)    

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _reset_dofs(self, env_ids, frames, env_ids_mask):
        if frames is not None:
            self.dof_pos[env_ids[env_ids_mask]] = self.motion_loader.get_dof_pos(frames[env_ids_mask])
            self.dof_vel[env_ids[env_ids_mask]] = self.motion_loader.get_dof_vel(frames[env_ids_mask])
            self.dof_pos[env_ids[~env_ids_mask]] = self.default_dof_pos * torch_rand_float(
                0.5, 1.5, (len(env_ids[~env_ids_mask]), self.num_dof), device=self.device
            )
            self.dof_vel[env_ids[~env_ids_mask]] = 0.0
            
            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_dof_state_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.dof_state),
                                                gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        else:
            super()._reset_dofs(env_ids)

    def _reset_root_states(self, env_ids, frames, env_ids_mask):
        if frames is not None:
            root_pos = self.motion_loader.get_base_pos(frames[env_ids_mask])
            root_pos[:, :2] = root_pos[:, :2] + self.env_origins[env_ids[env_ids_mask], :2]
            self.root_states[env_ids[env_ids_mask], :3] = root_pos
            root_ori = self.motion_loader.get_base_quat(frames[env_ids_mask])
            self.root_states[env_ids[env_ids_mask], 3:7] = root_ori
            self.root_states[env_ids[env_ids_mask], 7:10] = quat_rotate(root_ori, self.motion_loader.get_base_lin_vel(frames[env_ids_mask]))
            self.root_states[env_ids[env_ids_mask], 10:13] = quat_rotate(root_ori, self.motion_loader.get_base_ang_vel(frames[env_ids_mask]))

            if self.custom_origins:
                self.root_states[env_ids[~env_ids_mask]] = self.base_init_state
                self.root_states[env_ids[~env_ids_mask], :3] += self.env_origins[env_ids[~env_ids_mask]]
                self.root_states[env_ids[~env_ids_mask], :2] += torch_rand_float(
                    -1.0, 1.0, (len(env_ids[~env_ids_mask]), 2), device=self.device
                )
            else:
                self.root_states[env_ids[~env_ids_mask]] = self.base_init_state
                self.root_states[env_ids[~env_ids_mask], :3] += self.env_origins[env_ids[~env_ids_mask]]
            self.root_states[env_ids[~env_ids_mask], 7:13] = torch_rand_float(
                -0.5, 0.5, (len(env_ids[~env_ids_mask]), 6), device=self.device
            )

            env_ids_int32 = env_ids.to(dtype=torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.root_states),
                                                        gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        else:
            super()._reset_root_states(env_ids)

    def _compute_torques(self, actions):
        # save desired torques before clipping
        self.desired_torques = super()._compute_torques(actions)
        return torch.clip(self.desired_torques, min=self.min_torque, max=self.max_torque)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (only x axis)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :1] - self.base_lin_vel[:, :1]), dim=1)
        return torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_impacts(self):
        body_indices = [1, 2, 4, 5, 7, 8, 10, 11] # knees and feet
        # body_indices = self.feet_indices
        # acc = torch.norm(self.last_bodies_vel[:, body_indices] - self.rigid_body_states[:, body_indices, 7:10], dim=2) / self.dt
        acc = torch.abs((self.last_bodies_vel[:, body_indices, 2] - self.rigid_body_states[:, body_indices, 9]) / self.dt)
        acc = acc.clip(min=20.) - 20.
        # self.last_bodies_vel[:] = self.rigid_body_states[:, :, 7:10]
        # print(acc[0])
        # acc = (acc / (torch.norm(self.rigid_body_states[:, body_indices, 7:10], dim=2) + 1.e-6)).clip(min=50.) - 50.
        self.impact_reward = acc[0, 0] #torch.sum(torch.square(acc), dim=1) * (self.episode_length_buf != 1)
        return torch.sum(torch.square(acc), dim=1) * (self.episode_length_buf != 1)

    def _reward_cassi_style(self):
        cassi_observation_buf = torch.cat((self.cassi_observation_buf[:, 1:], self.next_cassi_observations.unsqueeze(1)), dim=1)
        cassi_style_reward = self.discriminator.predict_cassi_reward(cassi_observation_buf)
        return cassi_style_reward

    def _reward_dis_skill(self):
        dis_observation_buf = torch.cat((self.dis_observation_buf[:, 1:], self.next_dis_observations.unsqueeze(1)), dim=1)
        style_selector = self.get_style_selector()
        dis_skill_reward = self.discriminator_ensemble.compute_dis_skill_reward(dis_observation_buf, style_selector)
        return dis_skill_reward

    def _reward_dis_disdain(self):
        dis_observation_buf = torch.cat((self.dis_observation_buf[:, 1:], self.next_dis_observations.unsqueeze(1)), dim=1)
        dis_disdain_reward = self.discriminator_ensemble.compute_dis_disdain_reward(dis_observation_buf)
        return dis_disdain_reward

    def _reward_ang_vel_x(self):
        return torch.abs(self.base_ang_vel[:, 0])

    def _reward_lin_vel_y(self):
        return torch.abs(self.base_lin_vel[:, 1])

    def _reward_ang_vel_z(self):
        return torch.abs(self.base_ang_vel[:, 2])
