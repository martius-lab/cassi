#  Copyright 2021 ETH Zurich, NVIDIA CORPORATION
#  SPDX-License-Identifier: BSD-3-Clause

# python
import os
import time
import statistics
from collections import deque
from typing import Union

# torch
import torch
from torch.utils.tensorboard import SummaryWriter

# learning
import learning
from learning.algorithms import CASSI
from learning.modules import ActorCritic, ActorCriticRecurrent
from learning.modules.discriminator import Discriminator
from learning.modules.normalizer import Normalizer
from learning.modules.discriminator_ensemble import DiscriminatorEnsemble
from learning.env import VecEnv


class CASSIOnPolicyRunner:
    def __init__(self, env: VecEnv, train_cfg, log_dir=None, device="cpu"):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.discriminator_cfg = train_cfg["discriminator"]
        self.discriminator_ensemble_cfg = train_cfg["discriminator_ensemble"]

        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: Union[ActorCritic, ActorCriticRecurrent] = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"])  # CASSI
        cassi_expert_data = self.env.motion_loader
        cassi_state_normalizer = Normalizer(cassi_expert_data.observation_dim, self.device)
        if self.cfg["normalize_style_reward"]:
            cassi_style_reward_normalizer = Normalizer(1, self.device)
        else:
            cassi_style_reward_normalizer = None
        discriminator = Discriminator(
            observation_dim=cassi_expert_data.observation_dim,
            observation_horizon=self.env.reference_observation_horizon,
            device=self.device,
            state_normalizer=cassi_state_normalizer,
            reward_normalizer=cassi_style_reward_normalizer,
            **self.discriminator_cfg).to(self.device)
        discriminator_ensemble = DiscriminatorEnsemble(
            observation_dim=self.env.dis_observation_dim,
            observation_horizon=self.env.dis_observation_horizon,
            num_classes=self.env.dis_num_classes,
            device=self.device,
            **self.discriminator_ensemble_cfg).to(self.device)

        self.alg: CASSI = alg_class(actor_critic, discriminator, cassi_expert_data, discriminator_ensemble, device=self.device, **self.alg_cfg)
        self.env.discriminator = discriminator
        self.env.discriminator_ensemble = discriminator_ensemble
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.git_status_repos = [learning.__file__]

        _, _ = self.env.reset()

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        cassi_observation_buf = self.env.get_cassi_observation_buf()
        dis_observation_buf = self.env.get_dis_observation_buf()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs, cassi_observation_buf, dis_observation_buf = obs.to(self.device), critic_obs.to(self.device), cassi_observation_buf.to(self.device), dis_observation_buf.to(self.device)
        self.alg.actor_critic.train()  # switch to train mode (for dropout for example)
        self.alg.discriminator.train()
        self.alg.discriminator_ensemble.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs, cassi_observation_buf, dis_observation_buf)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    next_cassi_obs = self.env.get_cassi_observations()
                    next_dis_obs = self.env.get_dis_observations()
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, next_cassi_obs, next_dis_obs, rewards, dones = (
                        obs.to(self.device),
                        critic_obs.to(self.device),
                        next_cassi_obs.to(self.device),
                        next_dis_obs.to(self.device),
                        rewards.to(self.device),
                        dones.to(self.device),
                    )
                    cassi_observation_buf[:, :-1] = cassi_observation_buf[:, 1:].clone()
                    cassi_observation_buf[:, -1] = next_cassi_obs.clone()
                    dis_observation_buf[:, :-1] = dis_observation_buf[:, 1:].clone()
                    dis_observation_buf[:, -1] = next_dis_obs.clone()
                    style_selector = self.env.get_style_selector()
                    self.alg.process_env_step(rewards, dones, infos, next_cassi_obs, next_dis_obs, style_selector)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)

            mean_value_loss, mean_surrogate_loss, mean_cassi_loss, mean_grad_pen_loss, mean_policy_pred, mean_expert_pred, mean_discriminator_ensemble_loss, discriminator_ensemble_accuracy = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, "model_{}.pt".format(it)))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, "model_{}.pt".format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.policy_learning_rate, locs["it"])
        self.writer.add_scalar("Loss/CASSI", locs["mean_cassi_loss"], locs["it"])
        self.writer.add_scalar("Loss/cassi_grad", locs["mean_grad_pen_loss"], locs["it"])
        self.writer.add_scalar("Discriminator/policy_pred", locs["mean_policy_pred"], locs["it"])
        self.writer.add_scalar("Discriminator/expert_pred", locs["mean_expert_pred"], locs["it"])
        for i in range(self.alg.discriminator_ensemble_ensemble_size):
            self.writer.add_scalar(f"Discriminator_ensemble/loss/component_{i}", locs["mean_discriminator_ensemble_loss"][i], locs["it"])
            self.writer.add_scalar(f"Discriminator_ensemble/accuracy/component_{i}", locs["discriminator_ensemble_accuracy"][i], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_reward/time", statistics.mean(locs["rewbuffer"]), self.tot_time)
            self.writer.add_scalar("Train/mean_episode_length/time", statistics.mean(locs["lenbuffer"]), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs["rewbuffer"]) > 0:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'CASSI loss:':>{pad}} {locs['mean_cassi_loss']:.4f}\n"""
                f"""{'CASSI grad pen loss:':>{pad}} {locs['mean_grad_pen_loss']:.4f}\n"""
                f"""{'CASSI mean policy pred:':>{pad}} {locs['mean_policy_pred']:.4f}\n"""
                f"""{'CASSI mean expert pred:':>{pad}} {locs['mean_expert_pred']:.4f}\n"""
                f"""{'Discriminator ensemble loss:':>{pad}} {locs['mean_discriminator_ensemble_loss'].mean():.4f}\n"""
                f"""{'Discriminator ensemble accuracy:':>{pad}} {locs['discriminator_ensemble_accuracy'].mean():.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (
                f"""{'#' * width}\n"""
                f"""{str.center(width, ' ')}\n\n"""
                f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
            )
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (
            f"""{'-' * width}\n"""
            f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
            f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
            f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
            f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n"""
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "discriminator_state_dict": self.alg.discriminator.state_dict(),
                "discriminator_ensemble_state_dict": self.alg.discriminator_ensemble.state_dict(),
                "policy_optimizer_state_dict": self.alg.policy_optimizer.state_dict(),
                "discriminator_optimizer_state_dict": self.alg.discriminator_optimizer.state_dict(),
                "cassi_state_normalizer": self.alg.cassi_state_normalizer,
                "cassi_style_reward_normalizer": self.alg.cassi_style_reward_normalizer,
                "discriminator_ensemble_optimizer_state_dict": [discriminator_ensemble_optimizer.state_dict() for discriminator_ensemble_optimizer in self.alg.discriminator_ensemble_optimizer],
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.alg.cassi_state_normalizer = loaded_dict["cassi_state_normalizer"]
        self.alg.cassi_style_reward_normalizer = loaded_dict["cassi_style_reward_normalizer"]
        self.alg.discriminator_ensemble.load_state_dict(loaded_dict["discriminator_ensemble_state_dict"])
        if load_optimizer:
            self.alg.policy_optimizer.load_state_dict(loaded_dict["policy_optimizer_state_dict"])
            self.alg.discriminator_optimizer.load_state_dict(loaded_dict["discriminator_optimizer_state_dict"])
            for i in range(self.alg.discriminator_ensemble_ensemble_size):
                self.alg.discriminator_ensemble_optimizer[i].load_state_dict(loaded_dict["discriminator_ensemble_optimizer_state_dict"][i])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def add_git_repo_to_log(self, repo_file_path):
        self.git_status_repos.append(repo_file_path)
