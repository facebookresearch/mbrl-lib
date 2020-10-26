import pathlib
from typing import Callable, Tuple

import dmc2gym.wrappers
import gym
import gym.envs.mujoco
import gym.wrappers
import hydra
import numpy as np
import omegaconf
import pytorch_sac
import torch

import mbrl.env
import mbrl.env.wrappers


# TODO rename
def get_environment_from_str(
    cfg: omegaconf.DictConfig,
) -> Tuple[gym.Env, Callable, Callable]:
    if "dmcontrol___" in cfg.env:
        domain, task = cfg.env.split("___")[1].split("--")
        term_fn = getattr(mbrl.env.termination_fns, domain)
        reward_fn = getattr(mbrl.env.reward_fns, cfg.term_fn, None)
        env = dmc2gym.make(domain_name=domain, task_name=task)
    elif "gym___" in cfg.env:
        env = gym.make(cfg.env.split("___")[1])
        term_fn = getattr(mbrl.env.termination_fns, cfg.term_fn)
        reward_fn = getattr(mbrl.env.reward_fns, cfg.term_fn, None)
    elif cfg.env == "cartpole_continuous":
        env = mbrl.env.cartpole_continuous.CartPoleEnv()
        term_fn = getattr(mbrl.env.termination_fns, cfg.term_fn)
        reward_fn = getattr(mbrl.env.reward_fns, cfg.term_fn, None)
    else:
        raise ValueError("Invalid environment string.")

    if cfg.learned_rewards:
        reward_fn = None

    normalize = cfg.get("normalize", False)
    if normalize:
        env = mbrl.env.wrappers.NormalizedEnv(env)
    return env, term_fn, reward_fn


class freeze_mujoco_env:
    def __init__(self, env: gym.wrappers.TimeLimit):
        self._env = env
        self._init_state: np.ndarray = None
        self._elapsed_steps = 0
        self._step_count = 0

        if isinstance(self._env.env, gym.envs.mujoco.MujocoEnv):
            self._enter_method = self._enter_mujoco_gym
            self._exit_method = self._exit_mujoco_gym
        elif isinstance(self._env.env, dmc2gym.wrappers.DMCWrapper):
            self._enter_method = self._enter_dmcontrol
            self._exit_method = self._exit_dmcontrol
        else:
            raise RuntimeError("Tried to freeze an unsupported environment.")

    def _enter_mujoco_gym(self):
        self._init_state = (
            self._env.env.data.qpos.ravel().copy(),
            self._env.env.data.qvel.ravel().copy(),
        )
        self._elapsed_steps = self._env._elapsed_steps

    def _exit_mujoco_gym(self):
        self._env.set_state(*self._init_state)
        self._env._elapsed_steps = self._elapsed_steps

    def _enter_dmcontrol(self):
        self._init_state = self._env.env._env.physics.get_state().copy()
        self._elapsed_steps = self._env._elapsed_steps
        self._step_count = self._env.env._env._step_count

    def _exit_dmcontrol(self):
        with self._env.env._env.physics.reset_context():
            self._env.env._env.physics.set_state(self._init_state)
            self._env._elapsed_steps = self._elapsed_steps
            self._env.env._env._step_count = self._step_count

    def __enter__(self):
        return self._enter_method()

    def __exit__(self, *_args):
        return self._exit_method()


class SACAgent(mbrl.Agent):
    def __init__(self, sac_agent: pytorch_sac.SACAgent):
        self.sac_agent = sac_agent

    def act(
        self, obs: np.ndarray, sample: bool = False, batched: bool = False, **_kwargs
    ) -> np.ndarray:
        return self.sac_agent.act(obs, sample=sample, batched=batched)


def complete_sac_cfg(env: gym.Env, cfg: omegaconf.DictConfig) -> omegaconf.DictConfig:
    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    cfg.agent.obs_dim = obs_shape[0]
    cfg.agent.action_dim = act_shape[0]
    cfg.agent.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]

    return cfg


def get_agent(agent_path: pathlib.Path, env: gym.Env, agent_type: str) -> mbrl.Agent:
    if agent_type == "pytorch_sac":
        cfg = omegaconf.OmegaConf.load(agent_path / ".hydra" / "config.yaml")
        cfg.agent._target_ = "pytorch_sac.agent.sac.SACAgent"
        cfg = complete_sac_cfg(env, cfg)
        agent: pytorch_sac.SACAgent = hydra.utils.instantiate(cfg.agent)
        agent.critic.load_state_dict(torch.load(agent_path / "critic.pth"))
        agent.actor.load_state_dict(torch.load(agent_path / "actor.pth"))
        return SACAgent(agent)
    else:
        raise ValueError(
            f"Invalid agent type {agent_type}. Supported options are: 'pytorch_sac'."
        )
