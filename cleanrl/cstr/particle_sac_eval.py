from typing import Callable

import gymnasium as gym
import torch
import torch.nn as nn

import numpy as np


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    device: torch.device = torch.device("cpu"),
    capture_frame: bool=True,
    num_particles: int = 10,
    combine_particles: int = 0,
    estimate_state: int = 1,
    mean_state: int = 0,
    reward_func: str = 'prob',
    stochastic_env: int = 0,
    randomized_env: int = 0,
    tol: float = 0.1
    ):

    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, False, capture_frame, run_name, num_particles, combine_particles, estimate_state, mean_state, reward_func, stochastic_env, randomized_env, tol)])
    nx = np.prod(envs.single_observation_space.shape)
    nu = np.prod(envs.single_action_space.shape)
    nu_high, nu_low = envs.single_action_space.high, envs.single_action_space.low

    actor = Model[0](nx, nu, nu_high, nu_low, num_particles, combine_particles, estimate_state).to(device)
    actor_params, qf1_params, qf2_params = torch.load(model_path, map_location=device)
    actor.load_state_dict(actor_params)
    actor.eval()
    envs.single_observation_space.dtype = np.float32

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        with torch.no_grad():
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # next_obs, _, _, _, infos = envs.step(actions)
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


if __name__ == "__main__":

    from cleanrl.sac_continuous_action import Actor, SoftQNetwork, make_env

    # model_path = hf_hub_download(
    #     repo_id="cleanrl/HalfCheetah-v4-td3_continuous_action-seed1", filename="td3_continuous_action.cleanrl_model"
    # )
    evaluate(
        model_path,
        make_env,
        "ParticleStateEnv-v1",
        eval_episodes=10,
        run_name=f"eval",
        Model=(Actor, QNetwork),
        device="cpu",
        capture_frame=True
    )
