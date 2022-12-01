import datetime
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import gym
import numpy as np
import stable_baselines3
from huggingface_hub import HfApi, upload_folder
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecVideoRecorder,
    unwrap_vec_normalize,
)
from wasabi import Printer

msg = Printer()


def _generate_config(model: BaseAlgorithm, local_path: Path) -> None:
    """
    Generate a config.json file containing information
    about the agent and the environment
    :param model: name of the model zip file
    :param local_path: path of the local directory
    """
    unzipped_model_folder = model

    # Check if the user forgot to mention the extension of the file
    if model.endswith(".zip") is False:
        model += ".zip"

    # Step 1: Unzip the model
    with zipfile.ZipFile(local_path / model, "r") as zip_ref:
        zip_ref.extractall(local_path / unzipped_model_folder)

    # Step 2: Get data (JSON containing infos) and read it
    with open(Path.joinpath(local_path, unzipped_model_folder, "data")) as json_file:
        data = json.load(json_file)
        # Add system_info elements to our JSON
        data["system_info"] = stable_baselines3.get_system_info(print_info=False)[0]

    # Step 3: Write our config.json file
    with open(local_path / "config.json", "w") as outfile:
        json.dump(data, outfile)


def _evaluate_agent(
    model: BaseAlgorithm,
    eval_env: VecEnv,
    n_eval_episodes: int,
    is_deterministic: bool,
    local_path: Path,
) -> Tuple[float, float]:
    """
    Evaluate the agent using SB3 evaluate_policy method
    and create a results.json
    :param model: name of the model object
    :param eval_env: environment used to evaluate the agent
    :param n_eval_episodes: number of evaluation episodes
    :param is_deterministic: use deterministic or stochastic actions
    :param local_path: path of the local repository
    """
    # Step 1: Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes, is_deterministic
    )

    # Step 2: Create json evaluation
    # First get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "is_deterministic": is_deterministic,
        "n_eval_episodes": n_eval_episodes,
        "eval_datetime": eval_form_datetime,
    }

    # Step 3: Write a JSON file
    with open(local_path / "results.json", "w") as outfile:
        json.dump(evaluate_data, outfile)

    return mean_reward, std_reward


def is_atari(env_id: str) -> bool:
    """
    Check if the environment is an Atari one
    (Taken from RL-Baselines3-zoo)
    :param env_id: name of the environment
    """
    entry_point = gym.envs.registry.env_specs[env_id].entry_point
    return "AtariEnv" in str(entry_point)


def _generate_replay(
    model: BaseAlgorithm,
    eval_env: VecEnv,
    video_length: int,
    is_deterministic: bool,
    local_path: Path,
):
    """
    Generate a replay video of the agent
    :param model: trained model
    :param eval_env: environment used to evaluate the agent
    :param video_length: length of the video (in timesteps)
    :param is_deterministic: use deterministic or stochastic actions
    :param local_path: path of the local repository
    """
    # This is another temporary directory for video outputs
    # SB3 created a -step-0-to-... meta files as well as other
    # artifacts which we don't want in the repo.
    with tempfile.TemporaryDirectory() as tmpdirname:
        # Step 1: Create the VecVideoRecorder
        env = VecVideoRecorder(
            eval_env,
            tmpdirname,
            record_video_trigger=lambda x: x == 0,
            video_length=video_length,
            name_prefix="",
        )

        obs = env.reset()
        lstm_states = None
        episode_starts = np.ones((env.num_envs,), dtype=bool)

        try:
            for _ in range(video_length + 1):
                action, lstm_states = model.predict(
                    obs,
                    state=lstm_states,
                    episode_start=episode_starts,
                    deterministic=is_deterministic,
                )
                obs, _, episode_starts, _ = env.step(action)

            # Save the video
            env.close()

            # Convert the video with x264 codec
            inp = env.video_recorder.path
            out = os.path.join(local_path, "replay.mp4")
            os.system(f"ffmpeg -y -i {inp} -vcodec h264 {out}".format(inp, out))

        except KeyboardInterrupt:
            pass
        except Exception as e:
            msg.fail(str(e))
            # Add a message for video
            msg.fail(
                "We are unable to generate a replay of your agent, "
                "the package_to_hub process continues"
            )
            msg.fail(
                "Please open an issue at "
                "https://github.com/huggingface/huggingface_sb3/issues"
            )


def generate_metadata(
    model_name: str, env_id: str, mean_reward: float, std_reward: float
) -> Dict[str, Any]:
    """
    Define the tags for the model card
    :param model_name: name of the model
    :param env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    """
    metadata = {}
    metadata["library_name"] = "stable-baselines3"
    metadata["tags"] = [
        env_id,
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "stable-baselines3",
    ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=model_name,
        task_pretty_name="reinforcement-learning",
        task_id="reinforcement-learning",
        metrics_pretty_name="mean_reward",
        metrics_id="mean_reward",
        metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
        dataset_pretty_name=env_id,
        dataset_id=env_id,
    )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    return metadata


def _generate_model_card(
    model_name: str, env_id: str, mean_reward: float, std_reward: float
) -> Tuple[str, Dict[str, Any]]:
    """
    Generate the model card for the Hub
    :param model_name: name of the model
    :env_id: name of the environment
    :mean_reward: mean reward of the agent
    :std_reward: standard deviation of the mean reward of the agent
    """
    # Step 1: Select the tags
    metadata = generate_metadata(model_name, env_id, mean_reward, std_reward)

    # Step 2: Generate the model card
    model_card = f"""
# **{model_name}** Agent playing **{env_id}**
This is a trained model of a **{model_name}** agent playing **{env_id}**
using the [stable-baselines3 library](https://github.com/DLR-RM/stable-baselines3).
"""

    model_card += """
## Usage (with Stable-baselines3)
TODO: Add your code
```python
from stable_baselines3 import ...
from huggingface_sb3 import load_from_hub
...
```
"""

    return model_card, metadata


def _save_model_card(
    local_path: Path, generated_model_card: str, metadata: Dict[str, Any]
):
    """Saves a model card for the repository.
    :param local_path: repository directory
    :param generated_model_card: model card generated by _generate_model_card()
    :param metadata: metadata
    """
    readme_path = local_path / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    else:
        readme = generated_model_card

    with readme_path.open("w", encoding="utf-8") as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)


def _add_logdir(local_path: Path, logdir: Path):
    """Adds a logdir to the repository.
    :param local_path: repository directory
    :param logdir: logdir directory
    """
    if logdir.exists() and logdir.is_dir():
        # Add the logdir to the repository under new dir called logs
        repo_logdir = local_path / "logs"

        # Delete current logs if they exist
        if repo_logdir.exists():
            shutil.rmtree(repo_logdir)

        # Copy logdir into repo logdir
        shutil.copytree(logdir, repo_logdir)


def package_to_hub(
    model: BaseAlgorithm,
    model_name: str,
    model_architecture: str,
    env_id: str,
    eval_env: Union[VecEnv, gym.Env],
    repo_id: str,
    commit_message: str,
    is_deterministic: bool = True,
    n_eval_episodes=10,
    token: Optional[str] = None,
    video_length=1000,
    logs=None,
):
    """
    Evaluate, Generate a video and Upload a model to Hugging Face Hub.
    This method does the complete pipeline:
    - It evaluates the model
    - It generates the model card
    - It generates a replay video of the agent
    - It pushes everything to the hub
    :param model: trained model
    :param model_name: name of the model zip file
    :param model_architecture: name of the architecture of your model
        (DQN, PPO, A2C, SAC...)
    :param env_id: name of the environment
    :param eval_env: environment used to evaluate the agent
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param commit_message: commit message
    :param is_deterministic: use deterministic or stochastic actions (by default: True)
    :param n_eval_episodes: number of evaluation episodes (by default: 10)
    :param video_length: length of the video (in timesteps)
    :param logs: directory on local machine of tensorboard logs you'd like to upload
    """

    # Autowrap, so we only have VecEnv afterward
    if not isinstance(eval_env, VecEnv):
        eval_env = DummyVecEnv([lambda: eval_env])

    msg.info(
        "This function will save, evaluate, generate a video of your agent, "
        "create a model card and push everything to the hub. "
        "It might take up to 1min. \n "
        "This is a work in progress: if you encounter a bug, please open an issue."
    )

    repo_url = HfApi().create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True,
    )

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)

        # Step 1: Save the model
        model.save(tmpdirname / model_name)

        # Retrieve VecNormalize wrapper if it exists
        # we need to save the statistics
        maybe_vec_normalize = unwrap_vec_normalize(eval_env)

        # Save the normalization
        if maybe_vec_normalize is not None:
            maybe_vec_normalize.save(tmpdirname / "vec_normalize.pkl")
            # Do not update the stats at test time
            maybe_vec_normalize.training = False
            # Reward normalization is not needed at test time
            maybe_vec_normalize.norm_reward = False

        # We create two versions of the environment:
        # one for video generation and one for evaluation
        replay_env = eval_env

        # Deterministic by default (except for Atari)
        if is_deterministic:
            is_deterministic = not is_atari(env_id)

        # Step 2: Create a config file
        _generate_config(model_name, tmpdirname)

        # Step 3: Evaluate the agent
        mean_reward, std_reward = _evaluate_agent(
            model, eval_env, n_eval_episodes, is_deterministic, tmpdirname
        )

        # Step 4: Generate a video
        _generate_replay(model, replay_env, video_length, is_deterministic, tmpdirname)

        # Step 5: Generate the model card
        generated_model_card, metadata = _generate_model_card(
            model_architecture, env_id, mean_reward, std_reward
        )
        _save_model_card(tmpdirname, generated_model_card, metadata)

        # Step 6: Add logs if needed
        if logs:
            _add_logdir(tmpdirname, Path(logs))

        msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")

        repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=tmpdirname,
            path_in_repo="",
            commit_message=commit_message,
            token=token,
        )

        msg.info(
            f"Your model is pushed to the Hub. You can view your model here: {repo_url}"
        )
    return repo_url


def _copy_file(filepath: Path, dst_directory: Path):
    """
    Copy the file to the correct directory
    :param filepath: path of the file
    :param dst_directory: destination directory
    """
    dst = dst_directory / filepath.name
    shutil.copy(str(filepath.name), str(dst))


def push_to_hub(
    repo_id: str,
    filename: str,
    commit_message: str,
    token: Optional[str] = None,
):
    """
    Upload a model to Hugging Face Hub.
    :param repo_id: repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip or mp4 file from the repository
    :param commit_message: commit message
    :param token
    """

    repo_url = HfApi().create_repo(
        repo_id=repo_id,
        token=token,
        private=False,
        exist_ok=True,
    )

    # Add the model
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = Path(tmpdirname)
        filename_path = os.path.abspath(filename)
        _copy_file(Path(filename_path), tmpdirname)
        _save_model_card(tmpdirname, "", {})

        msg.info(f"Pushing repo {repo_id} to the Hugging Face Hub")
        repo_url = upload_folder(
            repo_id=repo_id,
            folder_path=tmpdirname,
            path_in_repo="",
            commit_message=commit_message,
            token=token,
        )

    msg.good(
        f"Your model has been uploaded to the Hub, you can find it here: {repo_url}"
    )
    return repo_url


def load_from_hub(repo_id: str, filename: str) -> str:
    """
    Download a model from Hugging Face Hub.
    :param repo_id: id of the model repository from the Hugging Face Hub
    :param filename: name of the model zip file from the repository
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "You need to install huggingface_hub to use `load_from_hub`. "
            "See https://pypi.org/project/huggingface-hub/ for installation."
        )

    # Get the model from the Hub, download and cache the model on your local disk
    downloaded_model_file = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        library_name="huggingface-mbrl-lib",
        library_version="2.1",
    )

    return downloaded_model_file
