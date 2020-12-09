"""Hello"""
import hydra
import numpy as np
import omegaconf
import torch

import mbrl.env
import mbrl.mbpo as mbpo
import mbrl.pets as pets
import mbrl.util as util


@hydra.main(config_path="conf", config_name="main")
def run(cfg: omegaconf.DictConfig):
    env, term_fn, reward_fn = mbrl.util.make_env(cfg)
    env.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.algorithm.name == "pets":
        return pets.train(env, term_fn, reward_fn, cfg)
    if cfg.algorithm.name == "mbpo":
        test_env, *_ = util.make_env(cfg)
        device = torch.device(cfg.device)
        return mbpo.train(env, test_env, term_fn, device, cfg)


if __name__ == "__main__":
    run()
