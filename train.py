import os
import torch
import numpy as np
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
import module.data_module as module_data
from module.trainer import *
from omegaconf import DictConfig, OmegaConf
from utils.utils import *
import hydra

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg):

    # configuration
    cfg.pwd = os.getcwd()
    # 0. DictConfig to dict
    config = OmegaConf.to_container(cfg, resolve=True)

    # 0.1. initialize wandb run
    if config['wandb']['enable']:
        wandb.init(project=config["wandb"]["project_name"])

    # 0.2. naming wandb run
    config['run_name'] = \
        f"plm={config['arch']['args']['plm_name']}"
    wandb.run.name = config['run_name']
    wandb.run.save()
    print(f"run name: {config['run_name']}")

    # 1. set data module
    data_module = init_obj(
        config["data_module"]["type"], config["data_module"]["args"], module_data
    )
    # 1.1. initialize dataset
    data_module.setup()

    # 2. set model
    model = init_obj(config["arch"]["type"], config["arch"]["args"], module_arch)

    # 3. set deivce(cpu or gpu)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    # 4. set loss function & matrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # 5. set optimizer & learning scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = init_obj(
        config["optimizer"]["type"],
        config["optimizer"]["args"],
        torch.optim,
        trainable_params,
    )
    lr_scheduler = init_obj(
        config["lr_scheduler"]["type"],
        config["lr_scheduler"]["args"],
        torch.optim.lr_scheduler,
        optimizer,
    )

    # 6. 위에서 설정한 내용들을 trainer에 넣는다.
    trainer = DefaultTrainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_module=data_module,
        lr_scheduler=lr_scheduler,
    )

    # 6. train
    best_dev_output = trainer.train()


if __name__ == "__main__":
    main()