import argparse
import torch
import numpy as np
import base.base_data_loader as data_modules
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
from module.trainer import Trainer
from omegaconf import DictConfig, OmegaConf
import hydra


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def init_obj(module_name, module_args, module, *args):
    return getattr(module, module_name)(*args, **module_args)

@hydra.main(config_path=".", config_name="config")
def main(cfg):

    # 0. DictConfig to dict
    config = OmegaConf.to_container(cfg, resolve=True)
    
    # 1. set data_module(=pl.DataModule class)
    data_module = init_obj(config['data_module']['type'], config['data_module']['args'], data_modules)

    # 2. set model(=nn.Module class)
    model = init_obj(config['arch']['type'], config['arch']['args'], module_arch)

    # 3. set deivce(cpu or gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 4. set loss function & matrics 
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # 5. set optimizer & learning scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = init_obj(config['optimizer']['type'], config['optimizer']['args'], torch.optim, trainable_params)
    lr_scheduler = init_obj(config['lr_scheduler']['type'], config['lr_scheduler']['args'], torch.optim.lr_scheduler, optimizer)

    # 6. 위에서 설정한 내용들을 trainer에 넣는다.
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_module=data_module,
                      lr_scheduler=lr_scheduler)

    # 6. train
    trainer.train()


if __name__ == '__main__':
    main()
