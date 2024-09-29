import torch
from tqdm import tqdm
import module.data_module as module_data
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
import pandas as pd
import os

from utils.utils import *

def inference(dataloader, model, criterion, metrics, device):
    outputs, targets = [], []
    total_loss = 0
    with torch.no_grad():
        for (inputs, target) in tqdm(dataloader, total=len(dataloader)):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            target = target.to(device)                
            output = model(inputs)
            loss = criterion(output, target)
            total_loss += loss.item()
            outputs.append(output)
            targets.append(target)

    outputs = torch.cat(outputs).squeeze()
    targets = torch.cat(targets).squeeze()
    result = {}
    result["loss"] = total_loss/len(dataloader)
    for metric in metrics:
        result[f"{metric.__name__}"] = metric(outputs, targets)

    return result, outputs

def main(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    config['data_module']['args']['shuffle'] = False

    # 1. set data_module
    data_module = init_obj(config['data_module']['type'], config['data_module']['args'], module_data)
    data_module.setup()
    train_dataloader = data_module.train_dataloader()
    dev_dataloader = data_module.dev_dataloader()
    test_dataloader = data_module.test_dataloader()

    # 2. set model
    model = init_obj(config['arch']['type'], config['arch']['args'], module_arch)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. set deivce(cpu or gpu)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    # 4. set loss function & matrics 
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # 5. inference
    train_result, train_outputs = inference(train_dataloader, model, criterion, metrics, device)
    dev_result, dev_outputs = inference(dev_dataloader, model, criterion, metrics, device)
    print(train_result)
    print(dev_result)
  
    test_outputs = []
    with torch.no_grad():
        for inputs in tqdm(test_dataloader):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            output = model(inputs)
            test_outputs.append(output)

    test_outputs = torch.cat(test_outputs).squeeze()

    train_outputs = train_outputs.tolist()
    dev_outputs = dev_outputs.tolist()
    test_outputs = test_outputs.tolist()

    # 6. save output
    pwd = os.getcwd()
    if not os.path.exists(f'{pwd}/output/'):
        os.makedirs(f'{pwd}/output/')
    folder_name = checkpoint_path.split("/")[-1].replace(".pth", "")
    folder_path = f'{pwd}/output/{folder_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()
    train_df['id'] = [f"boostcamp-sts-v1-train-{i:03d}" for i in range(len(train_outputs))]
    dev_df['id'] = [f"boostcamp-sts-v1-dev-{i:03d}" for i in range(len(dev_outputs))]
    test_df['id'] = [f"boostcamp-sts-v1-test-{i:03d}" for i in range(len(test_outputs))]
    train_df['target'] = train_outputs
    dev_df['target'] = dev_outputs
    test_df['target'] = test_outputs
    train_df.to_csv(f'{folder_path}/train_output.csv', index=False)
    dev_df.to_csv(f'{folder_path}/dev_output.csv', index=False)
    test_df.to_csv(f'{folder_path}/test_output.csv', index=False)

if __name__ == '__main__':

    checkpoint_path = '/Users/gj/Downloads/level1-semantictextsimilarity-nlp-11/saved/plm=klue-roberta-small_val-pearson=0.85345.pth'
    checkpoint_path = '/Users/gj/pytorch-template/saved/plm=klue-roberta-small_val-pearson=0.85868.pth'
    main(checkpoint_path)
    