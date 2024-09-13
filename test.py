import argparse
import torch
from tqdm import tqdm
import base.base_data_loader as module_data
import module.loss as module_loss
import module.metric as module_metric
import module.model as module_arch
from parse_config import ConfigParser
import pandas as pd

def main(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']

    # 1. set data_module(=pl.DataModule class)
    data_module = config.init_obj('data_module', module_data)
    data_module.setup('test')
    test_dataloader = data_module.test_dataloader()
    predict_dataloader = data_module.predict_dataloader()

    # 2. set model(=nn.Module class)
    model = config.init_obj('arch', module_arch)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 3. set deivce(cpu or gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 4. set loss function & matrics 
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    outputs, targets = [], []
    total_loss = 0
    with torch.no_grad():
        for (data, target) in tqdm(test_dataloader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            outputs.append(output)
            targets.append(target)

    outputs = torch.cat(outputs).squeeze()
    targets = torch.cat(targets).squeeze()
    result = {}
    result["val_loss"] = total_loss/len(test_dataloader)
    for metric in metrics:
        result[f"val_{metric.__name__}"] = metric(outputs, targets)
    
    output = pd.read_csv(config["data_module"]["args"]["test_path"])
    output['target'] = outputs.tolist()
    output.to_csv('output.csv', index=False)
    print(result)

    outputs = []
    with torch.no_grad():
        for data in tqdm(predict_dataloader):
            data = data.to(device)
            output = model(data)
            outputs.append(output)

    outputs = torch.cat(outputs).squeeze()
    output = pd.read_csv('./data/sample_submission.csv')
    output['target'] = outputs.tolist()
    output.to_csv('predict.csv', index=False)

if __name__ == '__main__':

    checkpoint_path = "./saved/STSModel_monologg-koelectra-base-v3-discriminator_val_pearson=0.9263085722923279.pth"
    main(checkpoint_path)
