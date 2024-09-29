#!/bin/bash

HUGGINGFACE_TOKEN=""
PROJECT_NAME="jeongwan"
SWEEP_CONFIG="sweep.yaml"
COUNT=2

use_sweep=true

# 1. login huggingface cli
#huggingface-cli login --token ${HUGGINGFACE_TOKEN}

# 2. install requirements
#pip3 install -r requirements.txt

# 3. train model (by wandb sweep or wandb run)
if ["${use_sweep}" = true]; then
    SWEEP_OUTPUT=$(wandb sweep -p ${PROJECT_NAME} ${SWEEP_CONFIG} 2>&1)
    SWEEP_ID=$(echo ${SWEEP_OUTPUT} | grep -o "wandb agent .*" | cut -d' ' -f3-)
    wandb agent --count ${COUNT} ${SWEEP_ID}
else
    python3 train.py

# 4. run inference.py