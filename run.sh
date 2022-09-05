#!/bin/bash

#NUM_PROCESSES=20
CONFIG='config/config_test.yaml'
SCORES_FOLDER='results'

export PYTHONPATH=$(pwd):$PYTHONPATH

echo 'Stage 4: Train DNN'
python3 train.py -v \
    --config $CONFIG \
    --use_gpu

#echo 'Stage 5: Final scores'
#python3 test.py -v \
#    --config $CONFIG \
#    --use_gpu \
#    --scores_folder $SCORES_FOLDER
