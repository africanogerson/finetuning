#!/bin/bash

NUM_PROCESSES=20
CONFIG='config/configheatKIOTO.yaml'
SCORES_FOLDER='results'

export PYTHONPATH=$(pwd):$PYTHONPATH

#echo 'Stage 1: Crop Mammograms'
#python3 src/gen_info_and_cropping.py \
#    --cases-data $CASES_DATA \
#    --controls-data $CONTROLS_DATA \
#    --output-data-folder $CROPPED_IMAGE_FOLDER \
#    --cases-exam-list-path $CASES_INITIAL_EXAM_LIST_PATH  \
#    --controls-exam-list-path $CONTROLS_INITIAL_EXAM_LIST_PATH  \
#    --cropped-exam-list-file $CROPPED_EXAM_LIST_FILE  \
#    --num-processes $NUM_PROCESSES

#echo 'Stage 2: Compute average crop sizes'
#python3 src/get_average_crops.py \
#    --cases-data $CASES_DATA \
#    --controls-data $CONTROLS_DATA \
#    --cropped-image-folder $CROPPED_IMAGE_FOLDER \
#    --output-file $AVG_CROPS_PATH
#
#echo 'Stage 3: Extract Centers'
#python3 breast_cancer_classifier/optimal_centers/get_optimal_centers.py \
#    --cropped-exam-list-path "${CROPPED_IMAGE_FOLDER}/cases/${CROPPED_EXAM_LIST_FILE}" \
#    --data-prefix "${CROPPED_IMAGE_FOLDER}/cases" \
#    --output-exam-list-path $CASES_EXAM_LIST_PATH \
#    --num-processes $NUM_PROCESSES
#python3 breast_cancer_classifier/optimal_centers/get_optimal_centers.py \
#    --cropped-exam-list-path "${CROPPED_IMAGE_FOLDER}/controls/${CROPPED_EXAM_LIST_FILE}" \
#    --data-prefix "${CROPPED_IMAGE_FOLDER}/controls" \
#    --output-exam-list-path $CONTROLS_EXAM_LIST_PATH \
#    --num-processes $NUM_PROCESSES

echo 'Stage 4: Train DNN'
python3 train.py -v \
    --config $CONFIG \
    --use_gpu

#echo 'Stage 5: Final scores'
#python3 test.py -v \
#    --config $CONFIG \
#    --use_gpu \
#    --scores_folder $SCORES_FOLDER
