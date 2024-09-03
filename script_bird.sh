#!/bin/bash

REPEAT=5
MODEL_PATH="Models/codegemma-7b-Q8_0.gguf"
RESULTS_DIR="Results/"
TOTAL=50
SIMPLE=20
MODERATE=30
CHALLENGING=50

# thresholds
EXACT=0.5
EXEC=0.5
CONSISTENCY=0.5

DB_DIR="Datasets/bird/databases"
DB_TABLE="Datasets/bird/dev_tables.json"
DATASET_TAG="_BIRD"
INPUT_JSON="input_list_BIRD.json"
DATASET_FILE="Datasets/bird/bird_dev_as_spider.json"

# python3 input_generator_bird.py \
#     --dataset $DATASET_FILE \
#     --dataset_tag $DATASET_TAG \
#     --repeat $REPEAT\
#     --total $TOTAL \
#     --simple $SIMPLE \
#     --moderate $MODERATE \
#     --challenging $CHALLENGING 

# CUDA_VISIBLE_DEVICES=0 python3 consistency_bird.py \
#     --modelpath $MODEL_PATH\
#     --dataset $DATASET_FILE \
#     --dbdir $DB_DIR\
#     --dbtable $DB_TABLE\
#     --dataset_tag $DATASET_TAG\
#     --inputjson $INPUT_JSON \
#     --output $RESULTS_DIR \
#     --exact_threshold $EXACT \
#     --exec_threshold $EXEC \
#     --consistency_threshold $CONSISTENCY 

INPUT_FILE="Results/newbird/codegemma-7b-Q8_0_Consistency_Results_BIRD.csv"
ITERATIONS=20

python3 ves_bird.py \
    --input_file $INPUT_FILE \
    --iterations $ITERATIONS