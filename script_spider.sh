#!/bin/bash

REPEAT=7
MODEL_PATH="Models/Codestral-22B-v0.1-Q8_0.gguf"
RESULTS_DIR="Results/Spider/"
TOTAL=1000
EASY=10
MEDIUM=20
HARD=30
EXTRA_HARD=40

# thresholds
EXACT=0.5
EXEC=0.5
CONSISTENCY=0.5

DB_DIR="Datasets/spider/database"
DB_TABLE="Datasets/spider/tables.json"
INPUT_JSON="input_list_SPIDER.json"
DATASET_FILE="Datasets/spider/dev_with_hardness.json"

# python3 input_generator_spider.py \
#     --dataset $DATASET_FILE \
#     --repeat $REPEAT\
#     --total $TOTAL \
#     --easy $EASY \
#     --medium $MEDIUM \
#     --hard $HARD \
#     --extra $EXTRA_HARD \
#     --output $INPUT_JSON 

# CUDA_VISIBLE_DEVICES=0 python3 consistency_spider.py \
#     --modelpath $MODEL_PATH\
#     --dataset $DATASET_FILE \
#     --dbdir $DB_DIR\
#     --dbtable $DB_TABLE\
#     --inputjson $INPUT_JSON \
#     --output $RESULTS_DIR \
#     --exact_threshold $EXACT \
#     --exec_threshold $EXEC \
#     --consistency_threshold $CONSISTENCY 

INPUT_FILE="Results/Spider/Codestral-22B-v0.1-Q8_0Codestral-22B-v0.1-Q8_0_Consistency_Results_SPIDER.csv"
ITERATIONS=100

python3 ves_spider.py \
    --input_file $INPUT_FILE \
    --iterations $ITERATIONS