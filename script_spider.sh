#!/bin/bash

REPEAT=5
MODEL_PATH="Models/codegemma-7b-Q8_0.gguf"
RESULTS_DIR="Results/"
TOTAL=10
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
DATASET_TAG="_SPIDER"
INPUT_JSON="input_list_SPIDER.json"
DATASET_FILE="Datasets/spider/dev_with_hardness.json"
OUTPUT="input_list_SPIDER.json"

python3 input_generator_spider.py \
    --dataset $DATASET_FILE \
    --repeat $REPEAT\
    --total $TOTAL \
    --easy $EASY \
    --medium $MEDIUM \
    --hard $HARD \
    --extra $EXTRA_HARD \
    --output $OUTPUT 

# CUDA_VISIBLE_DEVICES=0 python3 consistency_spider.py \
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

# INPUT_FILE="Results/codegemma-7b-Q8_0_Consistency_Results_SPIDER.csv"
# ITERATIONS=20

# python3 ves.py \
#     --input_file $INPUT_FILE \
#     --iterations $ITERATIONS