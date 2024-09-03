set -e

device="1"

REPEAT=5
RESULTS_DIR="Results/"

# thresholds
EXACT=0.5
EXEC=0.5
CONSISTENCY=0.5
ITERATIONS=20

tables_for_natsql="./data/preprocessed_data/test_tables_for_natsql.json"

if [ $1 = "base" ]
then
    model_path="./models/text2natsql-t5-base/checkpoint-14352"
    text2natsql_model_bs=4
elif [ $1 = "large" ]
then
    model_path="./models/text2natsql-t5-large/checkpoint-21216"
    text2natsql_model_bs=4
elif [ $1 = "3b" ]
then
    model_path="./models/text2natsql-t5-3b/checkpoint-78302"
    text2natsql_model_bs=6
else
    echo "The first arg must in [base, large, 3b]."
    exit
fi


if [ $2 = "spider" ]
then
    # spider's dev set
    table_path="./data/spider/tables.json"
    db_path="../Datasets/spider/database/"
    MODEL="RESDSQL_NATSQL_SPIDER"
    INPUT_LIST="../input_list_SPIDER.json"
    original_dev="./data/spider/dev_with_hardness.json"
    RESULTS_DIR="Results/Spider"
elif [ $2 = "bird" ]
then
    table_path="../Datasets/bird/dev_tables.json"
    db_path="../Datasets/bird/databases"
    MODEL="RESDSQL_NATSQL_BIRD"
    INPUT_LIST="../input_list_BIRD.json"
    original_dev="../Datasets/bird/bird_dev_as_spider.json"
    RESULTS_DIR="Results/Bird"
else
    echo "The second arg must in [spider, bird]."
    exit
fi

# echo prepare table file for natsql
# python NatSQL/table_transform.py \
#     --in_file $table_path \
#     --out_file $tables_for_natsql \
#     --correct_col_type \
#     --remove_start_table  \
#     --analyse_same_column \
#     --table_transform \
#     --correct_primary_keys \
#     --use_extra_col_types \
#     --db_path $db_path

# echo preprocess test set
# python preprocessing.py \
#     --mode "test" \
#     --table_path $table_path \
#     --input_dataset_path "${RESULTS_DIR}/preprocessed_final_list_RESDSQL_BIRD.json" \
#     --output_dataset_path "${RESULTS_DIR}/natsql_preprocessed_dev_${MODEL}.json" \
#     --db_path $db_path \
#     --target_type "sql" 

# echo predict probability for each schema item
# python schema_item_classifier.py \
#     --batch_size 32 \
#     --device $device \
#     --seed 42 \
#     --save_path "./models/text2sql_schema_item_classifier" \
#     --dev_filepath "${RESULTS_DIR}/natsql_preprocessed_dev_${MODEL}.json" \
#     --output_filepath "${RESULTS_DIR}/natsql_test_with_probs_resdsql_${MODEL}.json" \
#     --use_contents \
#     --add_fk_info \
#     --mode "test"

# echo generate text2sql test set
# python text2sql_data_generator.py \
#     --input_dataset_path "${RESULTS_DIR}/natsql_test_with_probs_resdsql_${MODEL}.json" \
#     --output_dataset_path "${RESULTS_DIR}/natsql_test_${MODEL}.json" \
#     --topk_table_num 4 \
#     --topk_column_num 5 \
#     --mode "test" \
#     --use_contents \
#     --add_fk_info \
#     --output_skeleton \
#     --target_type "sql"

# echo inference using the best text2sql ckpt
# python text2sql.py \
#     --batch_size 1 \
#     --device 1 \
#     --seed 42 \
#     --save_path $model_path \
#     --mode "eval" \
#     --dev_filepath "${RESULTS_DIR}/natsql_test_${MODEL}.json" \
#     --original_dev_filepath $original_dev \
#     --db_path $db_path \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --target_type "sql" \
#     --output "${RESULTS_DIR}/natsql_output.json"

# echo preparing for consistency
# python3 prepare_for_consistency.py\
#     --input "${RESULTS_DIR}/natsql_output.json"\
#     --original_dev $original_dev \
#     --output "${RESULTS_DIR}/natsql_output_for_consistency_${MODEL}.json"

if [ $2 = "spider" ]
then
    CUDA_VISIBLE_DEVICES=0 python3 consistency_spider.py \
    --model $MODEL\
    --dbdir $db_path\
    --dbtable $table_path\
    --inputjson "${RESULTS_DIR}/natsql_output_for_consistency_${MODEL}.json"\
    --output $RESULTS_DIR \
    --exact_threshold $EXACT \
    --exec_threshold $EXEC \
    --consistency_threshold $CONSISTENCY \
    --repeat $REPEAT
    python3 ves_spider.py \
        --model $MODEL \
        --input_file "${RESULTS_DIR}/natsql_output_for_consistency_${MODEL}.json" \
        --output "$RESULTS_DIR/" \
        --iterations $ITERATIONS
elif [ $2 = "bird" ]
then
    CUDA_VISIBLE_DEVICES=0 python3 consistency_bird.py \
    --model $MODEL\
    --dbdir $db_path\
    --dbtable $table_path\
    --inputjson "${RESULTS_DIR}/natsql_output_for_consistency_${MODEL}.json"\
    --output "${RESULTS_DIR}/" \
    --exact_threshold $EXACT \
    --exec_threshold $EXEC \
    --consistency_threshold $CONSISTENCY \
    --repeat $REPEAT
    python3 ves_bird.py \
    --model $MODEL \
    --input_file "${RESULTS_DIR}/natsql_output_for_consistency_${MODEL}.json" \
    --output "${RESULTS_DIR}/" \
    --iterations $ITERATIONS
else
    echo "The second arg must in [spider, bird]."
    exit
fi