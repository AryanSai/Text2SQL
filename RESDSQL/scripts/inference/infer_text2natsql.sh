set -e

device="1"

REPEAT=5
RESULTS_DIR="Results/"

# thresholds
EXACT=0.5
EXEC=0.5
CONSISTENCY=0.5

DB_DIR="./database"
DB_TABLE="./data/spider/tables.json"
MODEL="RESDSQL_NATSQL"

tables_for_natsql="./data/preprocessed_data/test_tables_for_natsql.json"

if [ $1 = "base" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-base/checkpoint-14352"
    text2natsql_model_bs=4
elif [ $1 = "large" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-large/checkpoint-21216"
    text2natsql_model_bs=4
elif [ $1 = "3b" ]
then
    text2natsql_model_save_path="./models/text2natsql-t5-3b/checkpoint-78302"
    text2natsql_model_bs=6
else
    echo "The first arg must in [base, large, 3b]."
    exit
fi

model_name="resdsql_$1_natsql"

if [ $2 = "spider" ]
then
    # spider's dev set
    table_path="./data/spider/tables.json"
    input_dataset_path="./data/spider/consistency.json"
    db_path="./database"
    output="./predictions/Spider-dev/$model_name/spider_consistency.sql"
elif [ $2 = "bird" ]
then
    table_path="Datasets/bird/dev_tables.json"
    input_dataset_path="Datasets/bird/bird_dev_as_spider.json"
    db_path="Datasets/bird/databases"
    output="Results/$model_name/pred_nat.sql"

else
    echo "The second arg must in [spider, bird]."
    exit
fi

# # # prepare table file for natsql
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

# # # preprocess test set
# python preprocessing.py \
#     --mode "test" \
#     --table_path $table_path \
#     --input_dataset_path 'preprocessed_final_list.json' \
#     --output_dataset_path 'preprocessed_dev.json' \
#     --db_path $db_path \
#     --target_type "sql" 

# # # predict probability for each schema item
# python schema_item_classifier.py \
#     --batch_size 32 \
#     --device $device \
#     --seed 42 \
#     --save_path "./models/text2sql_schema_item_classifier" \
#     --dev_filepath 'preprocessed_dev.json' \
#     --output_filepath "test_with_probs_natsql.json" \
#     --use_contents \
#     --add_fk_info \
#     --mode "test"

# # generate text2sql test set
# python text2sql_data_generator.py \
#     --input_dataset_path "test_with_probs_natsql.json" \
#     --output_dataset_path "natsql_test.json" \
#     --topk_table_num 4 \
#     --topk_column_num 5 \
#     --mode "test" \
#     --use_contents \
#     --add_fk_info \
#     --output_skeleton \
#     --target_type "sql"

# inference using the best text2sql ckpt
# python text2sql.py \
#     --batch_size 2 \
#     --device 1 \
#     --seed 42 \
#     --save_path $text2natsql_model_save_path \
#     --mode "eval" \
#     --dev_filepath "natsql_test.json" \
#     --original_dev_filepath './data/spider/dev_with_hardness.json' \
#     --db_path $db_path \
#     --num_beams 8 \
#     --num_return_sequences 8 \
#     --target_type "sql" \
#     --output 'natqsl_output.json'

# python3 prepare_for_consistency.py\
#     --input 'natqsl_output.json'\
#     --original_dev './data/spider/dev_with_hardness.json' \
#     --output 'natqsl_output_for_consistency.json'\

# CUDA_VISIBLE_DEVICES=0 python3 consistency_resdsql.py \
#     --model $MODEL \
#     --dbdir $DB_DIR\
#     --dbtable $DB_TABLE\
#     --inputjson 'natqsl_output_for_consistency.json'\
#     --output $RESULTS_DIR \
#     --exact_threshold $EXACT \
#     --exec_threshold $EXEC \
#     --consistency_threshold $CONSISTENCY \
#     --repeat $REPEAT

ITERATIONS=20

python3 ves.py \
    --model $MODEL \
    --input_file 'natqsl_output_for_consistency.json' \
    --iterations $ITERATIONS