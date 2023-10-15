python train_baseline_transformers.py \
    --model_name=statement_codet5.bin \
    --output_dir=./saved_models \
    --do_train \
    --do_test \
    --train_data_file=../../data/svp/processed_train.csv \
    --eval_data_file=../../data/svp/processed_val.csv \
    --test_data_file=../../data/svp/processed_test.csv \
    --epochs 10 \
    --encoder_block_size 512 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 1e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train_baseline_codet5_base_big_vul.log