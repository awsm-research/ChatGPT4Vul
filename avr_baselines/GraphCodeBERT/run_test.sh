python run.py \
--beam_size 1 \
--model_name fine_tuned_model.bin \
--do_test \
--model_type roberta \
--source_lang c_sharp \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--test_filename ../../data/avr/test.csv \
--output_dir ./saved_models \
--max_source_length 512 \
--max_target_length 256 \
--eval_batch_size 1 \
--seed 123456  2>&1 | tee beam1_test.log

python run.py \
--beam_size 3 \
--model_name fine_tuned_model.bin \
--do_test \
--model_type roberta \
--source_lang c_sharp \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--test_filename ../../data/avr/test.csv \
--output_dir ./saved_models \
--max_source_length 512 \
--max_target_length 256 \
--eval_batch_size 1 \
--seed 123456  2>&1 | tee beam3_test.log

python run.py \
--beam_size 5 \
--model_name fine_tuned_model.bin \
--do_test \
--model_type roberta \
--source_lang c_sharp \
--model_name_or_path microsoft/graphcodebert-base \
--tokenizer_name microsoft/graphcodebert-base \
--config_name microsoft/graphcodebert-base \
--test_filename ../../data/avr/test.csv \
--output_dir ./saved_models \
--max_source_length 512 \
--max_target_length 256 \
--eval_batch_size 1 \
--seed 123456  2>&1 | tee beam5_test.log