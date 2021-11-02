python run_classifier_single_layer_paddle.py \
                          --data_dir="IMDB_data" \
                          --task_name="IMDB" \
                          --output_dir="imdb_output" \
                          --model_name_or_path="furthered_imdb_pretrained" \
                          --model_dir="imdb_model" \
                          --do_lower_case \
                          --do_train --do_eval --discr\
                          --layers 11 \
                          --trunc_medium 128 \
                          --layer_learning_rate 2e-5 \
                          --layer_learning_rate_decay 0.95






