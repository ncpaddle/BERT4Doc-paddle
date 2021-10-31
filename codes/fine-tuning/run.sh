python run_classifier_single_layer_paddle.py \
                          --data_dir="../IMDB_data" \
                          --task_name="IMDB" \
                          --output_dir="../imdb_model" \
                          --model_name_or_path="furthered_imdb_pretrained" \
                          --do_lower_case \
                          --do_train --do_eval \
