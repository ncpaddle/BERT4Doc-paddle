# BERT4Doc-paddle

- (Unofficial) The repo is **Paddle** implementation of `BERT4Doc`.
- paper: [How to Fine-Tune BERT for Text Classification?](https://github.com/xuyige/BERT4doc-Classification)
- unofficial pytorch implementation: [xuyige/BERT4doc-Classification: Code and source for paper ](https://github.com/xuyige/BERT4doc-Classification)

- Dataset: IMDB, TREC and yahoo-answers

## Dependencies

- python>=3.6

- paddle == 2.1.3

- paddlenlp == 2.0.0

  

## Further Pre-training

1. Executing further pre-training based on IMDB dataset:

```bash
python main.py \
  --data_dir data/imdb_pretraining.json \
  --model_dir further_imdb_pretraining \
  --max_steps 100000 \
  --model_name_or_path bert-base-uncased
```

2. Executing further pre-training based on yahoo-answers dataset:

```bash
python main.py \
  --data_dir data/yahoo_pretraining.json \
  --model_dir further_imdb_pretraining \
  --max_steps 100000 \
  --model_name_or_path bert-base-uncased
```

You can download models trained by us in [here](https://drive.google.com/drive/folders/1_YDaG37w8EpVhiRIMsNPn00JYYUsULOU?usp=sharing). 



## Fine-tuning

1. Using the pre-training model based on IMDB to fine-tuning IMDB dataset;

```bash
python run_discriminative_paddle_decay.py \
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
```

2. Using the pre-traning model based on yahoo-answers to fine-tuning TREC dataset;

```bash
python run_discriminative_paddle_decay.py \
                          --data_dir="TREC_data" \
                          --task_name="TREC" \
                          --output_dir="trec_output" \
                          --model_name_or_path="furthered_trec_pretrained" \
                          --model_dir="trec_model" \
                          --do_lower_case \
                          --do_train --do_eval --discr\
                          --layers 11 \
                          --trunc_medium 128 \
                          --layer_learning_rate 2e-5 \
                          --layer_learning_rate_decay 0.95
```



## Experiment Results



| Further pre-training Dataset | Fine-tuning Dataset | Accuracy |
| ---------------------------- | ------------------- | -------- |
| IMDB                         | IMDB                | 94,76    |
| Yah. A                       | TREC                | 93.00    |



## Align

- `forward_diff`: [model_diff.txt](https://github.com/ncpaddle/BERT4Doc-paddle/blob/main/align_results/1.check_forward/log_diff/model_diff.txt)
- `metric_diff` and `loss_diff`: [metric_loss_diff.txt](https://github.com/ncpaddle/BERT4Doc-paddle/blob/main/align_results/3.check_metric_loss/log_diff/metric_diff_log.txt)
- `learning_rate_diff`: [lr_diff.txt](https://github.com/ncpaddle/BERT4Doc-paddle/blob/main/align_results/4.check_lr/log_diff/lr_diff_log.txt)
- `backward_diff`: [backward_loss_diff.txt](https://github.com/ncpaddle/BERT4Doc-paddle/blob/main/align_results/5.check_backward/log_diff/back_loss_log2.txt)

More details about align works in [here](https://github.com/ncpaddle/JointBERT-paddle/tree/main/align_works).

