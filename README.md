# BERT4Doc-paddle

(Unofficial) **Paddle** implementation of `BERT4Doc`: [How to Fine-Tune BERT for Text Classification?](https://github.com/xuyige/BERT4doc-Classification)

Dataset: IMDB, TREC and yahoo-answers

## Model Architecture

![image-20211102160155873](C:\Users\10594\AppData\Roaming\Typora\typora-user-images\image-20211102160155873.png)

## Dependencies

- python>=3.6

- paddle == 2.1.3

- paddlenlp == 2.0.0

  

## Further Pre-training

1. Executing further pre-training based on IMDB dataset:
2. Executing further pre-training based on yahoo-answers dataset:



## Fine-tuning

1. Using the pre-training model based on IMDB to fine-tuning IMDB dataset;
2. Using the pre-traning model based on yahoo-answers to fine-tuning TREC dataset;



## Experiment Results



## Align

- `forward_diff`: [model_diff.txt](https://github.com/ncpaddle/JointBERT-paddle/blob/main/align_works/1_check_forward/log_diff/model_diff.txt)
- `metric_diff` and `loss_diff`: [metric_loss_diff.txt](https://github.com/ncpaddle/JointBERT-paddle/blob/main/align_works/3_4_check_metric_loss/log_diff/metric_diff_log.txt)
- `backward_diff`: [backward_loss_diff.txt](https://github.com/ncpaddle/JointBERT-paddle/blob/main/align_works/5-7-8_check_optim-norm-backward/log_diff/loss_diff.txt)
- `train_align`: experiment results

More details about align works in [here](https://github.com/ncpaddle/JointBERT-paddle/tree/main/align_works).

