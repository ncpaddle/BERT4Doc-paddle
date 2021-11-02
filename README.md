# BERT4Doc-paddle

(Unofficial) **Paddle** implementation of `BERT4Doc`: [How to Fine-Tune BERT for Text Classification?](https://github.com/xuyige/BERT4doc-Classification)

Dataset: IMDB, TREC and yahoo-answers

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





TREC数据集

- 用作者进一步预训练的模型：0.93
- 用自己进一步预训练的模型：0.92

IMDB数据集

- 用作者进一步预训练的模型：0.94756
- 用自己进一步预训练的模型：







## Align

- `forward_diff`: [model_diff.txt](https://github.com/ncpaddle/BERT4Doc-paddle/blob/main/align_results/1.check_forward/log_diff/model_diff.txt)
- `metric_diff` and `loss_diff`: [metric_loss_diff.txt](https://github.com/ncpaddle/BERT4Doc-paddle/blob/main/align_results/3.check_metric_loss/log_diff/metric_diff_log.txt)
- `learning_rate_diff`: [lr_diff.txt](https://github.com/ncpaddle/BERT4Doc-paddle/blob/main/align_results/4.check_lr/log_diff/lr_diff_log.txt)
- `backward_diff`: [backward_loss_diff.txt](https://github.com/ncpaddle/BERT4Doc-paddle/blob/main/align_results/5.check_backward/log_diff/back_loss_log2.txt)

More details about align works in [here](https://github.com/ncpaddle/JointBERT-paddle/tree/main/align_works).

