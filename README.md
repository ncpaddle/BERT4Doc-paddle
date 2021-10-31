# BERT4Doc
BERT4Doc的paddlepaddle实现

问题：

- paddlenlp的BERTLMPredictionHead里的masked_positions问题；和tensorflow实现不一样。而huggingface版的没有这个参数
- 数据集问题 IMDB 是句子级，不知道怎么做MLM和NSP任务进行预训练

