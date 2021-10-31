# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json
import math
import six
import torch
import paddle
import paddle.nn as nn
from paddle.nn.layer import CrossEntropyLoss
from paddlenlp.transformers import BertPretrainedModel, BertModel



class BertForSequenceClassification(BertPretrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, bert, num_classes=2, dropout=None, layers=None, pooling=None):
        super(BertForSequenceClassification, self).__init__()
        self.num_classes = num_classes
        self.bert = bert  # allow bert to be config
        self.dropout = nn.Dropout(dropout if dropout is not None else
                                  self.bert.config["hidden_dropout_prob"])
        self.pooling = pooling
        if layers is None or layers[0] == -2:
            self.layers = []
        elif layers[0] == -1:
            self.layers = [i for i in range(self.bert.config["num_hidden_layers"])]
        else:
            self.layers = layers
        self.classifier = nn.Linear(
            self.bert.config["hidden_size"] * max(1, len(self.layers) if self.pooling is None else 1), num_classes)

        self.apply(self.init_weights)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None,
                labels=None):
        r"""
        The BertForSequenceClassification forward method, overrides the __call__() special method.

        Args:
            input_ids (Tensor):
                See :class:`BertModel`.
            token_type_ids (Tensor, optional):
                See :class:`BertModel`.
            position_ids(Tensor, optional):
                See :class:`BertModel`.
            attention_mask (list, optional):
                See :class:`BertModel`.

        Returns:
            Tensor: Returns tensor `logits`, a tensor of the input text classification logits.
            Shape as `[batch_size, num_classes]` and dtype as float32.

        Example:
            .. code-block::

                import paddle
                from paddlenlp.transformers.bert.modeling import BertForSequenceClassification
                from paddlenlp.transformers.bert.tokenizer import BertTokenizer

                tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
                model = BertForSequenceClassification.from_pretrained('bert-base-cased')

                inputs = tokenizer("Welcome to use PaddlePaddle and PaddleNLP!")
                inputs = {k:paddle.to_tensor([v]) for (k, v) in inputs.items()}
                outputs = model(**inputs)

                logits = outputs[0]
        """

        encoded_layers, pooled_output = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask)

        if len(self.layers) > 0:
            hidden_state = []
            for l in self.layers:
                hidden_state.append(encoded_layers[l][:, 0].unsqueeze(1))
            hidden_state = paddle.concat(hidden_state, dim=1)  # shape: bs, layers_num, 768

            if self.pooling == 'max':
                hidden_state, _ = paddle.max(hidden_state, dim=1)
            elif self.pooling == 'mean':
                hidden_state = paddle.mean(hidden_state, dim=1)
            else:
                hidden_state = paddle.reshape(hidden_state, shape=[hidden_state.shape[0], -1])

            hidden_state = self.dropout(hidden_state)
            logits = self.classifier(hidden_state)
        else:
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits
        return logits
