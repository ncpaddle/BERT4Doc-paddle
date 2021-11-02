import paddle
import paddle.io
import paddlenlp
from paddle.io import DataLoader
from paddle.optimizer import AdamW
from paddlenlp.transformers import LinearDecayWithWarmup

import os
import logging
from tqdm import tqdm, trange

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset):
        self.args = args
        self.train_dataset = train_dataset
        self.model = paddlenlp.transformers.BertForPretraining.from_pretrained('bert-base-uncased'
                                                                               if not args.model_name_or_path else
                                                                               args.model_name_or_path)

    def gather_indexes(self, positions):
        flat_offsets = paddle.reshape(
            paddle.arange(0, positions.shape[0], dtype=paddle.int32) * 128, [-1, 1])
        flat_positions = paddle.reshape(positions + flat_offsets, [-1])
        return flat_positions

    def train(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, shuffle=False)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            p.name for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)
        ]

        scheduler = LinearDecayWithWarmup(learning_rate=self.args.learning_rate, warmup=int(self.args.warmup_steps),
                                          total_steps=t_total)

        clip = paddle.nn.ClipGradByNorm(clip_norm=self.args.max_grad_norm)
        optimizer = AdamW(
            parameters=self.model.parameters(),
            apply_decay_param_fun=lambda x: x in optimizer_grouped_parameters,
            learning_rate=scheduler,
            epsilon=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
            grad_clip=clip)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        optimizer.clear_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                inputs = {'input_ids': batch[0],
                          'attention_mask': paddle.unsqueeze(batch[1], [1, 2]),
                          'token_type_ids': batch[2],
                          'masked_positions': self.gather_indexes(batch[3])}

                labels = batch[4]
                next_sentence_label = batch[5]

                prediction_scores, seq_relationship_score = self.model(**inputs)
                # print(prediction_scores.shape, seq_relationship_score.shape)

                total_loss = None
                if labels is not None and next_sentence_label is not None:
                    loss_fct = paddle.nn.CrossEntropyLoss()
                    masked_lm_loss = loss_fct(prediction_scores.reshape([-1, prediction_scores.shape[-1]]), labels.reshape([-1]))
                    next_sentence_loss = loss_fct(seq_relationship_score.reshape([-1, 2]), next_sentence_label.reshape([-1]))
                    total_loss = masked_lm_loss + next_sentence_loss

                    print(total_loss)

                    if self.args.gradient_accumulation_steps > 1:
                        total_loss = total_loss / self.args.gradient_accumulation_steps

                    total_loss.backward()

                    tr_loss += total_loss.item()
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:

                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        optimizer.clear_grad()
                        global_step += 1

                        if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                            self.save_model()
                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break
        return global_step, tr_loss / global_step

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        paddle.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)