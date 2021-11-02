import argparse

from trainer import Trainer
from data_loader import DataReader
from paddle.io import Dataset


def main(args):

    train_dataset = DataReader(args)
    trainer = Trainer(args, train_dataset)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="yahoo_pretraining.json", type=str, help="The input data dir")
    parser.add_argument("--model_dir", default=None, required=True, type=str, help="Path to save, load model")
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=10, type=int)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--max_steps", default=-1, type=int, help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--logging_steps', type=int, default=200, help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=200, help="Save checkpoint every X updates steps.")
    parser.add_argument("--model_name_or_path", default="", type=str, help="")

    args = parser.parse_args()
    main(args)