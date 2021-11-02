import json
import paddle
from paddle.io import Dataset



class DataReader(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args
        (self.input_ids, self.attention_mask, self.token_type_ids,
         self.masked_positions, self.masked_lm_ids, self.next_sentence_labels) = self.process()

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.token_type_ids[idx], \
         self.masked_positions[idx], self.masked_lm_ids[idx], self.next_sentence_labels[idx]

    def __len__(self):
        return len(self.input_ids)

    def process(self):
        with open(self.args.data_dir, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data
        input_ids = []
        attention_mask = []
        token_type_ids = []
        masked_positions = []
        masked_lm_ids = []
        next_sentence_labels = []
        for d in data:
            input_ids.append(d['input_ids'])
            attention_mask.append(d['input_mask'])
            token_type_ids.append(d['segment_ids'])
            masked_positions.append(d['masked_lm_positions'])
            masked_lm_ids.append(d['masked_lm_ids'])
            next_sentence_labels.append(d['next_sentence_labels'])

        return paddle.to_tensor(input_ids), \
               paddle.to_tensor(attention_mask), \
               paddle.to_tensor(token_type_ids), \
               paddle.to_tensor(masked_positions), \
               paddle.to_tensor(masked_lm_ids), \
               paddle.to_tensor(next_sentence_labels)
