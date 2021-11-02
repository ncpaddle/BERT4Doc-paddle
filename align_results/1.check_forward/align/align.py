from codes.fine_tuning.modeling_single_layer import BertForSequenceClassification as torch_BertForSequenceClassification, BertConfig
from codes.fine_tuning.modeling_single_layer_paddle import BertForSequenceClassification as paddle_BertForSequenceClassification
import json
import paddle
import torch
from reprod_log import ReprodLogger, ReprodDiffHelper

with open("../../fake_data.json", 'r', encoding='utf-8') as f:
    input_data = json.load(f)

paddle_input = {k: paddle.to_tensor(v) for k, v in input_data.items()}
paddle_model = paddle_BertForSequenceClassification.from_pretrained('../../../furthered_imdb_pretrained')
paddle_model.eval()
paddle_out = paddle_model(**paddle_input)


torch_input = {k: torch.tensor(v) for k, v in input_data.items()}
torch_input['attention_mask'] = torch_input['attention_mask'].squeeze(1).squeeze(1)
config = BertConfig.from_json_file('../../../furthered_imdb_pretrained/model.json')
torch_model = torch_BertForSequenceClassification(config, num_labels=2)
torch_model.load_state_dict(torch.load('../../furthered_imdb_pretrained/pytorch_model.bin'))
torch_model.eval()
torch_out = torch_model(**torch_input)

lr_paddle = ReprodLogger()
lr_paddle.add('output_logits', paddle_out.numpy())
lr_paddle.save("forward_paddle.npy")

lr_torch = ReprodLogger()
lr_torch.add('output_logits', torch_out.detach().numpy())
lr_torch.save("forward_torch.npy")

diff = ReprodDiffHelper()
paddle_diff = diff.load_info("forward_paddle.npy")
torch_diff = diff.load_info("forward_torch.npy")
diff.compare_info(paddle_diff, torch_diff)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/model_diff.txt')





