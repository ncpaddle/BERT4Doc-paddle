from reprod_log import ReprodLogger, ReprodDiffHelper
import numpy as np
import paddle
import torch

diff = ReprodDiffHelper()
paddle_diff = diff.load_info("../log_reprod/lr_paddle.npy")
torch_diff = diff.load_info("../log_reprod/lr_torch.npy")
diff.compare_info(paddle_diff, torch_diff)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/lr_diff_log.txt')
