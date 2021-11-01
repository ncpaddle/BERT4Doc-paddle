from reprod_log import ReprodLogger, ReprodDiffHelper
import numpy as np


diff = ReprodDiffHelper()
paddle_diff = diff.load_info("train_data_paddle.npy")
torch_diff = diff.load_info("train_data_torch.npy")
diff.compare_info(paddle_diff, torch_diff)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/train_data_diff_log.txt')