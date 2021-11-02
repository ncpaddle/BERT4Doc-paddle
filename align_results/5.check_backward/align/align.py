from reprod_log import ReprodLogger, ReprodDiffHelper
import numpy as np


diff = ReprodDiffHelper()
paddle_diff = diff.load_info("../log_reprod/back_loss_paddle.npy")
torch_diff = diff.load_info("../log_reprod/back_loss_torch.npy")
diff.compare_info(paddle_diff, torch_diff)
diff.report(diff_method='mean', diff_threshold=1e-4, path='../log_diff/back_loss_log2.txt')
