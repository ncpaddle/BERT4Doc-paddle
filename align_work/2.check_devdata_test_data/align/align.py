from reprod_log import ReprodLogger, ReprodDiffHelper


diff = ReprodDiffHelper()
paddle_diff = diff.load_info("../log_reprod/paddle_devdata.npy")
torch_diff = diff.load_info("../log_reprod/torch_devdata.npy")
diff.compare_info(paddle_diff, torch_diff)
diff.report(diff_method='mean', diff_threshold=1e-5, path='../log_diff/dev_diff.txt')





