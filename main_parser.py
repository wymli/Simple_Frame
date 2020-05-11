import argparse
import re
import torch
import os

# def check_device(arg):
#     if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
#         raise argparse.ArgumentTypeError(
#             'Wrong device format: {}'.format(arg)
#         )

#     if arg != 'cpu':
#         splited_device = arg.split(':')

#         if (not torch.cuda.is_available()) or \
#                 (len(splited_device) > 1 and
#                     int(splited_device[1]) > torch.cuda.device_count()):
#             raise argparse.ArgumentTypeError(
#                 'Wrong device: {} is not available'.format(arg)
#             )

#     return arg


# def check_taskType(arg):
#     return "classification" if str(arg)[0] == 'c' else 'regression'




def get_basic_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str, required=True,
                        help='fileName of model class')
    parser.add_argument('--dataset_path',
                        type=str, required=True,
                        help='Where to load DataSet , filetype:.csv')
    parser.add_argument('--model_config',
                        type=str, required=True,
                        help='hyperParameters config_path')
    parser.add_argument('--task_type',
                        type=str, required=False, default="classification",
                        help='classification or regression')
    parser.add_argument('--multi_label',
                        type=int, required=False, default=1,
                        help='is multi_label or not')
    parser.add_argument('--split_path',
                        type=str, required=False,
                        help='cross_val splits.json')
    parser.add_argument('--result_folder',
                        type=str, required=False,default=None,
                        help='Where to save metric mean_std')
    parser.add_argument('--k_fold',
                        type=int, required=False, default=5,
                        help='Outer kfold')
    parser.add_argument('--metric_type',
                        type=str, required=False, default="auc",
                        help='auc-roc')

    return parser.parse_args()

# def check_args(args):
#     if any( not [os.path.exists(file) for file in [args.dataset_path , args.model_config,args.split_path]]):
#         raise "file not existed"

