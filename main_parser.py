import argparse
import re
import torch


def check_device(arg):
    if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
        raise argparse.ArgumentTypeError(
            'Wrong device format: {}'.format(arg)
        )

    if arg != 'cpu':
        splited_device = arg.split(':')

        if (not torch.cuda.is_available()) or \
                (len(splited_device) > 1 and
                    int(splited_device[1]) > torch.cuda.device_count()):
            raise argparse.ArgumentTypeError(
                'Wrong device: {} is not available'.format(arg)
            )

    return arg


def get_basic_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--model_name',
    #                     type=str, required=True,
    #                     help='fileName of model class')
    parser.add_argument('--config_path',
                        type=str, required=True,
                        help='hyperParameters config_path')
    parser.add_argument('--dataset_path',
                        type=str, required=True,
                        help='Where to load DataSet , filetype:.csv')
    parser.add_argument('--split_path',
                        type=str, required=False,
                        help='cross_val splits.json')
    parser.add_argument('--result_folder',
                        type=str, required=False,
                        help='Where to save metric mean_std')
    parser.add_argument('--outer_kfold',
                        type=int, required=False,
                        help='Outer kfold')
    # parser.add_argument('--is_cv',
    #                     type=str, required=False, default="True",
    #                     help='Which mode to switch, cv or not')
    parser.add_argument('--metric_type',
                        type=str, required=False, default="auc",
                        help='For assessment')

    return parser.parse_args()
