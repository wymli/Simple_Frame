from main_parser import get_basic_parser
from models import netWrapper
import pandas as pd
import json
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
from util.logger import Logger
import time
from util.config_from_dict import getConfig


def getDate():
    ret = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
    return ret


class dataLoader_provider:
    def __init__(self, dataset_path, split_path, batch_size):
        self.batch_size = batch_size
        self.split_path = split_path
        self.dataset_path = dataset_path
        if split_path == None:
            self.split_path = dataset_path.replace("csv", "json")

        self.splits = self.get_splits_indices()

    def get_splits_indices(self):
        with open(self.split_path, "r") as fp:
            splits = json.load(fp)
        return splits

    def get_loader_one_fold(self, idx: int):
        indices = self.splits[idx]
        testSet_indices = indices["test"]
        # 这里的0是holdout,即内部不做cv,inner_kfold=None
        trainSet_indices = indices["train"]

        df = pd.read_csv(self.dataset_path)
        # y 这里的dataset是.csv的,没有特征的,需要函数处理得到真正的输入!
        # y loader出来的是data,用点访问键 , 详情见模型的写法
        # y 图卷积的写法:
        # y x, edge_index, batch = data.x, data.edge_index, data.batch
        # y label = data.y

        train_loader = DataLoader(
            df[trainSet_indices], self.batch_size, shuffle=True)
        test_loader = DataLoader(
            df[testSet_indices], self.batch_size, shuffle=True)
        return train_loader, test_loader


def kfold_train_test(model_name, dataset_path, split_path=None,  outer_kfold=5,  args=None):
    log_folder = args["result_folder"]
    if log_folder == None:
        log_folder = f"{model_name}_log_{getDate()}"
        os.mkdir(log_folder)

    model = getConfig("models", model_name)(
        args["dim_features"], args["dim_target"], args)
    optimizer = getConfig("optimizers", args["optimizer"])(model.parameters(),
                                                           lr=args['learning_rate'], weight_decay=args['l2'])
    loss_fn = getConfig("losses", args["loss"])()

    loader_provider = dataLoader_provider(
        dataset_path, split_path, args["batch_size"])
    net = netWrapper.NetWrapper(
        model, loss_fn, device=args["device"], classification=True, metric_type=args["metric_type"])
    metrics = []

    # losses = []
    for i in range(outer_kfold):
        train_loader, _, test_loader = loader_provider.get_loader_one_fold(i)
        metric, loss = net.train_test_one_fold(train_loader, test_loader, max_epochs=100, optimizer=optimizer, scheduler=None, clipping=None,
                                               early_stopping=None, logger=Logger(os.path.join(log_folder, f"k_fold_{i}")), log_every=10)
        metrics.append(metric)
        # losses.append(loss)
    metric_mean = np.array(metrics).mean()
    metric_std = np.array(metrics).std()
    result = Logger(os.path.join(args.result_folder, "final_metric_mean_std"))
    result.log(f"Mean: {metric_mean}, Std: {metric_std}").flush()


def main():
    cli_args = get_basic_parser()
    json_args = json.load(open(cli_args.config_path, "r"))
    args = cli_args.__dict__.update(json_args)
    kfold_train_test(args["model_name"], args["dataset_path"],
                     args["split_path"], args["outer_kfold"], args)


if __name__ == "__main__":
    main()
