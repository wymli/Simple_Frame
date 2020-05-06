from main_parser import get_basic_parser
from models import netWrapper
import pandas as pd
import json
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
import time
from utils.config_from_dict import getConfig
from graph_dataset import graph_data


def getDate():
    ret = time.strftime('%m-%d-%H-%M', time.localtime(time.time()))
    return ret


class dataLoader_provider:
    def __init__(self, model_name, dataset_path, split_path, batch_size, shuffle=False):
        self.model_name = model_name
        self.batch_size = batch_size
        self.split_path = split_path
        self.dataset_path = dataset_path
        self.shuffle = shuffle
        if split_path == None:
            self.split_path = dataset_path.replace("csv", "json")

        self.splits = self.get_splits_indices()

    def get_splits_indices(self):
        with open(self.split_path, "r") as fp:
            splits = json.load(fp)
        return splits

    def get_loader_one_fold(self, split_idx: int, label_index=0):
        indices = self.splits[split_idx]
        testset_indices = indices["test"]
        trainset_indices = indices["train"]

        # df = pd.read_csv(self.dataset_path)

        if self.model_name in ["DGCNN", "GIN", "ECC", "GraphSAGE", "DiffPool"]:  # 图模型
            train_dataset, test_dataset = graph_data.load_data_from_pt(
                self.dataset_path, trainset_indices, testset_indices, label_index)
            train_loader, test_loader = graph_data.construct_dataloader(
                train_dataset, test_dataset, self.batch_size, self.shuffle)

        # train_loader = DataLoader(
        #     df[trainSet_indices], self.batch_size, shuffle=True)
        # test_loader = DataLoader(
        #     df[testSet_indices], self.batch_size, shuffle=True)
        return train_loader, test_loader


def get_targets_len(dataset_path):
    df = pd.read_csv(dataset_path)
    targets_len = len(df.columns.tolist())-1
    return targets_len


def kfold_train_test(model_name, dataset_path, split_path=None,  k_fold=5,  args=None):
    model_class = getConfig("models", model_name)
    if model_name in ["DGCNN", "GIN", "ECC", "GraphSAGE", "DiffPool"]:
        # 实例化model
        # 这里args还要再fix一下,在模型内部需要访问args.dataset.max_num_nodes
        model = model_class(args["dim_features"], args["dim_target"], args)

    optimizer = getConfig("optimizers", args["optimizer"])(model.parameters(),
                                                           lr=args['learning_rate'], weight_decay=args['l2'])
    loss_fn = getConfig("losses", args["loss"])()

    loader_provider = dataLoader_provider(
        model_name, dataset_path, split_path, args["batch_size"])
    net = netWrapper.NetWrapper(
        model, loss_fn, device=args["device"], classification=True, metric_type=args["metric_type"])

    # metrics = []
    # losses = []
    # for i in range(k_fold):
    #     train_loader, test_loader = loader_provider.get_loader_one_fold(i)
    #     metric, loss = net.train_test_one_fold(train_loader, test_loader, max_epochs=100, optimizer=optimizer, scheduler=None, clipping=None,
    #                                            early_stopping=None, log_every=10)
    #     metrics.append(metric)
    #     # losses.append(loss)
    # metric_mean = np.array(metrics).mean()
    # metric_std = np.array(metrics).std()

    # multi_label
    targets_len = 1
    if args["multi_label"]:  # 可以去掉,此时不需要multi_label这个参数
        targets_len = get_targets_len(dataset_path)

    metric_mean = 0
    metric_std = 0
    multi_label_metrics = []
    for label_index in range(targets_len):
        metrics = []
        for i in range(k_fold):
            train_loader,  test_loader = loader_provider.get_loader_one_fold(
                i, label_index=label_index)
            metric, loss = net.train_test_one_fold(
                train_loader, test_loader)
            metrics.append(metric)
        multi_label_metrics.append(np.array(metrics).mean())
        metric_mean = multi_label_metrics[0]
        metric_std = np.array(metrics).std()

    if len(multi_label_metrics) != 1:  # 多标签
        metric_mean = np.array(multi_label_metrics).mean()
        metric_std = np.array(multi_label_metrics).std()

    if args["result_folder"] == None:
        args["result_folder"] = f"RESULT_{model_name}"
        os.mkdir(args["result_folder"])

    with open(os.path.join(args["result_folder"], "final_metrics")) as f:
        f.write("\n".join(metrics))
        f.writelines("")
        f.write(f"[metric_mean] {metric_mean} [metric_std] {metric_std}")


def main():
    cli_args = get_basic_parser()
    json_args = json.load(open(cli_args.model_config, "r"))
    args = cli_args.__dict__
    args.update(json_args)
    print(args)
    kfold_train_test(args["model_name"], args["dataset_path"],
                     args["split_path"], args["k_fold"], args)


if __name__ == "__main__":
    main()
