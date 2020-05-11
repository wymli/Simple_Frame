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
        train_loader = None
        test_loader = None
        # ! load_data_from_pt每一折都重新读取pt文件 
        if self.model_name in ["DGCNN", "GIN", "ECC", "GraphSAGE", "DiffPool"]:  # 图模型
            train_dataset, test_dataset = graph_data.load_data_from_pt_graph(
                self.dataset_path, trainset_indices, testset_indices, label_index)
            train_loader, test_loader = graph_data.construct_dataloader(
                train_dataset, test_dataset, self.batch_size, self.shuffle)

        return train_loader, test_loader


def get_targets_len(dataset_path):
    '''
    param: csv path
    '''
    df = pd.read_csv(dataset_path)
    targets_len = len(df.columns.tolist())-1
    return targets_len


def kfold_train_test(model_name, dataset_path, split_path=None,  k_fold=5,  args=None):
    model_class = getConfig("models", model_name)
    if model_name in ["DGCNN", "GIN", "ECC", "GraphSAGE", "DiffPool"]:
        # 实例化model
        # 这里args还要再fix一下,在模型内部需要访问args.dataset.max_num_nodes,直接硬编码算了
        if args["dataset_name"] == "bbbp":
            args["dim_features"] = 186
            args["dim_target"] = 2
            args["max_num_nodes"] = 132
        model = model_class(args["dim_features"], args["dim_target"], args)

    optimizer = getConfig("optimizers", args["optimizer"])(model.parameters(),
                                                           lr=args['learning_rate'], weight_decay=args['l2'])
    loss_fn = getConfig("losses", args["loss"])()

    loader_provider = dataLoader_provider(
        model_name, dataset_path, split_path, args["batch_size"])
    net = netWrapper.NetWrapper(
        model, loss_fn, device=args["device"], classification=True, metric_type=args["metric_type"])


    # multi_label = 1 default
    targets_len = args["multi_label"]  #

    metric_mean = 0
    metric_std = 0
    multi_label_metrics = []
    for label_index in range(targets_len):
        metrics = []
        for i in range(k_fold):
            print("-"*10)
            train_loader,  test_loader = loader_provider.get_loader_one_fold(
                i, label_index=label_index)
            metric, loss = net.train_test_one_fold(
                train_loader, test_loader, optimizer=optimizer)
            metrics.append(metric)
        multi_label_metrics.append(np.array(metrics).mean())
        metric_mean = multi_label_metrics[0]
        metric_std = np.array(metrics).std()

    if len(multi_label_metrics) != 1:  # 多标签
        metric_mean = np.array(multi_label_metrics).mean()
        metric_std = np.array(multi_label_metrics).std()

    if args["result_folder"] == None:
        args["result_folder"] = f"RESULT_{model_name}"
        if not os.path.exists(args["result_folder"]):
            os.mkdir(args["result_folder"])

    with open(os.path.join(args["result_folder"], "Final_Result_%.0f" % time.time()), "w") as f:
        f.write(str(args))
        f.write("\n")
        f.write("[Metric every fold]:\n")
        f.write("\n".join(map(str, metrics)))
        f.write("\n")
        f.write(f"[metrics_mean] {metric_mean} [metrics_std] {metric_std}")
    print("-"*10)
    print(f"[metrics_mean] {metric_mean} [metrics_std] {metric_std}")


def main():
    cli_args = get_basic_parser()
    json_args = json.load(open(cli_args.model_config, "r"))
    json_dict = {k: v[0] for k, v in json_args.items()}
    args = cli_args.__dict__
    args.update(json_dict)
    args["dataset_name"] = args["dataset_path"].split("/")[-1].split(".")[0]
    print(args)
    kfold_train_test(args["model_name"], args["dataset_path"],
                     args["split_path"], args["k_fold"], args)


if __name__ == "__main__":
    main()
