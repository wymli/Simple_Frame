
----

### 预计的命令行参数:

- config_path
- dataset_path
  - .csv
- split_path
- result_folder
- outer_kfold
- metric_type
  - auc
---

- config的一些超参:
  - max_epochs
  - device
  - loss_fn
  - optimizer
  - model_name
  - batch_size
  - ...
  <!-- -  [] early_stop
       - use_loss
       - use_metric -->
---
- 每个模型特异的:
  - config.json
  - data_process_func_smiles2inputVec
    - ```py
      df = pd.read_csv(dataset_path)
      dataset = process_data(df)
      dataloader = DataLoader(dataset[indices] , batch_size)
      ```
  - 数据集的访问
  - ``` python
    # 我这边GNN的dataset形式
    for data in dataloader:
      x, edge_index, batch = data.x, data.edge_index, data.batch
      label = data.y
    ```
  - 这边GNN的做法是直接将data传进model
  - ```py
    for data in dataloader:
      model.train()
      output = model(data)
      loss = loss_fn(output , data.y)
      ...
    ```

### 2020-04-26


### running pipeline:
- prepare data_split
```bash
python data_split.py --input_file = 'XXXX.csv' --random_split(store_true)/--scaffold_split(store_true) --k_fold = 5 --output_dir = './data_split/'
```
- run exp
```bash
python main.py --model_name = '' --task_type = classification/regression --multi_label(store_true) --dataset_path = '' --split_path = '' --k_fold = 5 --model_config
```
- prepare data
```py
python prepare_feats.py --dataset_path "" --dataset_type ""
```
- 目录结构(raw_dir,processed_dir为程序创建):
  - data
    - dataset_name
      - .csv
      - raw(folder)
        - .txt
      - processed(folder)
        - .pt
### Needed to change:
- [x] data_split
- [ ] dataset返回形式
  - [x] graph_data
- [ ] dataloader
  - [x] graph_data
- [x] multi_label
- [ ] visualization
- [ ] 载入config

#### add data_split.py
      scaffold, random_split

#### dataset: 统一返回形式：(features, labels)
```
      for data in dataloader:
        feats, label = data
        x, edge_index, batch = feats.x, feats.edge_index, feats.batch
```
#### dataloader: 
      main.py # line 34
      class dataLoader_provider
      每个模型构建 load_data_from_df & construct_loader(即get_loader_one_fold)函数
          load_data_from_df return feats, labels
          construct_loader return train_loader, test_loader
      ->
      load_data_from_pt
- 实现(graph_data):
```py
# dataset
for _, smiles, label in df.itertuples():
        data = smiles_to_graphData(smiles, label)
        dataList.append((data, label))
#dataloader
DataLoader(dataList[train_indices], batch_size, shuffle)
```
#### mulit-label:
- 应该是在main.py加个for循环
```py
df = pd.read_csv(csv_path)
target_name_list = df.columns.tolist()
target_name_list.remove('smiles')

multi_label_metrics = []
for target_name in target_name_list:
    metrics = []
    for i in range(k_fold):
      train_loader,  test_loader = loader_provider.get_loader_one_fold(i)
      metric, loss = net.train_test_one_fold
      metrics.append(metric)
    multi_label_metrics.append(np.array(metrics).mean())
metric_mean = np.array(multi_label_metrics).mean()
metric_std = np.array(multi_label_metrics).std()
```
- 实现如下:
```py
targets_len = 1
if args["multi_label"]: #可以去掉,此时不需要multi_label这个参数
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

if len(multi_label_metrics) != 1: #多标签
    metric_mean = np.array(multi_label_metrics).mean()
    metric_std = np.array(multi_label_metrics).std()
```
#### visualization
  - pass

