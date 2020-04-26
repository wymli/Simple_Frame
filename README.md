
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
#### python data_split.py --input_file = 'XXXX.csv' --random_split(store_true)/--scaffold_split(store_true) --k_fold = 5 --output_dir = './data_split/'


#### python main.py --model_name = '' --task_type = classification/regression --multi_label(store_true) --dataset_path = '' --split_path = '' --k_fold = 5 --model_config

### Needed to change:
#### add data_split.py
      scaffold, random_split

#### dataset: 统一返回形式：(features, labels)
     for data in dataloader:
      x, edge_index, batch = data.x, data.edge_index, data.batch
      label = data.y
->
      for data in dataloader:
        feats, label = data
        x, edge_index, batch = feats.x, feats.edge_index, feats.batch

#### dataloader: 
      main.py # line 34
      class dataLoader_provider
      每个模型构建 load_data_from_df & construct_loader(get_loader_one_fold)函数
          load_data_from_df return feats, labels
          construct_loader return train_loader, test_loader

#### mulit-label:
      应该是在main.py加个for循环
        df = pd.read_csv(csv_path)
        target_name_list = df.columns.tolist()
        target_name_list.remove('smiles')

        multi_label_metrics = []
        for target_name in target_name_list:
           metrics = []
            for i in range(k_fold):
              train_loader, _, test_loader = loader_provider.get_loader_one_fold(i)
              metric, loss = net.train_test_one_fold
              metrics.append(metric)
            multi_label_metrics.append(np.array(metrics).mean())
        metric_mean = np.array(multi_label_metrics).mean()
        metric_std = np.array(multi_label_metrics).std()

#### visualization

