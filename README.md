
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
  - 数据集的访问方式
    - map-style datasets,
    - iterable-style datasets.
  - ``` python
    # 我这边GNN的dataset是map-style的,字典访问
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
