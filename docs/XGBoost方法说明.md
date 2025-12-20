# XGBoost方法训练和测试模块文档

## 模块概述

`xgboost_method_v2.py` 实现了基于XGBoost的能源系统预测模型，用于与深度学习模型进行性能对比。该模块将图结构数据转换为表格格式，使用XGBoost回归模型进行训练和预测。

### 核心特性

- **数据转换**：将图数据（形状为 `[num_samples, 2, n_his, n_vertex]`）转换为XGBoost可用的扁平特征向量
- **统一训练**：合并蒸汽和空气数据，使用统一的模型集合进行训练
- **多输出预测**：为每个输出维度训练独立的XGBoost模型
- **完整评估**：计算MAE、RMSE、MAPE、WMAPE等多种评估指标
- **数据反标准化**：自动处理标准化和反标准化，确保评估指标在原始尺度上

## 功能说明

### 1. 数据加载和预处理

模块从pickle文件加载预处理后的数据，包括：
- 训练集、验证集、测试集（蒸汽和空气数据）
- 标准化器（用于后续反标准化）

### 2. 特征工程

将图结构数据转换为表格特征：
- 输入：`[num_samples, 2, n_his, n_vertex]` 或 `[num_samples, 1, n_his, n_vertex]`
- 输出：`[num_samples, 2 * n_his * n_vertex]` 或 `[num_samples, n_his * n_vertex]`

### 3. 模型训练

- **统一训练策略**：将蒸汽和空气数据合并，通过padding统一特征维度，并添加类型标识特征
- **多输出模型**：为每个输出维度训练一个独立的XGBoost模型
- **固定轮数训练**：已移除早停机制，使用固定轮数（默认1500）进行训练

### 4. 模型评估

在验证集和测试集上计算以下指标：
- **MAE**（平均绝对误差）
- **RMSE**（均方根误差）
- **MAPE**（平均绝对百分比误差）
- **WMAPE**（加权平均绝对百分比误差）

分别计算蒸汽和空气的流量G和压力P的指标。

## API文档

### 主要函数

#### `load_data(data_path, Sscore_path, Escore_path)`

从pickle文件加载预处理后的数据。

**参数：**
- `data_path` (str): 数据pickle文件路径
- `Sscore_path` (str): 蒸汽数据标准化器pickle文件路径
- `Escore_path` (str): 电力数据标准化器pickle文件路径

**返回：**
- `S_score`: 蒸汽数据标准化器
- `E_score`: 电力数据标准化器
- 训练集、验证集、测试集数据（numpy格式）

**示例：**
```python
S_score, E_score, S_x_train, S_y_train, S_x_val, S_y_val, S_x_test, S_y_test, \
    E_x_train, E_y_train, E_x_val, E_y_val, E_x_test, E_y_test = load_data(
        "data/data.pickle",
        "data/S_score.pkl",
        "data/E_score.pkl"
    )
```

---

#### `prepare_features(x_data, n_his=12, n_vertex=31)`

将图数据转换为XGBoost可用的表格特征格式。

**参数：**
- `x_data` (np.ndarray): 输入数据，形状为 `[num_samples, 2, n_his, n_vertex]` 或 `[num_samples, 1, n_his, n_vertex]`
- `n_his` (int): 历史时间步数，默认为12
- `n_vertex` (int): 节点数量，默认为31

**返回：**
- `features` (np.ndarray): 扁平化的特征矩阵，形状为 `[num_samples, 2 * n_his * n_vertex]` 或 `[num_samples, n_his * n_vertex]`

**示例：**
```python
# 蒸汽数据特征准备
S_X_train = prepare_features(S_x_train, n_his=12, n_steam=24)
# 结果形状: [num_samples, 2 * 12 * 24] = [num_samples, 576]
```

---

#### `prepare_targets(y_data, indices=None)`

准备目标变量（标签）。

**参数：**
- `y_data` (np.ndarray): 标签数据，形状为 `[num_samples, 2, n_vertex]`
- `indices` (List[int], optional): 要评估的节点索引列表，如果为None则使用所有节点

**返回：**
- `targets` (np.ndarray): 目标变量，形状为 `[num_samples, 2 * len(indices)]`

**示例：**
```python
# 选择特定节点进行评估
steam_indices = [0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23]
S_y_train_selected = prepare_targets(S_y_train, steam_indices)
# 结果形状: [num_samples, 2 * 11] = [num_samples, 22]
```

---

#### `train_xgboost_model(X_train, y_train, X_val, y_val, n_outputs, params=None, verbose=True)`

训练XGBoost回归模型（多输出）。

为每个输出维度训练一个独立的XGBoost模型。

**参数：**
- `X_train` (np.ndarray): 训练集特征，形状为 `[num_train, num_features]`
- `y_train` (np.ndarray): 训练集标签，形状为 `[num_train, n_outputs]`
- `X_val` (np.ndarray): 验证集特征，形状为 `[num_val, num_features]`
- `y_val` (np.ndarray): 验证集标签，形状为 `[num_val, n_outputs]`
- `n_outputs` (int): 输出维度数量
- `params` (Dict, optional): XGBoost超参数字典，如果为None则使用默认参数
- `verbose` (bool): 是否打印训练信息，默认为True

**返回：**
- `models` (List[xgb.XGBRegressor]): XGBoost模型列表，每个模型对应一个输出维度

**默认参数：**
```python
{
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 1000,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}
```

**示例：**
```python
models = train_xgboost_model(
    X_train, y_train, X_val, y_val,
    n_outputs=y_train.shape[1],
    params=custom_params
)
```

---

#### `predict_with_models(models, X)`

使用多个XGBoost模型进行预测。

**参数：**
- `models` (List[xgb.XGBRegressor]): XGBoost模型列表
- `X` (np.ndarray): 输入特征，形状为 `[num_samples, num_features]`

**返回：**
- `predictions` (np.ndarray): 预测结果，形状为 `[num_samples, n_outputs]`

**示例：**
```python
y_pred = predict_with_models(models, X_test)
```

---

#### `evaluate_metrics(y_true, y_pred, scaler, indices, n_steam=24, is_steam=True)`

计算评估指标（MAE, RMSE, MAPE, WMAPE）。

**参数：**
- `y_true` (np.ndarray): 真实值，形状为 `[num_samples, 2 * len(indices)]`
- `y_pred` (np.ndarray): 预测值，形状为 `[num_samples, 2 * len(indices)]`
- `scaler` (List[StandardScaler]): 标准化器列表，用于反标准化
- `indices` (List[int]): 节点索引列表
- `n_steam` (int): 蒸汽节点数量，默认为24
- `is_steam` (bool): 是否为蒸汽数据，True表示蒸汽，False表示空气

**返回：**
- `MAE_G, RMSE_G, MAPE_G, WMAPE_G`: 流量G的指标
- `MAE_P, RMSE_P, MAPE_P, WMAPE_P`: 压力P的指标

**示例：**
```python
mae_G, rmse_G, mape_G, wmape_G, mae_P, rmse_P, mape_P, wmape_P = \
    evaluate_metrics(
        y_true, y_pred, S_score, steam_indices,
        n_steam=24, is_steam=True
    )
```

---

#### `merge_features_and_targets(S_X, S_y, E_X, E_y, n_his=12, n_steam=24, n_air=7)`

合并蒸汽和空气的特征和目标。

通过padding将特征维度统一，并添加类型标识特征。

**参数：**
- `S_X` (np.ndarray): 蒸汽特征，形状为 `[num_samples, 2 * n_his * n_steam]`
- `S_y` (np.ndarray): 蒸汽目标，形状为 `[num_samples, n_steam_outputs]`
- `E_X` (np.ndarray): 空气特征，形状为 `[num_samples, 2 * n_his * n_air]`
- `E_y` (np.ndarray): 空气目标，形状为 `[num_samples, n_air_outputs]`
- `n_his` (int): 历史时间步数
- `n_steam` (int): 蒸汽节点数量
- `n_air` (int): 空气节点数量

**返回：**
- `merged_X` (np.ndarray): 合并后的特征，形状为 `[num_total, max_features + 1]`
- `merged_y` (np.ndarray): 合并后的目标，形状为 `[num_total, n_steam_outputs + n_air_outputs]`
- `steam_output_dim` (int): 蒸汽输出维度
- `air_output_dim` (int): 空气输出维度

**说明：**
- 特征维度通过padding统一到最大维度
- 添加类型标识特征（0表示蒸汽，1表示空气）
- 目标维度扩展为 `[steam_outputs, air_outputs]` 的拼接

---

#### `split_predictions(predictions, steam_output_dim, air_output_dim, steam_indices, air_indices)`

从合并的预测结果中分离出蒸汽和空气的预测。

**参数：**
- `predictions` (np.ndarray): 合并的预测结果，形状为 `[num_samples, total_output_dim]`
- `steam_output_dim` (int): 蒸汽输出维度
- `air_output_dim` (int): 空气输出维度
- `steam_indices` (np.ndarray): 蒸汽样本的索引数组
- `air_indices` (np.ndarray): 空气样本的索引数组

**返回：**
- `S_predictions` (np.ndarray): 蒸汽预测结果，形状为 `[num_steam, steam_output_dim]`
- `E_predictions` (np.ndarray): 空气预测结果，形状为 `[num_air, air_output_dim]`

---

#### `train_and_evaluate(data_path, Sscore_path, Escore_path, n_his, n_steam, n_air, params)`

完整的训练和评估流程（合并蒸汽和空气训练）。

**参数：**
- `data_path` (str): 数据文件路径，默认为 `"data/data.pickle"`
- `Sscore_path` (str): 蒸汽标准化器路径，默认为 `"data/S_score.pkl"`
- `Escore_path` (str): 电力标准化器路径，默认为 `"data/E_score.pkl"`
- `n_his` (int): 历史时间步数，默认为12
- `n_steam` (int): 蒸汽节点数量，默认为24
- `n_air` (int): 空气节点数量，默认为7
- `params` (Dict, optional): XGBoost超参数，如果为None则使用默认参数（合并训练时自动增大规模）

**返回：**
- `dict`: 包含以下键的字典
  - `unified_models`: 训练好的模型列表
  - `validation_metrics`: 验证集指标字典
  - `test_metrics`: 测试集指标字典

**默认参数（合并训练）：**
```python
{
    "max_depth": 8,          # 增加深度（单独训练时为6）
    "n_estimators": 1500,    # 增加树的数量（单独训练时为1000）
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}
```

## 使用示例

### 基础使用

```bash
# 使用默认参数运行
python xgboost_method_v2.py
```

### 自定义参数

```bash
python xgboost_method_v2.py \
    --data_path data/data.pickle \
    --Sscore_path data/S_score.pkl \
    --Escore_path data/E_score.pkl \
    --n_his 12 \
    --n_steam 24 \
    --n_air 7 \
    --max_depth 8 \
    --learning_rate 0.1 \
    --n_estimators 1500
```

### 在Python代码中使用

```python
from xgboost_method_v2 import train_and_evaluate

# 设置参数
xgb_params = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "max_depth": 8,
    "learning_rate": 0.1,
    "n_estimators": 1500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# 训练和评估
results = train_and_evaluate(
    data_path="data/data.pickle",
    Sscore_path="data/S_score.pkl",
    Escore_path="data/E_score.pkl",
    n_his=12,
    n_steam=24,
    n_air=7,
    params=xgb_params
)

# 访问结果
print(f"测试集MSE: {results['test_metrics']['test_MSE']}")
print(f"蒸汽流量G MAE: {results['test_metrics']['steam_MAE_G']}")
print(f"空气压力P RMSE: {results['test_metrics']['air_RMSE_P']}")
```

## 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_path` | str | `data/data.pickle` | 数据文件路径 |
| `--Sscore_path` | str | `data/S_score.pkl` | 蒸汽标准化器路径 |
| `--Escore_path` | str | `data/E_score.pkl` | 电力标准化器路径 |
| `--n_his` | int | 12 | 历史时间步数 |
| `--n_steam` | int | 24 | 蒸汽节点数量 |
| `--n_air` | int | 7 | 空气节点数量 |
| `--max_depth` | int | 8 | 树的最大深度（合并训练建议使用8） |
| `--learning_rate` | float | 0.1 | 学习率 |
| `--n_estimators` | int | 1500 | 树的数量（合并训练建议使用1500） |

## 评估节点

模块默认评估以下节点（与深度学习模型保持一致）：

**蒸汽节点（11个）：**
- 索引：`[0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23]`

**空气节点（5个）：**
- 索引：`[0, 1, 4, 5, 6]`

## 输出说明

### 控制台输出

程序运行时会输出以下信息：

1. **数据加载信息**
   - 训练集、验证集、测试集大小

2. **特征准备信息**
   - 特征维度、输出维度

3. **训练进度**
   - 使用tqdm显示训练进度条

4. **验证集结果**
   - 蒸汽和空气的流量G和压力P的MAE、RMSE、MAPE、WMAPE

5. **测试集结果**
   - 蒸汽和空气的流量G和压力P的MAE、RMSE、MAPE、WMAPE
   - 测试集MSE损失
   - 总训练时间

### 输出示例

```
================================================================================
XGBoost能源系统预测模型
================================================================================

[1/5] 加载数据...
训练集大小: 1000
验证集大小: 200
测试集大小: 200

[2/5] 准备特征和目标变量...
蒸汽特征维度: 576
空气特征维度: 168
蒸汽输出维度: 22
空气输出维度: 10

[3/5] 合并蒸汽和空气数据...
合并后特征维度: 577
合并后输出维度: 32 (蒸汽: 22, 空气: 10)
合并后训练集大小: 2000

[4/5] 训练统一的XGBoost模型（合并蒸汽和空气）...
开始训练XGBoost模型，共 32 个输出维度...
训练模型: 100%|████████████| 32/32 [02:30<00:00,  4.69s/it]
统一模型训练完成，耗时: 150.23 秒

[5/5] 在验证集和测试集上评估模型...

================================================================================
验证集结果:
================================================================================
蒸汽 - 流量G: MAE=0.012345 | RMSE=0.023456 | MAPE=1.234567% | WMAPE=0.00123456
蒸汽 - 压力P: MAE=0.045678 | RMSE=0.067890 | MAPE=2.345678% | WMAPE=0.00234567
空气 - 流量G: MAE=0.034567 | RMSE=0.056789 | MAPE=1.567890% | WMAPE=0.00156789
空气 - 压力P: MAE=0.078901 | RMSE=0.090123 | MAPE=3.456789% | WMAPE=0.00345678

================================================================================
测试集结果:
================================================================================
蒸汽 - 流量G: MAE=0.012345 | RMSE=0.023456 | MAPE=1.234567% | WMAPE=0.00123456
蒸汽 - 压力P: MAE=0.045678 | RMSE=0.067890 | MAPE=2.345678% | WMAPE=0.00234567
空气 - 流量G: MAE=0.034567 | RMSE=0.056789 | MAPE=1.567890% | WMAPE=0.00156789
空气 - 压力P: MAE=0.078901 | RMSE=0.090123 | MAPE=3.456789% | WMAPE=0.00345678

测试集MSE损失: 0.001234
总训练时间: 150.23 秒
================================================================================
```

## 注意事项

### 1. 数据格式要求

- 数据文件必须是pickle格式，包含训练集、验证集、测试集
- 标准化器文件必须是pickle格式，包含StandardScaler对象
- 数据维度必须与参数设置一致（`n_his`, `n_steam`, `n_air`）

### 2. 内存使用

- 合并训练会增加内存使用，因为需要将特征padding到最大维度
- 如果内存不足，可以考虑：
  - 减小 `n_estimators`（树的数量）
  - 减小 `max_depth`（树的深度）
  - 使用更小的 `subsample` 和 `colsample_bytree`

### 3. 训练时间

- 合并训练时，模型规模会增大（`max_depth=8`, `n_estimators=1500`），训练时间会相应增加
- 训练时间与输出维度数量成正比（每个输出维度训练一个模型）
- 可以使用 `n_jobs=-1` 并行训练（默认已启用）

### 4. 模型保存

- 当前版本不自动保存模型，如需保存，可以手动保存返回的模型列表：
  ```python
  import pickle
  with open('xgboost_models.pkl', 'wb') as f:
      pickle.dump(results['unified_models'], f)
  ```

### 5. 与深度学习模型的对比

- XGBoost方法将图数据转换为表格格式，丢失了图结构信息
- 深度学习模型（如GNN）可以更好地利用图结构信息
- XGBoost方法训练速度快，但可能在某些任务上性能不如深度学习模型
- 建议同时运行两种方法进行对比

### 6. 早停机制

- 当前版本已移除早停机制，使用固定轮数训练
- 如需恢复早停机制，需要修改 `train_xgboost_model` 函数

## 依赖要求

```bash
# 必需依赖
uv pip install numpy xgboost scikit-learn tqdm

# 或使用pip
pip install numpy xgboost scikit-learn tqdm
```

## 常见问题

### Q1: 训练时间过长怎么办？

**A:** 可以尝试：
- 减小 `n_estimators`（如改为1000或800）
- 减小 `max_depth`（如改为6）
- 使用更小的数据集进行测试

### Q2: 内存不足错误

**A:** 可以尝试：
- 减小 `n_estimators`
- 减小 `max_depth`
- 减小 `subsample` 和 `colsample_bytree`（如改为0.6）

### Q3: 预测结果不理想

**A:** 可以尝试：
- 调整超参数（学习率、树的深度等）
- 增加 `n_estimators`
- 检查数据质量和特征工程

### Q4: 如何单独训练蒸汽或空气数据？

**A:** 当前版本设计为合并训练。如需单独训练，可以：
- 修改代码，只使用蒸汽或空气数据
- 使用 `xgboost_method_v1.py`（如果存在单独训练的版本）

## 版本历史

- **v2**: 合并蒸汽和空气数据统一训练，移除早停机制，增大模型规模
- **v1**: 初始版本（如果存在）

## 相关文档

- [快速开始指南](快速开始.md)
- [架构说明](架构说明.md)
- [报告生成功能说明](报告生成功能说明.md)

