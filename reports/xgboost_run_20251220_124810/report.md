# XGBoost训练报告
**生成时间**: 2025年12月20日 12:51:25

---

## 1. 运行概览

- **总运行时间**: 0小时 3分钟 14秒
- **模型数量**: 32 个（每个输出维度一个模型）
- **特征维度**: 577
- **输出维度**: 32 (蒸汽: 22, 空气: 10)
- **测试MSE**: 0.100559

## 2. 数据信息

- **训练集大小**: 6902 (蒸汽: 3451, 空气: 3451)
- **验证集大小**: 836
- **测试集大小**: 836
- **蒸汽节点索引**: [0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23]
- **空气节点索引**: [0, 1, 4, 5, 6]

## 3. 模型结构

- **模型类型**: XGBoost
- **模型数量**: 32 个独立的XGBoost回归模型
- **每个模型对应一个输出维度**: 采用多输出回归策略

## 4. 超参数配置

### 4.1 XGBoost参数

- **objective**: reg:squarederror
- **tree_method**: hist
- **max_depth**: 9
- **learning_rate**: 0.044913467579125975
- **n_estimators**: 1500
- **subsample**: 0.9958479357109641
- **colsample_bytree**: 0.8100683062187806
- **min_child_weight**: 7
- **gamma**: 0.2735927466196289
- **reg_alpha**: 0.4157721299338367
- **reg_lambda**: 0.6943785653552943
- **random_state**: 42
- **n_jobs**: -1

### 4.2 完整参数列表

<details>
<summary>点击展开完整参数列表</summary>

```json
{
  "objective": "reg:squarederror",
  "tree_method": "hist",
  "max_depth": 9,
  "learning_rate": 0.044913467579125975,
  "n_estimators": 1500,
  "subsample": 0.9958479357109641,
  "colsample_bytree": 0.8100683062187806,
  "min_child_weight": 7,
  "gamma": 0.2735927466196289,
  "reg_alpha": 0.4157721299338367,
  "reg_lambda": 0.6943785653552943,
  "random_state": 42,
  "n_jobs": -1
}
```

</details>

## 5. 训练过程

### 5.1 损失曲线

![损失曲线](images\loss_curves.png)

## 6. 验证过程

### 6.1 验证集指标

#### 蒸汽系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.087876 | 834.398254 |
| RMSE | 0.149901 | 1714.667030 |
| MAPE | 1.495446% | 0.175081% |
| WMAPE | 0.00946075 | 0.00171825 |

#### 空气系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.107237 | 9462.747545 |
| RMSE | 0.144076 | 11645.288503 |
| MAPE | 0.989258% | 1.200880% |
| WMAPE | 0.00905177 | 0.01149718 |

## 7. 测试过程

### 7.1 测试损失

- **测试MSE**: 0.100559

### 7.2 详细评估指标

#### 蒸汽系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.263014 | 1784.347371 |
| RMSE | 0.605079 | 3176.206567 |
| MAPE | 2.875992% | 0.369203% |
| WMAPE | 0.02945331 | 0.00362844 |

#### 空气系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.146508 | 30363.083901 |
| RMSE | 0.336435 | 32576.567870 |
| MAPE | 1.216424% | 4.279089% |
| WMAPE | 0.01242785 | 0.04087135 |

## 8. 保存的模型权重

以下模型权重文件已保存：

- `reports\xgboost_run_20251220_124810\models\xgboost_model_0.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_1.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_2.pkl` (1.12 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_3.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_4.pkl` (1.13 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_5.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_6.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_7.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_8.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_9.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_10.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_11.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_12.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_13.pkl` (1.10 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_14.pkl` (1.10 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_15.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_16.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_17.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_18.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_19.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_20.pkl` (1.09 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_21.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_22.pkl` (1.06 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_23.pkl` (1.06 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_24.pkl` (1.06 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_25.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_26.pkl` (1.06 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_27.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_28.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_29.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_30.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_124810\models\xgboost_model_31.pkl` (1.07 MB)

## 9. 模型最终结果总结

### 9.1 测试性能

模型在测试集上的表现：

- **蒸汽流量预测**: MAE=0.263014, RMSE=0.605079, MAPE=2.875992%, WMAPE=0.02945331

- **蒸汽压力预测**: MAE=1784.347371, RMSE=3176.206567, MAPE=0.369203%, WMAPE=0.00362844

- **空气流量预测**: MAE=0.146508, RMSE=0.336435, MAPE=1.216424%, WMAPE=0.01242785

- **空气压力预测**: MAE=30363.083901, RMSE=32576.567870, MAPE=4.279089%, WMAPE=0.04087135

## 10. 可视化图片

所有生成的可视化图片已保存在 `images/` 目录下。

### 10.1 训练损失曲线

![训练损失曲线](images\loss_curves.png)

### 10.2 特征重要性图

![特征重要性图](images\feature_importance.png)

### 10.3 验证集 - 蒸汽系统预测对比

![验证集 - 蒸汽系统预测对比](images\val_steam_predictions_vs_actual.png)

### 10.4 验证集 - 空气系统预测对比

![验证集 - 空气系统预测对比](images\val_air_predictions_vs_actual.png)

### 10.5 验证集 - 蒸汽系统残差图

![验证集 - 蒸汽系统残差图](images\val_steam_residuals.png)

### 10.6 验证集 - 空气系统残差图

![验证集 - 空气系统残差图](images\val_air_residuals.png)

### 10.7 测试集 - 蒸汽系统预测对比

![测试集 - 蒸汽系统预测对比](images\test_steam_predictions_vs_actual.png)

### 10.8 测试集 - 空气系统预测对比

![测试集 - 空气系统预测对比](images\test_air_predictions_vs_actual.png)

### 10.9 测试集 - 蒸汽系统残差图

![测试集 - 蒸汽系统残差图](images\test_steam_residuals.png)

### 10.10 测试集 - 空气系统残差图

![测试集 - 空气系统残差图](images\test_air_residuals.png)

