# XGBoost训练报告
**生成时间**: 2025年12月20日 10:19:22

---

## 1. 运行概览

- **总运行时间**: 0小时 3分钟 8秒
- **模型数量**: 32 个（每个输出维度一个模型）
- **特征维度**: 577
- **输出维度**: 32 (蒸汽: 22, 空气: 10)
- **测试MSE**: 0.100652

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
- **max_depth**: 8
- **learning_rate**: 0.1
- **n_estimators**: 1500
- **subsample**: 0.8
- **colsample_bytree**: 0.8
- **min_child_weight**: 3
- **gamma**: 0.1
- **reg_alpha**: 0.1
- **reg_lambda**: 1.0
- **random_state**: 42
- **n_jobs**: -1

### 4.2 完整参数列表

<details>
<summary>点击展开完整参数列表</summary>

```json
{
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
| MAE | 0.099159 | 769.225755 |
| RMSE | 0.151622 | 1527.012067 |
| MAPE | 1.733140% | 0.161174% |
| WMAPE | 0.01067544 | 0.00158404 |

#### 空气系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.101189 | 19992.222609 |
| RMSE | 0.150889 | 24500.230170 |
| MAPE | 0.803750% | 2.399699% |
| WMAPE | 0.00854125 | 0.02429042 |

## 7. 测试过程

### 7.1 测试损失

- **测试MSE**: 0.100652

### 7.2 详细评估指标

#### 蒸汽系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.276173 | 1763.358063 |
| RMSE | 0.595685 | 3130.232917 |
| MAPE | 3.278682% | 0.364324% |
| WMAPE | 0.03092694 | 0.00358576 |

#### 空气系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.138788 | 27000.861269 |
| RMSE | 0.346704 | 29986.206464 |
| MAPE | 1.011741% | 3.602528% |
| WMAPE | 0.01177300 | 0.03634551 |

## 8. 保存的模型权重

以下模型权重文件已保存：

- `reports\xgboost_run_20251220_101613\models\xgboost_model_0.pkl` (1.06 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_1.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_2.pkl` (1.11 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_3.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_4.pkl` (1.12 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_5.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_6.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_7.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_8.pkl` (1.06 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_9.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_10.pkl` (1.06 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_11.pkl` (1.06 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_12.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_13.pkl` (1.08 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_14.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_15.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_16.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_17.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_18.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_19.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_20.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_21.pkl` (1.07 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_22.pkl` (1.04 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_23.pkl` (1.05 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_24.pkl` (1.05 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_25.pkl` (1.05 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_26.pkl` (1.04 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_27.pkl` (1.05 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_28.pkl` (1.05 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_29.pkl` (1.05 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_30.pkl` (1.05 MB)
- `reports\xgboost_run_20251220_101613\models\xgboost_model_31.pkl` (1.05 MB)

## 9. 模型最终结果总结

### 9.1 测试性能

模型在测试集上的表现：

- **蒸汽流量预测**: MAE=0.276173, RMSE=0.595685, MAPE=3.278682%, WMAPE=0.03092694

- **蒸汽压力预测**: MAE=1763.358063, RMSE=3130.232917, MAPE=0.364324%, WMAPE=0.00358576

- **空气流量预测**: MAE=0.138788, RMSE=0.346704, MAPE=1.011741%, WMAPE=0.01177300

- **空气压力预测**: MAE=27000.861269, RMSE=29986.206464, MAPE=3.602528%, WMAPE=0.03634551

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

