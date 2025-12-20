# XGBoost报告生成功能说明

## 功能概述

XGBoost报告生成功能会在每次运行结束后自动生成一份详细的markdown格式报告，包含训练过程的完整记录、模型信息、评估结果和可视化图表。该功能与深度学习模型的报告生成功能类似，但针对XGBoost的特点进行了优化。

## 报告内容

生成的报告包含以下内容：

### 1. 运行概览
- 总运行时间
- 模型数量（每个输出维度一个模型）
- 特征维度和输出维度
- 测试MSE损失

### 2. 数据信息
- 训练集、验证集、测试集大小
- 蒸汽和空气的训练样本数
- 蒸汽节点索引和空气节点索引

### 3. 模型结构
- 模型类型（XGBoost）
- 模型数量（每个输出维度一个独立的XGBoost回归模型）
- 多输出回归策略说明

### 4. 超参数配置
- XGBoost所有超参数（max_depth、learning_rate、n_estimators等）
- 完整参数列表（JSON格式，可展开查看）

### 5. 训练过程
- 训练和验证损失曲线（普通尺度和对数尺度）
- 每个输出维度的训练历史（通过eval_set记录）

### 6. 验证过程
- 验证集详细评估指标（MAE、RMSE、MAPE、WMAPE）
  - 蒸汽系统：流量G和压力P的指标
  - 空气系统：流量G和压力P的指标

### 7. 测试过程
- 测试MSE损失
- 详细评估指标（MAE、RMSE、MAPE、WMAPE）
  - 蒸汽系统：流量G和压力P的指标
  - 空气系统：流量G和压力P的指标

### 8. 保存的模型权重
- 所有保存的模型文件路径（每个输出维度一个模型文件）
- 每个模型文件的大小

### 9. 模型最终结果总结
- 测试性能总结（各指标的详细数值）

### 10. 可视化图片
- 训练损失曲线
- 特征重要性图（Top N特征）
- 验证集预测对比图（预测值vs真实值散点图）
- 验证集残差图
- 测试集预测对比图（预测值vs真实值散点图）
- 测试集残差图

## 报告存储位置

报告存储在 `reports/` 目录下，每次运行会创建一个带时间戳的文件夹：

```
reports/
└── xgboost_run_YYYYMMDD_HHMMSS/
    ├── report.md          # 主报告文件
    ├── images/            # 可视化图片目录
    │   ├── loss_curves.png
    │   ├── feature_importance.png
    │   ├── val_steam_predictions_vs_actual.png
    │   ├── val_air_predictions_vs_actual.png
    │   ├── val_steam_residuals.png
    │   ├── val_air_residuals.png
    │   ├── test_steam_predictions_vs_actual.png
    │   ├── test_air_predictions_vs_actual.png
    │   ├── test_steam_residuals.png
    │   └── test_air_residuals.png
    └── models/            # 模型权重目录
        ├── xgboost_model_0.pkl
        ├── xgboost_model_1.pkl
        └── ...
```

## 使用方法

报告生成功能已自动集成到 `xgboost_method_v2.py` 中，默认启用。每次运行训练脚本时，程序会自动：

1. 创建报告目录（带时间戳）
2. 记录模型信息和超参数
3. 记录训练过程中的验证损失（通过eval_set）
4. 保存所有模型权重
5. 生成可视化图表
6. 生成完整的markdown报告

### 启用/禁用报告生成

在调用 `train_and_evaluate` 函数时，可以通过 `enable_report` 参数控制：

```python
# 启用报告生成（默认）
results = train_and_evaluate(
    data_path="data/data.pickle",
    enable_report=True,  # 默认值为True
)

# 禁用报告生成
results = train_and_evaluate(
    data_path="data/data.pickle",
    enable_report=False,
)
```

### 命令行使用

运行 `xgboost_method_v2.py` 时，报告功能默认启用：

```bash
python xgboost_method_v2.py
```

## 可视化图表说明

### 1. 损失曲线（loss_curves.png）
- 显示训练和验证损失随boosting轮数的变化
- 包含普通尺度和对数尺度两个子图
- 损失值是对所有输出维度损失的平均值

### 2. 特征重要性图（feature_importance.png）
- 显示Top N（默认30）个最重要的特征
- 重要性值是对所有模型特征重要性的平均值
- 帮助理解哪些特征对预测最重要

### 3. 预测对比图（predictions_vs_actual.png）
- 散点图显示预测值vs真实值
- 包含R²系数
- 对角线表示完美预测
- 分别生成验证集和测试集的图表
- 分别生成蒸汽系统和空气系统的图表

### 4. 残差图（residuals.png）
- 残差vs预测值散点图：检查残差的分布是否均匀
- 残差分布直方图：检查残差是否符合正态分布
- 分别生成验证集和测试集的图表
- 分别生成蒸汽系统和空气系统的图表

## 技术细节

### 训练历史记录

XGBoost的训练历史通过 `eval_set` 参数记录。在训练每个输出维度的模型时，会同时记录训练集和验证集上的RMSE损失：

```python
model.fit(
    X_train,
    y_train[:, i],
    eval_set=[(X_train, y_train[:, i]), (X_val, y_val[:, i])],
    eval_metric="rmse",
    verbose=False,
)
```

训练完成后，从 `model.evals_result_` 中提取损失历史。

### 模型保存

每个输出维度训练一个独立的XGBoost模型，所有模型都保存在 `models/` 目录下，使用pickle格式：

```python
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
```

### 报告生成时机

报告在训练和评估完成后自动生成，包括：
1. 训练过程记录
2. 验证集评估
3. 测试集评估
4. 可视化图表生成
5. Markdown报告生成

## 注意事项

1. **训练时间**：报告生成会增加少量时间开销，主要用于可视化图表的生成
2. **存储空间**：每个模型文件通常为几MB到几十MB，取决于模型复杂度
3. **内存使用**：生成可视化图表时需要加载预测结果，对于大数据集可能占用较多内存
4. **中文支持**：报告和图表中的中文需要系统支持Microsoft YaHei字体

## 与深度学习报告的区别

1. **训练过程**：XGBoost没有epoch概念，而是通过boosting轮数记录训练过程
2. **模型结构**：XGBoost模型结构相对简单，主要是超参数配置
3. **模型数量**：XGBoost为每个输出维度训练一个独立模型，而深度学习通常使用一个多输出模型
4. **可视化**：增加了特征重要性图，这对XGBoost模型特别有意义

