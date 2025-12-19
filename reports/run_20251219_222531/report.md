# 训练报告
**生成时间**: 2025年12月19日 22:39:24

---

## 1. 运行概览

- **总运行时间**: 0小时 13分钟 51秒
- **训练轮数**: 40 epochs
- **最终训练损失**: 0.073192
- **最终验证损失**: 0.026912
- **测试损失**: 0.114348

## 2. 模型结构

### 2.1 模型基本信息

- **模型名称**: STGCNGraphConv
- **总参数量**: 294,035
- **可训练参数量**: 294,035

### 2.2 模型结构配置

```python
blocks = [[12], [64, 64, 64], [64, 64, 64], [128, 128], [2]]
```

### 2.3 完整模型结构

```
STGCNGraphConv(
  (steam_embed): NodeEmbedding(
    (fno): FNO1d(
      (fc0): Linear(in_features=2, out_features=128, bias=True)
      (conv0): SpectralConv1d()
      (conv1): SpectralConv1d()
      (conv2): SpectralConv1d()
      (conv3): SpectralConv1d()
      (w0): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
      (w1): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
      (w2): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
      (w3): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
      (fc1): Linear(in_features=128, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=12, bias=True)
    )
  )
  (e_embed): NodeEmbedding(
    (fno): FNO1d(
      (fc0): Linear(in_features=2, out_features=128, bias=True)
      (conv0): SpectralConv1d()
      (conv1): SpectralConv1d()
      (conv2): SpectralConv1d()
      (conv3): SpectralConv1d()
      (w0): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
      (w1): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
      (w2): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
      (w3): Conv1d(12, 12, kernel_size=(1,), stride=(1,))
      (fc1): Linear(in_features=128, out_features=128, bias=True)
      (fc2): Linear(in_features=128, out_features=12, bias=True)
    )
  )
  (steam_edge_embed): edge_embed(
    (fc1): Linear(in_features=2, out_features=8, bias=False)
    (fc2): Linear(in_features=8, out_features=1, bias=False)
    (relu): ReLU()
    (dropout): Dropout(p=0.01, inplace=False)
  )
  (stblock1): STConvBlock(
    (tmp_conv1): TemporalConvLayer(
      (align): Align(
        (align_conv): Conv2d(12, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (align_t): Align(
        (align_conv): Conv2d(12, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (causal_conv): CausalConv2d(12, 128, kernel_size=(3, 1), stride=(1, 1))
      (relu): ReLU()
      (silu): SiLU()
    )
    (graph_conv): GraphConvLayer(
      (conv1): GCNConv(64, 64)
      (dropout1): Dropout(p=0.1, inplace=False)
      (conv3): GCNConv(64, 64)
      (tanh): Tanh()
    )
    (tmp_conv2): TemporalConvLayer(
      (align): Align(
        (align_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (align_t): Align(
        (align_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (causal_conv): CausalConv2d(64, 128, kernel_size=(3, 1), stride=(1, 1))
      (relu): ReLU()
      (silu): SiLU()
    )
    (tc2_ln): LayerNorm((31, 64), eps=1e-05, elementwise_affine=True)
    (tanh): Tanh()
    (dropout): Dropout(p=0.05, inplace=False)
  )
  (stblock2): STConvBlock(
    (tmp_conv1): TemporalConvLayer(
      (align): Align(
        (align_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (align_t): Align(
        (align_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (causal_conv): CausalConv2d(64, 128, kernel_size=(3, 1), stride=(1, 1))
      (relu): ReLU()
      (silu): SiLU()
    )
    (graph_conv): GraphConvLayer(
      (conv1): GCNConv(64, 64)
      (dropout1): Dropout(p=0.1, inplace=False)
      (conv3): GCNConv(64, 64)
      (tanh): Tanh()
    )
    (tmp_conv2): TemporalConvLayer(
      (align): Align(
        (align_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (align_t): Align(
        (align_conv): Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))
      )
      (causal_conv): CausalConv2d(64, 128, kernel_size=(3, 1), stride=(1, 1))
      (relu): ReLU()
      (silu): SiLU()
    )
    (tc2_ln): LayerNorm((31, 64), eps=1e-05, elementwise_affine=True)
    (tanh): Tanh()
    (dropout): Dropout(p=0.05, inplace=False)
  )
  (output): OutputBlock(
    (tmp_conv1): TemporalConvLayer(
      (align): Align(
        (align_conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
      )
      (align_t): Align(
        (align_conv): Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
      )
      (causal_conv): CausalConv2d(64, 256, kernel_size=(4, 1), stride=(1, 1))
      (relu): ReLU()
      (silu): SiLU()
    )
    (fc1): Linear(in_features=128, out_features=128, bias=True)
    (fc2): Linear(in_features=128, out_features=2, bias=True)
    (tc1_ln): LayerNorm((31, 128), eps=1e-05, elementwise_affine=True)
    (tc1_ln_t): LayerNorm((1, 128), eps=1e-05, elementwise_affine=True)
    (tanh): Tanh()
    (dropout): Dropout(p=0.05, inplace=False)
  )
)
```

## 3. 超参数配置

### 3.1 训练参数

- **epochs**: 40
- **batch_size**: 128
- **lr**: 0.001
- **opt**: adam
- **weight_decay_rate**: 0.1
- **step_size**: 5
- **gamma**: 0.95
- **patience**: 10
- **save_interval**: 5

### 3.2 模型结构参数

- **n_vertex**: 31
- **n_steam**: 24
- **n_air**: 7
- **n_his**: 12
- **n_pred**: 1
- **steam_dim**: 2
- **electricity_dim**: 2
- **embed_dim**: 12
- **mid_dim**: 64
- **Kt**: 3
- **stblock_num**: 2
- **act_func**: glu
- **droprate**: 0.05
- **weight_type**: 3

### 3.3 完整参数列表

<details>
<summary>点击展开完整参数列表</summary>

```json
{
  "enable_cuda": true,
  "seed": 42,
  "dataset": "IES",
  "train_ratio": 0.8,
  "valid_ratio": 0.1,
  "need_grad": false,
  "n_vertex": 31,
  "n_steam": 24,
  "n_air": 7,
  "n_mid_vertex": 15,
  "n_his": 12,
  "n_pred": 1,
  "steam_dim": 2,
  "electricity_dim": 2,
  "steam_edge_dim": 2,
  "ele_edge_dim": 2,
  "embed_dim": 12,
  "mid_dim": 64,
  "num_heads": 4,
  "time_dim": 12,
  "time_intvl": 10,
  "Kt": 3,
  "stblock_num": 2,
  "act_func": "glu",
  "enable_bias": true,
  "droprate": 0.05,
  "pregraph_conv_type": "preGraphLayer_nomid",
  "weight_type": 3,
  "lr": 0.001,
  "weight_decay_rate": 0.1,
  "batch_size": 128,
  "epochs": 40,
  "opt": "adam",
  "step_size": 5,
  "gamma": 0.95,
  "patience": 10,
  "save_interval": 5,
  "lam": 0.05,
  "Tb": 549.9658495560284,
  "Ta": 288.15,
  "R": 0.4615,
  "R_air": 0.287,
  "T0": 273.15,
  "exp_data": false
}
```

</details>

## 4. 训练过程

### 4.1 训练历史

| Epoch | 训练损失 | 验证损失 | 学习率 | GPU内存(MB) |
|-------|----------|----------|--------|-------------|
| 1 | 0.837227 | 0.884417 | 0.0010000000 | 598.33 |
| 2 | 0.618904 | 0.855417 | 0.0010000000 | 598.33 |
| 3 | 0.511812 | 0.750795 | 0.0010000000 | 598.33 |
| 4 | 0.412366 | 0.673453 | 0.0010000000 | 598.33 |
| 5 | 0.374802 | 0.565315 | 0.0009500000 | 598.33 |
| 6 | 0.349468 | 0.459675 | 0.0009500000 | 598.33 |
| 7 | 0.280622 | 0.380442 | 0.0009500000 | 598.33 |
| 8 | 0.271117 | 0.307768 | 0.0009500000 | 598.33 |
| 9 | 0.236495 | 0.264562 | 0.0009500000 | 598.33 |
| 10 | 0.203192 | 0.267579 | 0.0009025000 | 598.33 |
| 11 | 0.192508 | 0.175636 | 0.0009025000 | 598.33 |
| 12 | 0.109316 | 0.104751 | 0.0009025000 | 598.33 |
| 13 | 0.071352 | 0.045665 | 0.0009025000 | 598.33 |
| 14 | 0.071867 | 0.047069 | 0.0009025000 | 598.33 |
| 15 | 0.051089 | 0.041375 | 0.0008573750 | 598.33 |
| 16 | 0.063694 | 0.030567 | 0.0008573750 | 598.33 |
| 17 | 0.060551 | 0.047092 | 0.0008573750 | 598.33 |
| 18 | 0.063698 | 0.171986 | 0.0008573750 | 598.33 |
| 19 | 0.071477 | 0.036369 | 0.0008573750 | 598.33 |
| 20 | 0.075321 | 0.035834 | 0.0008145062 | 598.33 |
| 21 | 0.058991 | 0.204420 | 0.0008145062 | 598.33 |
| 22 | 0.065837 | 0.029278 | 0.0008145062 | 598.33 |
| 23 | 0.103400 | 0.048218 | 0.0008145062 | 598.33 |
| 24 | 0.055418 | 0.028202 | 0.0008145062 | 598.33 |
| 25 | 0.078590 | 0.047666 | 0.0007737809 | 598.33 |
| 26 | 0.054566 | 0.110223 | 0.0007737809 | 598.33 |
| 27 | 0.058906 | 0.032338 | 0.0007737809 | 598.33 |
| 28 | 0.066359 | 0.028297 | 0.0007737809 | 598.33 |
| 29 | 0.050333 | 0.091430 | 0.0007737809 | 598.33 |
| 30 | 0.071757 | 0.026965 | 0.0007350919 | 598.33 |
| 31 | 0.059156 | 0.033072 | 0.0007350919 | 598.33 |
| 32 | 0.074818 | 0.030686 | 0.0007350919 | 598.33 |
| 33 | 0.079944 | 0.052360 | 0.0007350919 | 598.33 |
| 34 | 0.055411 | 0.034248 | 0.0007350919 | 598.33 |
| 35 | 0.070844 | 0.041721 | 0.0006983373 | 598.33 |
| 36 | 0.052864 | 0.032218 | 0.0006983373 | 598.33 |
| 37 | 0.068627 | 0.030133 | 0.0006983373 | 598.33 |
| 38 | 0.069008 | 0.065139 | 0.0006983373 | 598.33 |
| 39 | 0.055105 | 0.028760 | 0.0006983373 | 598.33 |
| 40 | 0.073192 | 0.026912 | 0.0006634204 | 598.33 |

### 4.2 损失曲线

![损失曲线](images\loss_curves.png)

### 4.3 学习率变化

![学习率曲线](images\learning_rate.png)

### 4.4 GPU内存使用

![GPU内存使用](images\gpu_memory.png)

## 5. 验证过程

验证损失在训练过程中持续监控，用于早停机制和模型选择。

- **最佳验证损失**: 0.026912 (Epoch 40)
- **最终验证损失**: 0.026912

## 6. 测试过程

### 6.1 测试损失

- **测试MSE**: 0.114348

### 6.2 详细评估指标

#### 蒸汽系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.294116 | 2668.909417 |
| RMSE | 0.488594 | 3803.538562 |
| WMAPE | 0.03293621 | 0.00542718 |

#### 空气系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.107812 | 37473.909300 |
| RMSE | 0.268790 | 42442.581589 |
| WMAPE | 0.00914541 | 0.05044314 |

## 7. 保存的模型权重

以下模型权重文件已保存：

- `reports\run_20251219_222531\models\model_nofno_notime_5.pth` (1.22 MB)
- `reports\run_20251219_222531\models\model_nofno_notime_10.pth` (1.22 MB)
- `reports\run_20251219_222531\models\model_nofno_notime_15.pth` (1.22 MB)
- `reports\run_20251219_222531\models\model_nofno_notime_20.pth` (1.22 MB)
- `reports\run_20251219_222531\models\model_nofno_notime_25.pth` (1.22 MB)
- `reports\run_20251219_222531\models\model_nofno_notime_30.pth` (1.22 MB)
- `reports\run_20251219_222531\models\model_nofno_notime_35.pth` (1.22 MB)
- `reports\run_20251219_222531\models\model_nofno_notime_40.pth` (1.22 MB)

## 8. 模型最终结果总结

### 8.1 训练效果

- 训练损失从 0.837227 降至 0.073192 (降低 91.26%)
- 验证损失从 0.884417 降至 0.026912 (降低 96.96%)

### 8.2 测试性能

模型在测试集上的表现：

- **蒸汽流量预测**: MAE=0.294116, RMSE=0.488594, WMAPE=0.03293621

- **蒸汽压力预测**: MAE=2668.909417, RMSE=3803.538562, WMAPE=0.00542718

- **空气流量预测**: MAE=0.107812, RMSE=0.268790, WMAPE=0.00914541

- **空气压力预测**: MAE=37473.909300, RMSE=42442.581589, WMAPE=0.05044314

## 9. 可视化图片

所有生成的可视化图片已保存在 `images/` 目录下。

![损失曲线](images\loss_curves.png)

![学习率曲线](images\learning_rate.png)

![GPU内存使用](images\gpu_memory.png)

