# 训练报告
**生成时间**: 2025年12月19日 23:01:48

---

## 1. 运行概览

- **总运行时间**: 0小时 5分钟 23秒
- **训练轮数**: 32 epochs
- **最终训练损失**: 0.052642
- **最终验证损失**: 0.060796
- **测试损失**: 0.119007

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
| 1 | 0.841871 | 0.922012 | 0.0010000000 | 598.10 |
| 2 | 0.599979 | 0.754877 | 0.0010000000 | 598.10 |
| 3 | 0.534699 | 0.887348 | 0.0010000000 | 598.10 |
| 4 | 0.467228 | 0.737190 | 0.0010000000 | 598.10 |
| 5 | 0.378004 | 0.570000 | 0.0009500000 | 598.10 |
| 6 | 0.318255 | 0.488392 | 0.0009500000 | 598.10 |
| 7 | 0.356450 | 0.428245 | 0.0009500000 | 598.10 |
| 8 | 0.259532 | 0.346141 | 0.0009500000 | 598.10 |
| 9 | 0.246227 | 0.293826 | 0.0009500000 | 598.10 |
| 10 | 0.203801 | 0.293197 | 0.0009025000 | 598.10 |
| 11 | 0.192517 | 0.185907 | 0.0009025000 | 598.10 |
| 12 | 0.109883 | 0.065359 | 0.0009025000 | 598.10 |
| 13 | 0.065544 | 0.268163 | 0.0009025000 | 598.10 |
| 14 | 0.072395 | 0.032484 | 0.0009025000 | 598.10 |
| 15 | 0.070046 | 0.033872 | 0.0008573750 | 598.10 |
| 16 | 0.052584 | 0.058134 | 0.0008573750 | 598.10 |
| 17 | 0.061589 | 0.030515 | 0.0008573750 | 598.10 |
| 18 | 0.089674 | 0.040428 | 0.0008573750 | 598.10 |
| 19 | 0.063151 | 0.058238 | 0.0008573750 | 598.10 |
| 20 | 0.061148 | 0.039207 | 0.0008145062 | 598.10 |
| 21 | 0.053991 | 0.028376 | 0.0008145062 | 598.10 |
| 22 | 0.057705 | 0.026222 | 0.0008145062 | 598.10 |
| 23 | 0.067030 | 0.041981 | 0.0008145062 | 598.10 |
| 24 | 0.065623 | 0.051275 | 0.0008145062 | 598.10 |
| 25 | 0.056553 | 0.048107 | 0.0007737809 | 598.10 |
| 26 | 0.070057 | 0.041638 | 0.0007737809 | 598.10 |
| 27 | 0.068970 | 0.042380 | 0.0007737809 | 598.10 |
| 28 | 0.065887 | 0.071107 | 0.0007737809 | 598.10 |
| 29 | 0.056622 | 0.031818 | 0.0007737809 | 598.10 |
| 30 | 0.083845 | 0.030013 | 0.0007350919 | 598.10 |
| 31 | 0.059961 | 0.033524 | 0.0007350919 | 598.10 |
| 32 | 0.052642 | 0.060796 | 0.0007350919 | 598.10 |

### 4.2 损失曲线

![损失曲线](images\loss_curves.png)

### 4.3 学习率变化

![学习率曲线](images\learning_rate.png)

### 4.4 GPU内存使用

![GPU内存使用](images\gpu_memory.png)

## 5. 验证过程

验证损失在训练过程中持续监控，用于早停机制和模型选择。

- **最佳验证损失**: 0.026222 (Epoch 22)
- **最终验证损失**: 0.060796

## 6. 测试过程

### 6.1 测试损失

- **测试MSE**: 0.119007

### 6.2 详细评估指标

#### 蒸汽系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.223441 | 4324.467452 |
| RMSE | 0.398709 | 5071.720526 |
| MAPE | 3.129670 | 0.892068 |
| WMAPE | 0.02502179 | 0.00879373 |

#### 空气系统

| 指标 | 流量G | 压力P |
|------|-------|-------|
| MAE | 0.145914 | 23507.686693 |
| RMSE | 0.279680 | 29267.435466 |
| MAPE | 1.325409 | 3.377865 |
| WMAPE | 0.01237749 | 0.03164339 |

## 7. 保存的模型权重

以下模型权重文件已保存：

- `reports\run_20251219_225623\models\model_nofno_notime_5.pth` (1.22 MB)
- `reports\run_20251219_225623\models\model_nofno_notime_10.pth` (1.22 MB)
- `reports\run_20251219_225623\models\model_nofno_notime_15.pth` (1.22 MB)
- `reports\run_20251219_225623\models\model_nofno_notime_20.pth` (1.22 MB)
- `reports\run_20251219_225623\models\model_nofno_notime_25.pth` (1.22 MB)
- `reports\run_20251219_225623\models\model_nofno_notime_30.pth` (1.22 MB)

## 8. 模型最终结果总结

### 8.1 训练效果

- 训练损失从 0.841871 降至 0.052642 (降低 93.75%)
- 验证损失从 0.922012 降至 0.060796 (降低 93.41%)

### 8.2 测试性能

模型在测试集上的表现：

- **蒸汽流量预测**: MAE=0.223441, RMSE=0.398709, MAPE=3.129670, WMAPE=0.02502179

- **蒸汽压力预测**: MAE=4324.467452, RMSE=5071.720526, MAPE=0.892068, WMAPE=0.00879373

- **空气流量预测**: MAE=0.145914, RMSE=0.279680, MAPE=1.325409, WMAPE=0.01237749

- **空气压力预测**: MAE=23507.686693, RMSE=29267.435466, MAPE=3.377865, WMAPE=0.03164339

## 9. 可视化图片

所有生成的可视化图片已保存在 `images/` 目录下。

![损失曲线](images\loss_curves.png)

![学习率曲线](images\learning_rate.png)

![GPU内存使用](images\gpu_memory.png)

