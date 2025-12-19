# EnG2S - 能源系统时空图卷积网络

## 项目简介

本项目实现了一个基于时空图卷积网络（STGCN）的能源系统预测模型，用于预测工业能源系统中蒸汽和电力网络的流量（G）和压力（P）。模型结合了图神经网络和傅里叶神经算子（FNO），能够同时捕获时间和空间维度的依赖关系。

## 项目结构

```
EnG2S/
├── main.py                 # 主程序入口
├── model/                  # 模型定义
│   ├── my_layers.py       # 神经网络层定义
│   └── my_model.py        # STGCN模型定义
├── script/                # 工具脚本
│   ├── get_data.py        # 数据加载和处理
│   ├── utility.py         # 评估和工具函数
│   └── earlystopping.py   # 早停机制
├── data/                  # 数据目录
│   ├── data.pickle        # 预处理后的数据
│   ├── S_score.pkl        # 蒸汽数据标准化器
│   ├── E_score.pkl        # 电力数据标准化器
│   └── *.xlsx             # 原始Excel数据文件
└── docs/                  # 文档目录
```

## 主要特性

1. **时空图卷积网络（STGCN）**：同时捕获时间和空间依赖关系
2. **傅里叶神经算子（FNO）**：用于节点特征嵌入，捕获长期时间依赖
3. **物理约束**：可选的PDE损失，确保预测符合物理规律
4. **多类型节点支持**：同时处理蒸汽和电力两种不同类型的节点
5. **边特征嵌入**：将边的物理属性（长度、直径）编码为边权重

## 环境要求

- Python 3.7+
- PyTorch 1.8+
- PyTorch Geometric
- NumPy
- Pandas
- scikit-learn

## 安装依赖

```bash
# 使用uv pip安装（推荐）
uv pip install torch torch-geometric numpy pandas scikit-learn openpyxl tqdm

# 或使用pip
pip install torch torch-geometric numpy pandas scikit-learn openpyxl tqdm
```

## 使用方法

### 1. 数据准备

首次运行需要从Excel文件准备数据。取消注释 `main.py` 中的以下代码：

```python
GATG = 'data/steam_G.xlsx'  # 蒸汽流量数据
GATP = "data/steam_P.xlsx"  # 蒸汽压力数据
GATE1 = 'data/preair_G.xlsx'  # 电力流量数据
GATE2 = "data/preair_P.xlsx"  # 电力压力数据
data_preparate(args, GATG, GATP, GATE1, GATE2, device)
```

这将生成预处理后的数据文件：
- `data/data.pickle`
- `data/S_score.pkl`
- `data/E_score.pkl`

### 2. 训练模型

直接运行主程序：

```bash
python main.py
```

### 3. 配置参数

可以通过命令行参数调整模型配置，例如：

```bash
python main.py --epochs 50 --batch_size 64 --lr 0.0005 --patience 20
```

主要参数说明：
- `--epochs`: 训练轮数（默认：40）
- `--batch_size`: 批次大小（默认：128）
- `--lr`: 学习率（默认：0.001）
- `--n_his`: 历史时间步数（默认：12）
- `--n_pred`: 预测时间步数（默认：3）
- `--patience`: 早停耐心值（默认：30）
- `--need_grad`: 是否使用PDE损失（默认：False）

完整参数列表请参考 `main.py` 中的 `get_parameters()` 函数。

## 模型架构

### 输入
- **蒸汽节点特征**：流量G和压力P的历史数据
- **电力节点特征**：流量G和压力P的历史数据
- **图结构**：节点之间的连接关系（边索引）
- **边属性**：管道的物理参数（长度、直径）

### 处理流程
1. **节点嵌入**：使用FNO对每个节点的特征进行嵌入
2. **边嵌入**：将边的物理属性编码为边权重
3. **时空卷积**：通过两个STConv块捕获时空依赖
4. **输出预测**：生成未来时间步的流量和压力预测

### 输出
- **流量G**：每个节点在预测时间步的流量值
- **压力P**：每个节点在预测时间步的压力值

## 评估指标

模型在测试集上计算以下指标：
- **MAE**（平均绝对误差）
- **RMSE**（均方根误差）
- **WMAPE**（加权平均绝对百分比误差）

分别对蒸汽和电力的流量（G）和压力（P）进行计算。

## 文件说明

### main.py
主程序文件，包含：
- 参数解析和配置
- 数据加载
- 模型训练
- 模型验证和测试

### model/my_layers.py
定义了各种神经网络层：
- `NodeEmbedding`: 节点嵌入层（使用FNO）
- `FNO1d`: 一维傅里叶神经算子
- `TemporalConvLayer`: 时间卷积层
- `GraphConvLayer`: 图卷积层
- `STConvBlock`: 时空卷积块
- `OutputBlock`: 输出块

### model/my_model.py
定义了主模型 `STGCNGraphConv`，整合所有层构建完整的预测模型。

### script/get_data.py
数据加载和处理功能：
- `get_train_data()`: 从Excel加载原始数据
- `get_load_data()`: 加载图结构和物理参数
- `data_transform()`: 将时间序列转换为滑动窗口格式
- `data_split()`: 将数据转换为图数据格式

### script/utility.py
工具函数：
- `evaluate_model()`: 评估模型MSE损失
- `evaluate_metric()`: 计算详细评估指标
- `custom_collate()`: 自定义批处理函数
- `calculate_pde_loss()`: 计算PDE损失
- `caculate_grad()`: 计算梯度损失

### script/earlystopping.py
实现早停机制，防止模型过拟合。

## 注意事项

1. **数据格式**：Excel文件应包含时间序列数据，每列代表一个节点，每行代表一个时间步。
2. **图结构**：当前图结构硬编码在 `get_data.py` 中，需要根据实际系统调整。
3. **物理参数**：管道长度和直径等参数需要根据实际系统配置。
4. **GPU支持**：如果使用GPU，确保CUDA已正确安装。

## 常见问题

### Q: 如何修改图结构？
A: 在 `script/get_data.py` 的 `get_train_data()` 或 `get_load_data()` 函数中修改 `edge_index` 列表。

### Q: 如何调整节点数量？
A: 修改 `main.py` 中的参数：`--n_vertex`、`--n_steam`、`--n_air`。

### Q: 如何使用PDE损失？
A: 设置 `--need_grad True`，模型将在训练时计算PDE损失并加入总损失。

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，请提交Issue或Pull Request。

