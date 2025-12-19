"""
模型层定义文件

本文件定义了STGCN模型所需的各种神经网络层，包括：
1. 节点嵌入层（使用FNO进行特征提取）
2. 傅里叶神经算子（FNO）
3. 时间编码层
4. 因果卷积层
5. 时间卷积层
6. 图卷积层
7. 边嵌入层
8. 时空卷积块
9. 输出块
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import GCNConv, GATConv
import torch.nn.init as init
import torch
from torch import nn
from torch.nn import MultiheadAttention
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class NodeEmbedding(nn.Module):
    """
    节点嵌入层

    使用一维傅里叶神经算子（FNO1d）对每个节点的特征进行嵌入。
    该层将原始节点特征（如流量G和压力P）转换为更高维的嵌入表示。

    输入形状: [batch_size, feature_dim, time_steps, num_nodes]
    输出形状: [batch_size, embed_dim, time_steps, num_nodes]
    """

    def __init__(self, feature_dim):
        """
        初始化节点嵌入层

        参数:
            feature_dim: 输入特征维度（如2，表示流量G和压力P）
        """
        super(NodeEmbedding, self).__init__()
        self.fno = FNO1d(feature_dim)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, feature_dim, time_steps, num_nodes]

        返回:
            output_data: 嵌入后的特征，形状为 [batch_size, embed_dim, time_steps, num_nodes]
        """
        # 更优雅写法：批处理所有节点，无需循环，利用reshape和permute一次传给FNO
        b, f, t, n = x.shape  # b:batch, f:feature_dim, t:time, n:num_nodes

        # [batch, feature_dim, time_steps, num_nodes] -> [batch*num_nodes, time_steps, feature_dim]
        x_reshaped = x.permute(0, 3, 2, 1).reshape(b * n, t, f)
        # 批量通过FNO进行嵌入，[batch*num_nodes, time_steps, embed_dim]
        output = self.fno(x_reshaped)
        # [batch*num_nodes, time_steps, embed_dim] -> [batch, num_nodes, embed_dim, time_steps]
        output = output.reshape(b, n, t, -1).permute(0, 3, 2, 1)
        # [batch, embed_dim, time_steps, num_nodes]
        output_data = output
        return output_data


class FNO1d(nn.Module):
    """
    一维傅里叶神经算子（Fourier Neural Operator, FNO）

    FNO是一种用于学习偏微分方程解的神经网络架构，通过在频域进行卷积操作
    来捕获时间序列中的长期依赖关系。本实现结合了频域卷积和空间域卷积。

    输入形状: [batch_size, time_steps, feature_dim]
    输出形状: [batch_size, time_steps, embed_dim]
    """

    def __init__(self, feature_dim, modes=16, width=12):
        """
        初始化FNO1d层

        参数:
            feature_dim: 输入特征维度（如2，表示流量G和压力P）
            modes: 傅里叶模式数（频域卷积保留的频率分量数）
            width: 中间层宽度
        """
        super(FNO1d, self).__init__()

        self.modes1 = modes
        self.width = width

        # 输入投影层：将特征维度扩展到128
        self.fc0 = nn.Linear(feature_dim, 128)
        init.xavier_normal_(self.fc0.weight)

        # 四个频域卷积层（SpectralConv）和对应的空间域卷积层
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        # 空间域1D卷积层（用于残差连接）
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        init.xavier_normal_(self.w0.weight)
        init.xavier_normal_(self.w1.weight)
        init.xavier_normal_(self.w2.weight)
        init.xavier_normal_(self.w3.weight)

        # 输出投影层
        self.fc1 = nn.Linear(128, 128)
        init.xavier_normal_(self.fc1.weight)
        self.fc2 = nn.Linear(128, 12)  # 输出维度为12（embed_dim）
        init.xavier_normal_(self.fc2.weight)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, time_steps, feature_dim]

        返回:
            x: 输出张量，形状为 [batch_size, time_steps, embed_dim]
        """
        # 输入投影：[batch_size, time_steps, feature_dim] -> [batch_size, time_steps, 128]
        x = self.fc0(x)

        # 第一层：频域卷积 + 空间域卷积（残差连接）
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.tanh(x)

        # 第二层
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.tanh(x)

        # 第三层
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.tanh(x)

        # 第四层（无激活函数）
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        # 输出投影
        x = self.fc1(x)
        x = F.tanh(x)
        x = self.fc2(x)
        return x


class SpectralConv1d(nn.Module):
    """
    一维谱卷积层（频域卷积）

    在频域（Fourier域）进行卷积操作，通过傅里叶变换将信号转换到频域，
    在频域进行乘法操作，然后逆变换回时域。这种方法可以高效地捕获长期依赖关系。
    """

    def __init__(self, in_channels, out_channels, modes1):
        """
        初始化谱卷积层

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            modes1: 保留的傅里叶模式数（最多为 floor(N/2) + 1，其中N是时间步数）
        """
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # 要相乘的傅里叶模式数，最多为 floor(N/2) + 1
        self.scale = 1 / (in_channels * out_channels)  # 权重初始化缩放因子
        # 初始化复数权重参数
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, in_channels, time_steps]

        返回:
            x: 输出张量，形状为 [batch_size, out_channels, time_steps]
        """
        batchsize = x.shape[0]
        # 计算傅里叶系数（实值FFT，只保留正频率部分）
        x_ft = torch.fft.rfft(x)

        # 在频域进行卷积（只对前modes1个频率分量进行乘法）
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )

        # 逆傅里叶变换回物理空间（时域）
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

    def compl_mul1d(self, input, weights):
        """
        复数乘法（在频域进行卷积）

        参数:
            input: 输入频域信号，形状为 [batch_size, in_channels, modes1]
            weights: 权重，形状为 [in_channels, out_channels, modes1]

        返回:
            输出频域信号，形状为 [batch_size, out_channels, modes1]
        """
        return torch.einsum("bix,iox->box", input, weights)


class TimeEncode(torch.nn.Module):
    """
    时间编码层

    将时间戳编码为固定维度的向量表示，使用多个不同频率的余弦函数来捕获
    时间的周期性特征。这种编码方式可以更好地表示时间信息。
    """

    def __init__(self, args, factor=5):
        """
        初始化时间编码层

        参数:
            args: 配置参数对象
            factor: 频率因子（未使用，保留用于未来扩展）
        """
        super(TimeEncode, self).__init__()
        time_dim = args.time_dim
        self.factor = factor

        # 初始化基础频率：从10^0到10^9的对数均匀分布
        self.basis_freq = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim))).float()
        )
        # 初始化相位参数
        self.phase = torch.nn.Parameter(torch.zeros(time_dim).float())

        # 可选的线性变换层（当前未使用）
        self.dense = torch.nn.Linear(time_dim, time_dim, bias=False)
        torch.nn.init.xavier_normal_(self.dense.weight)

    def forward(self, ts):
        """
        前向传播

        参数:
            ts: 时间戳张量，形状为 [batch_size, seq_len]

        返回:
            harmonic: 编码后的时间特征，形状为 [batch_size, seq_len, time_dim]
        """
        batch_size = ts.size(0)
        seq_len = ts.size(1)

        # 调整维度：[batch_size, seq_len] -> [batch_size, seq_len, 1]
        ts = ts.view(batch_size, seq_len, 1)
        # 计算时间与频率的乘积：[batch_size, seq_len, time_dim]
        map_ts = ts * self.basis_freq.view(1, 1, -1)
        # 加上相位偏移
        map_ts += self.phase.view(1, 1, -1)

        # 应用余弦函数并平方，然后归一化
        harmonic = torch.cos(map_ts)
        return harmonic**2 / torch.sqrt(torch.Tensor([12]).cuda())
        # 可选：通过线性层进一步变换
        # return self.dense(harmonic**2)


class Align(nn.Module):
    """
    对齐层

    用于调整特征图的通道数，使其与后续层的输入要求匹配。
    如果输入通道数大于输出通道数，使用1x1卷积进行降维；
    如果输入通道数小于输出通道数，使用零填充进行升维。
    """

    def __init__(self, c_in, c_out):
        """
        初始化对齐层

        参数:
            c_in: 输入通道数
            c_out: 输出通道数
        """
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        # 1x1卷积用于降维
        self.align_conv = nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, 1)
        )
        init.xavier_normal_(self.align_conv.weight)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, c_in, timestep, n_vertex]

        返回:
            x: 输出张量，形状为 [batch_size, c_out, timestep, n_vertex]
        """
        if self.c_in > self.c_out:
            # 降维：使用1x1卷积
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            # 升维：使用零填充
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat(
                [
                    x,
                    torch.zeros(
                        [batch_size, self.c_out - self.c_in, timestep, n_vertex]
                    ).to(x),
                ],
                dim=1,
            )
        else:
            # 通道数相同，直接返回
            x = x

        return x


class CausalConv1d(nn.Conv1d):
    """
    一维因果卷积层

    因果卷积确保输出只依赖于当前时刻及之前时刻的输入，不会使用未来信息。
    这对于时间序列预测任务非常重要，保证了模型在预测时不会"偷看"未来数据。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        enable_padding=False,
        dilation=1,
        groups=1,
        bias=True,
    ):
        """
        初始化因果卷积层

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            enable_padding: 是否启用填充（左填充，保证因果性）
            dilation: 膨胀率
            groups: 分组卷积数
            bias: 是否使用偏置
        """
        if enable_padding == True:
            # 计算左填充大小，确保输出长度不变且保持因果性
            self.__padding = (kernel_size - 1) * dilation
        else:
            self.__padding = 0
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=self.__padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        init.xavier_normal_(self.weight)

        if bias:
            # 初始化偏置为0
            init.constant_(self.bias, 0)

    def forward(self, input):
        """
        前向传播

        参数:
            input: 输入张量

        返回:
            result: 输出张量（如果使用了填充，会截断多余的输出）
        """
        result = super(CausalConv1d, self).forward(input)
        if self.__padding != 0:
            # 截断右端的填充部分，保持输出长度与输入相同
            return result[:, :, : -self.__padding]

        return result


class CausalConv2d(nn.Conv2d):
    """
    二维因果卷积层

    在时间维度上保持因果性，确保输出只依赖于当前时刻及之前时刻的输入。
    空间维度（节点维度）上可以正常卷积。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        enable_padding=False,
        dilation=1,
        groups=1,
        bias=True,
    ):
        """
        初始化二维因果卷积层

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小（可以是整数或元组）
            stride: 步长
            enable_padding: 是否启用填充（左填充，保证因果性）
            dilation: 膨胀率
            groups: 分组卷积数
            bias: 是否使用偏置
        """
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            # 计算左填充大小（在时间维度上）
            self.__padding = [
                int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))
            ]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        init.xavier_normal_(self.weight)
        if bias:
            # 初始化偏置为0
            init.constant_(self.bias, 0)

    def forward(self, input):
        """
        前向传播

        参数:
            input: 输入张量，形状为 [batch_size, channels, time_steps, n_vertex]

        返回:
            result: 输出张量
        """
        if self.__padding != 0:
            # 在时间维度左侧和空间维度左侧进行填充
            # F.pad的格式：(pad_left, pad_right, pad_top, pad_bottom)
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        result = super(CausalConv2d, self).forward(input)

        return result


class TemporalConvLayer(nn.Module):
    """
    时间卷积层

    在时间维度上进行因果卷积，支持多种激活函数（GLU、GTU、ReLU、SiLU）。
    GLU（Gated Linear Unit）和GTU（Gated Tanh Unit）是门控激活函数，
    可以更好地控制信息流。
    """

    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        """
        初始化时间卷积层

        参数:
            Kt: 时间卷积核大小
            c_in: 输入通道数
            c_out: 输出通道数
            n_vertex: 节点数
            act_func: 激活函数类型（'glu', 'gtu', 'relu', 'silu'）
        """
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)  # 用于对齐输入通道数
        self.align_t = Align(c_in, c_out)  # 备用对齐层

        # 根据激活函数类型决定输出通道数
        if act_func == "glu" or act_func == "gtu":
            # GLU/GTU需要两倍的输出通道（一半用于门控，一半用于特征）
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=2 * c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1,
            )
        else:
            self.causal_conv = CausalConv2d(
                in_channels=c_in,
                out_channels=c_out,
                kernel_size=(Kt, 1),
                enable_padding=False,
                dilation=1,
            )
        self.act_func = act_func
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, c_in, time_steps, n_vertex]

        返回:
            x: 输出张量，形状为 [batch_size, c_out, time_steps-Kt+1, n_vertex]
        """
        # 对齐输入通道数并截取用于残差连接的部分
        x_in = self.align(x)[:, :, self.Kt - 1 :, :]

        # 因果卷积
        x_causal_conv = self.causal_conv(x)

        # 根据激活函数类型进行处理
        if self.act_func == "glu" or self.act_func == "gtu":
            # 将输出分为两部分：特征部分和门控部分
            x_p = x_causal_conv[:, : self.c_out, :, :]  # 特征部分
            x_q = x_causal_conv[:, -self.c_out :, :, :]  # 门控部分

            if self.act_func == "glu":
                # GLU: (x_p + x_in) * sigmoid(x_q)
                x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
            else:
                # GTU: tanh(x_p + x_in) * sigmoid(x_q)
                x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))

        elif self.act_func == "relu":
            x = self.relu(x_causal_conv + x_in)

        elif self.act_func == "silu":
            x = self.silu(x_causal_conv + x_in)

        else:
            raise NotImplementedError(f"错误: 激活函数 {self.act_func} 未实现。")

        return x


class edge_embed(nn.Module):
    """
    边嵌入层

    将边的物理属性（如管道长度和直径）编码为边权重。
    通过神经网络学习边特征的表示，用于图卷积操作。
    """

    def __init__(self, steam_feature_num, droprate):
        """
        初始化边嵌入层

        参数:
            steam_feature_num: 边特征数量（未使用，保留用于兼容性）
            droprate: Dropout比率
        """
        super(edge_embed, self).__init__()
        self.fc1 = nn.Linear(2, 8, bias=False)  # 输入2个特征（长度/直径比和直径的平方）
        self.fc2 = nn.Linear(8, 1, bias=False)  # 输出1个边权重
        init.kaiming_uniform_(
            self.fc1.weight, nonlinearity="relu"
        )  # 使用kaiming初始化，适配ReLU激活函数
        init.kaiming_uniform_(self.fc2.weight, nonlinearity="relu")

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, weight1, weight2, ratio):
        """
        前向传播

        参数:
            weight1: 第一个边属性（如管道长度或直径），[23]
            weight2: 第二个边属性（如管道直径），[23]
            ratio: 缩放比例因子

        返回:
            x: 嵌入后的边权重，形状为 [num_edges]
        """
        # 构建边特征：[(weight2/weight1)*ratio, weight2^2]
        x = torch.stack((weight2 / weight1 * ratio, weight2**2), dim=0)  # [2, 23]
        x = x.permute(1, 0)  # [23, 2]

        # 通过全连接层进行嵌入
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        return x  # [23, 1]


class GraphConvLayer(torch.nn.Module):
    """
    图卷积层

    使用图卷积网络（GCN）在空间维度（节点维度）上进行信息传播。
    对每个时间步分别进行图卷积操作，捕获节点之间的空间依赖关系。
    """

    def __init__(self, in_channels, out_channels, weight_type):
        """
        初始化图卷积层

        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            weight_type: 边权重类型（未使用，保留用于兼容性）
        """
        super(GraphConvLayer, self).__init__()
        # 两层GCN：先扩展到64维，再压缩到输出维度
        self.conv1 = GCNConv(in_channels, 64)
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv3 = GCNConv(64, out_channels)
        self.tanh = nn.Tanh()

    def forward(self, x, edge_index, edge_weight):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, num_features, time_steps, vertex]
            edge_index: 边索引，形状为 [2, num_edges]
            edge_weight: 边权重，形状为 [num_edges]

        返回:
            outputs: 输出张量，形状为 [batch_size, out_channels, time_steps, vertex]
        """
        batch_size, num_features, time_steps, vertex = x.size()

        # 存储每个时间步的输出
        outputs = []

        # 对每个时间步分别进行图卷积
        for t in range(time_steps):
            # 提取第t个时间步的数据：[batch_size, num_features, vertex]
            x_t = x[:, :, t, :].transpose(1, 2)  # [batch_size, vertex, num_features]

            # 第一层GCN
            x_t = self.conv1(x_t, edge_index, edge_weight)
            x_t = self.dropout1(x_t)

            # 第二层GCN
            x_t = self.conv3(x_t, edge_index, edge_weight)
            x_t = self.tanh(x_t)

            # 恢复维度顺序
            x_t = x_t.transpose(1, 2).contiguous()  # [batch_size, out_channels, vertex]

            outputs.append(x_t)

        # 堆叠所有时间步的输出
        outputs = torch.stack(outputs, dim=2)

        return outputs


class STConvBlock(nn.Module):
    """
    时空卷积块（Spatio-Temporal Convolution Block）

    包含 'TGTND' 结构：
    - T: 门控时间卷积层（GLU或GTU）
    - G: 图卷积层
    - T: 门控时间卷积层（GLU或GTU）
    - N: 层归一化（Layer Normalization）
    - D: Dropout

    该块同时捕获时间维度和空间维度的依赖关系。
    """

    def __init__(
        self,
        Kt,
        n_vertex,
        last_block_channel,
        channels,
        act_func,
        droprate,
        weight_type,
    ):
        """
        初始化时空卷积块

        参数:
            Kt: 时间卷积核大小
            n_vertex: 节点数
            last_block_channel: 上一个块的输出通道数
            channels: 当前块的通道数列表 [ch1, ch2, ch3]
            act_func: 激活函数类型
            droprate: Dropout比率
            weight_type: 边权重类型
        """
        super(STConvBlock, self).__init__()
        # 第一个时间卷积层
        self.tmp_conv1 = TemporalConvLayer(
            Kt, last_block_channel, channels[0], n_vertex, act_func
        )
        # 图卷积层
        self.graph_conv = GraphConvLayer(channels[0], channels[1], weight_type)
        # 第二个时间卷积层
        self.tmp_conv2 = TemporalConvLayer(
            Kt, channels[1], channels[2], n_vertex, act_func
        )
        # 层归一化
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]])
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x, edge_index, edge_weight):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, channels, time_steps, n_vertex]
            edge_index: 边索引
            edge_weight: 边权重

        返回:
            x: 输出张量
        """
        # 第一个时间卷积
        x = self.tmp_conv1(x)

        # 图卷积（空间维度）
        x = self.graph_conv(x, edge_index, edge_weight)
        x = self.dropout(x)
        x = self.tanh(x)

        # 第二个时间卷积
        x = self.tmp_conv2(x)

        # 层归一化（需要调整维度顺序）
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)

        return x


class OutputBlock(nn.Module):
    """
    输出块

    包含 'TNFF' 结构：
    - T: 门控时间卷积层（GLU或GTU）
    - N: 层归一化（Layer Normalization）
    - F: 全连接层
    - F: 全连接层

    将时空特征转换为最终的预测输出（流量G和压力P）。
    """

    def __init__(
        self,
        Ko,
        last_block_channel,
        channels,
        end_channel,
        n_vertex,
        act_func,
        bias,
        droprate,
    ):
        """
        初始化输出块

        参数:
            Ko: 输出时间卷积核大小（通常等于剩余的时间步数）
            last_block_channel: 上一个块的输出通道数
            channels: 通道数列表 [ch1, ch2]
            end_channel: 最终输出通道数（通常为2，表示流量G和压力P）
            n_vertex: 节点数
            act_func: 激活函数类型
            bias: 是否使用偏置
            droprate: Dropout比率
        """
        super(OutputBlock, self).__init__()
        # 时间卷积层
        self.tmp_conv1 = TemporalConvLayer(
            Ko, last_block_channel, channels[0], n_vertex, act_func
        )
        # 两个全连接层
        self.fc1 = nn.Linear(
            in_features=channels[0], out_features=channels[1], bias=bias
        )
        self.fc2 = nn.Linear(
            in_features=channels[1], out_features=end_channel, bias=bias
        )
        # 层归一化
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]])
        self.tc1_ln_t = nn.LayerNorm([1, channels[0]])  # 备用（未使用）
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        """
        前向传播

        参数:
            x: 输入张量，形状为 [batch_size, channels, time_steps, n_vertex]

        返回:
            x: 输出张量，形状为 [batch_size, end_channel, 1, n_vertex]
        """
        # 时间卷积
        x = self.tmp_conv1(x)

        # 层归一化（调整维度顺序）
        x = self.tc1_ln(
            x.permute(0, 2, 3, 1)
        )  # [batch_size, time_steps, n_vertex, channels]

        # 第一个全连接层
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.dropout(x)

        # 第二个全连接层
        x = self.fc2(x).permute(
            0, 3, 1, 2
        )  # [batch_size, end_channel, time_steps, n_vertex]

        return x
