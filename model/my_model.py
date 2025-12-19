"""
主模型定义文件

本文件定义了STGCN（Spatio-Temporal Graph Convolutional Network）模型，
用于能源系统（蒸汽和电力）的时空预测任务。
"""

import torch
import torch.nn as nn
from model import my_layers


class STGCNGraphConv(nn.Module):
    """
    时空图卷积网络（STGCN）模型
    
    该模型结合了：
    1. 节点嵌入层：使用FNO对蒸汽和电力节点特征进行嵌入
    2. 边嵌入层：将边的物理属性编码为边权重
    3. 时空卷积块：同时捕获时间和空间依赖关系
    4. 输出层：生成最终的预测结果（流量G和压力P）
    
    输入:
        - x_steam: 蒸汽节点特征 [batch_size, steam_dim, time_steps, n_steam]
        - x_e: 电力节点特征 [batch_size, electricity_dim, time_steps, n_air]
        - t: 时间信息
        - edge_index: 图的边索引
        - steam_weight, steam_d: 蒸汽边的物理属性
        - e_weight, e_d: 电力边的物理属性
    
    输出:
        - x: 预测结果 [batch_size, 2, 1, n_vertex] (2表示流量G和压力P)
    """

    def __init__(self, args, blocks):
        """
        初始化STGCN模型
        
        参数:
            args: 配置参数对象
            blocks: 模型各层的通道数配置列表
        """
        super(STGCNGraphConv, self).__init__()

        # 节点嵌入层：将原始特征（流量G和压力P）转换为嵌入表示
        self.steam_embed = my_layers.NodeEmbedding(args.steam_dim)  # 蒸汽节点嵌入
        self.e_embed = my_layers.NodeEmbedding(args.electricity_dim)  # 电力节点嵌入
        
        # 边嵌入层：将边的物理属性编码为边权重
        self.steam_edge_embed = my_layers.edge_embed(args.steam_edge_dim, 0.01)

        # 虚拟边权重（用于连接不同类型的节点）
        self.virtual_edge = nn.Parameter(torch.randn(1, 1))
        
        # 两个时空卷积块
        self.stblock1 = my_layers.STConvBlock(args.Kt, args.n_vertex, blocks[0][-1], blocks[0 + 1], args.act_func,
                                               args.droprate, args.weight_type)
        self.stblock2 = my_layers.STConvBlock(args.Kt, args.n_vertex, blocks[1][-1], blocks[1 + 1], args.act_func,
                                               args.droprate, args.weight_type)

        # 计算输出块的时间维度
        # 每个时空卷积块会减少 (Kt-1)*2 个时间步
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        
        if self.Ko > 1:
            # 如果还有时间维度，使用OutputBlock
            self.output = my_layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], args.n_vertex, args.act_func,
                                             args.enable_bias, args.droprate)
        elif self.Ko == 0:
            # 如果时间维度被完全压缩，使用全连接层
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.leaky_relu = nn.LeakyReLU()
            self.silu = nn.SiLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, x_steam, x_e, t, edge_index, steam_weight, steam_d, e_weight, e_d, unknown_flag):
        """
        前向传播
        
        参数:
            x_steam: 蒸汽节点特征 [batch_size, steam_dim, time_steps, n_steam]
            x_e: 电力节点特征 [batch_size, electricity_dim, time_steps, n_air]
            t: 时间信息（未使用，保留用于未来扩展）
            edge_index: 图的边索引 [2, num_edges]
            steam_weight: 蒸汽边权重（如管道长度）
            steam_d: 蒸汽边直径
            e_weight: 电力边权重（如管道长度）
            e_d: 电力边直径
        
        返回:
            x: 预测结果 [batch_size, 2, 1, n_vertex] (2表示流量G和压力P)
        """
        # 嵌入边特征
        # 蒸汽边：使用长度/直径比和直径的平方作为特征
        steam_feature = self.steam_edge_embed(steam_weight, steam_d, 1E2)
        # 电力边：使用相同的嵌入方式
        e_feature = self.steam_edge_embed(e_weight, e_d, 1E3)
        # 虚拟边（用于连接不同类型的节点）
        virtual_feature = self.virtual_edge
        # 拼接所有边权重
        edge_weight = torch.cat((steam_feature, e_feature, virtual_feature), dim=0)

        # 嵌入节点特征
        x_steam = self.steam_embed(x_steam)  # [batch_size, embed_dim, time_steps, n_steam]
        x_e = self.e_embed(x_e)  # [batch_size, embed_dim, time_steps, n_air]

        # 在节点维度上拼接蒸汽和电力节点
        x = torch.cat((x_steam, x_e), dim=-1)  # [batch_size, embed_dim, time_steps, n_vertex]

        # 通过两个时空卷积块
        x = self.stblock1(x, edge_index, edge_weight)
        x = self.stblock2(x, edge_index, edge_weight)

        # 输出层
        if self.Ko > 1:
            # 使用OutputBlock（包含时间卷积）
            x = self.output(x)
        elif self.Ko == 0:
            # 使用全连接层（时间维度已被完全压缩）
            x = self.fc1(x.permute(0, 2, 3, 1))  # [batch_size, time_steps, n_vertex, channels]
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)  # [batch_size, 2, time_steps, n_vertex]
        
        return x




