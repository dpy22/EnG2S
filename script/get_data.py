"""
数据加载和处理模块

本模块提供了从Excel文件加载数据、数据预处理、数据转换和数据分割等功能。
"""

import pandas as pd
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np


def get_train_data(excel_path_G, excel_path_P, excel_path_e1, excel_path_e2, device, args):
    """
    从Excel文件加载训练数据
    
    参数:
        excel_path_G: 蒸汽流量数据Excel文件路径
        excel_path_P: 蒸汽压力数据Excel文件路径
        excel_path_e1: 压缩空气流量数据Excel文件路径
        excel_path_e2: 压缩空气压力数据Excel文件路径
        device: 计算设备（CPU或CUDA）
        args: 配置参数对象
    
    返回:
        G_df: 蒸汽流量数据DataFrame
        P_df: 蒸汽压力数据DataFrame
        e1_df: 压缩空气流量数据DataFrame
        e2_df: 压缩空气压力数据DataFrame
        edge_index: 图的边索引
        S1: 蒸汽管道长度列表
        S2: 蒸汽管道直径列表
        E1: 压缩空气管道长度列表
        E2: 压缩空气管道直径列表
        t: 时间信息张量
    """

    # 定义图的边连接关系（有向图）
    # edge_index[0] 是源节点，edge_index[1] 是目标节点
    edge_index = [[0, 1, 2, 2, 1, 5, 5, 7, 8, 9, 9, 11, 11, 13, 7, 15, 16, 15, 18, 18, 20, 16, 22, 24, 25, 26, 27, 27, 26, 0],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 24]]
    edge_index = torch.tensor(edge_index).to(device)
    
    # 蒸汽管道的物理参数
    steam1 = [59, 273, 118, 99, 120, 29, 110, 42, 282, 53, 23, 20, 28, 23, 151, 314, 114, 90, 19, 10, 44, 14, 102]  # 管道长度（米）
    steam2 = [0.6, 0.35, 0.25, 0.3, 0.6, 0.35, 0.6, 0.6, 0.6, 0.25, 0.45, 0.35, 0.45, 0.35, 0.6, 0.45, 0.4, 0.45, 0.35, 0.45, 0.3, 0.45, 0.35]  # 管道直径（米）
    
    # 电力管道的物理参数
    e1 = [1340, 2919, 2834, 2300, 1350, 1290]  # 管道长度（米）
    e2 = [0.8, 0.8, 0.6, 0.4, 0.4, 0.4]  # 管道直径（米）
    
    # 时间信息（时间步，单位：分钟）
    t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    # 转换为张量并移动到指定设备
    t = torch.tensor(t, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    t = torch.unsqueeze(t, dim=-1)  # [12, 1]
    t = t.expand(12, args.n_vertex)  # [12, n_vertex] 扩展到所有节点
    steam1 = torch.tensor(steam1, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    steam2 = torch.tensor(steam2, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    e1 = torch.tensor(e1, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    e2 = torch.tensor(e2, dtype=torch.float32, requires_grad=args.need_grad).to(device)

    # 从Excel文件读取数据
    G_df = pd.read_excel(excel_path_G)  # 蒸汽流量
    G_df = G_df.abs()  # 取绝对值（确保非负）
    P_df = pd.read_excel(excel_path_P)  # 蒸汽压力
    e1_df = pd.read_excel(excel_path_e1)  # 压缩空气流量
    e1_df = e1_df.abs()  # 取绝对值
    e2_df = pd.read_excel(excel_path_e2)  # 压缩空气压力

    return G_df, P_df, e1_df, e2_df, edge_index, steam1, steam2, e1, e2, t


def get_load_data(args):
    """
    加载图结构和物理参数（不加载Excel数据）
    
    用于从pickle文件加载预处理后的数据时，只需要图结构和物理参数。
    
    参数:
        args: 配置参数对象
    
    返回:
        edge_index: 图的边索引
        steam1: 蒸汽管道长度
        steam2: 蒸汽管道直径
        e1: 压缩空气管道长度
        e2: 压缩空气管道直径
        t: 时间信息张量
    """
    # 定义图的边连接关系
    edge_index = [[0, 1, 2, 2, 1, 5, 5, 7, 8, 9, 9, 11, 11, 13, 7, 15, 16, 15, 18, 18, 20, 16, 22, 24, 25, 26, 27, 27, 26, 0],
                  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 24]]
    device = torch.device("cuda" if args.enable_cuda else "cpu")
    edge_index = torch.tensor(edge_index).to(device)
    
    # 蒸汽管道物理参数
    steam1 = [59, 273, 118, 99, 120, 29, 110, 42, 282, 53, 23, 20, 28, 23, 151, 314, 114, 90, 19, 10, 44, 14, 102]
    steam2 = [0.6, 0.35, 0.25, 0.3, 0.6, 0.35, 0.6, 0.6, 0.6, 0.25, 0.45, 0.35, 0.45, 0.35, 0.6, 0.45, 0.4, 0.45, 0.35, 0.45, 0.3, 0.45, 0.35]
    
    # 压缩空气管道物理参数
    e1 = [1340, 2919, 2834, 2300, 1350, 1290]
    e2 = [0.8, 0.8, 0.6, 0.4, 0.4, 0.4]

    # 时间信息
    t = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    
    # 转换为张量
    steam1 = torch.tensor(steam1, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    steam2 = torch.tensor(steam2, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    e1 = torch.tensor(e1, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    e2 = torch.tensor(e2, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    t = torch.tensor(t, dtype=torch.float32, requires_grad=args.need_grad).to(device)
    t = torch.unsqueeze(t, dim=-1)
    t = t.expand(12, args.n_vertex)
    
    return edge_index, steam1, steam2, e1, e2, t

def load_data(dataset_name, len_train, len_val):
    """
    将数据集分割为训练集、验证集和测试集
    
    参数:
        dataset_name: 数据集（DataFrame或numpy数组）
        len_train: 训练集长度
        len_val: 验证集长度
    
    返回:
        train: 训练集
        val: 验证集
        test: 测试集（剩余部分）
    """
    train = dataset_name[: len_train]
    val = dataset_name[len_train: len_train + len_val]
    test = dataset_name[len_train + len_val:]
    return train, val, test

def data_transform(data, n_his, n_pred, device, args):
    """
    将时间序列数据转换为滑动窗口格式
    
    使用滑动窗口方法将时间序列转换为监督学习格式：
    - 输入：过去n_his个时间步的数据
    - 输出：未来第n_pred个时间步的数据
    
    参数:
        data: 时间序列数据，形状为 [time_steps, n_vertex]
        n_his: 历史时间步数（输入序列长度）
        n_pred: 预测时间步数（预测未来第几个时间步）
        device: 计算设备
        args: 配置参数对象
    
    返回:
        x: 输入数据，形状为 [num_samples, 1, n_his, n_vertex]
        y: 标签数据，形状为 [num_samples, 1, n_vertex]
    """
    n_vertex = data.shape[1]  # 节点数
    len_record = len(data)  # 总时间步数
    num = len_record - n_his - n_pred  # 可生成的样本数
    
    # 初始化输入和输出数组
    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, 1, n_vertex])

    # 生成滑动窗口样本
    for i in range(num):
        head = i  # 窗口起始位置
        tail = i + n_his  # 窗口结束位置
        # 输入：从head到tail的历史数据
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        # 输出：tail + n_pred - 1时刻的数据（预测目标）
        y[i] = data[tail + n_pred - 1]
    
    # 转换为PyTorch张量
    x = torch.tensor(x, dtype=torch.float32, device=device, requires_grad=args.need_grad)
    y = torch.tensor(y, dtype=torch.float32, device=device)

    return x, y

def data_split(S_data_x, S_data_y, E_data_x, E_data_y, edge_index, S1, S2, E1, E2, t):
    """
    将数据转换为PyTorch Geometric的Data对象列表
    
    为每个样本创建一个Data对象，包含节点特征、边信息、标签等。
    
    参数:
        S_data_x: 蒸汽输入数据 [num_samples, 1, n_his, n_steam]
        S_data_y: 蒸汽标签数据 [num_samples, 1, n_steam]
        E_data_x: 压缩空气输入数据 [num_samples, 1, n_his, n_air]
        E_data_y: 压缩空气标签数据 [num_samples, 1, n_air]
        edge_index: 图的边索引
        S1: 蒸汽管道长度
        S2: 蒸汽管道直径
        E1: 压缩空气管道长度
        E2: 压缩空气管道直径
        t: 时间信息
    
    返回:
        data_list: Data对象列表，每个元素代表一个样本
    """
    data_list = []
    for i in range(S_data_x.shape[0]):
        # 提取第i个样本的数据
        xs = S_data_x[i]  # 蒸汽输入
        ys = S_data_y[i]  # 蒸汽标签
        xe = E_data_x[i]  # 压缩空气输入
        ye = E_data_y[i]  # 压缩空气标签

        # 创建PyTorch Geometric的Data对象
        data = Data(x=xs, edge_index=edge_index, edge_attr=S1, y=ys)
        # 添加额外的属性
        data.xe = xe  # 压缩空气节点特征
        data.ye = ye  # 压缩空气标签
        data.S2 = S2  # 蒸汽管道直径
        data.E1 = E1  # 压缩空气管道长度
        data.E2 = E2  # 压缩空气管道直径
        data.t = t  # 时间信息
        
        data_list.append(data)
    return data_list


