"""
主程序文件 - 能源系统时空图卷积网络（STGCN）训练和测试

本文件实现了基于图卷积网络的能源系统（蒸汽和电力）预测模型。
主要功能包括：
1. 环境配置和随机种子设置
2. 参数解析和模型配置
3. 数据加载和预处理
4. 模型训练、验证和测试
"""

import logging
import argparse
import random
import os
import tqdm
import numpy as np
import math
import pandas as pd
from sklearn import preprocessing
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import my_model
from script import get_data, utility, earlystopping
import pickle
import torch.nn.functional as F
from tqdm import tqdm
# from itertools import islice
# from torch.utils.tensorboard import SummaryWriter

def set_env(seed):
    """
    设置运行环境，确保实验的可重复性
    
    参数:
        seed (int): 随机种子值，用于固定随机数生成器的状态
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def get_parameters():
    """
    解析命令行参数并配置模型参数
    
    返回:
        args: 解析后的参数对象
        device: 计算设备（CPU或CUDA）
        blocks: 模型各层的通道数配置列表
    """
    parser = argparse.ArgumentParser(description='STGCN - 时空图卷积网络用于能源系统预测')
    
    # 基础配置
    parser.add_argument('--enable_cuda', type=bool, default=True, help='是否启用CUDA，默认为True')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于稳定实验结果')
    parser.add_argument('--dataset', type=str, default='IES', help='数据集名称')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--valid_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--need_grad', type=bool, default=False, help='是否需要计算梯度（用于PDE损失）')
    
    # 图结构参数
    parser.add_argument('--n_vertex', type=int, default=31, help='图中节点总数')
    parser.add_argument('--n_steam', type=int, default=24, help='蒸汽节点数量')
    parser.add_argument('--n_air', type=int, default=7, help='空气节点数量')
    parser.add_argument('--n_mid_vertex', type=int, default=15, help='中间节点数量')
    
    # 时间序列参数
    parser.add_argument('--n_his', type=int, default=12, help='历史时间步数（输入序列长度）')
    parser.add_argument('--n_pred', type=int, default=3, help='预测时间步数，默认为3')
    
    # 特征维度参数
    parser.add_argument('--steam_dim', type=int, default=2, help='蒸汽节点特征维度（通常为流量G和压力P）')
    parser.add_argument('--electricity_dim', type=int, default=2, help='电力节点特征维度')
    parser.add_argument('--steam_edge_dim', type=int, default=2, help='蒸汽边特征维度')
    parser.add_argument('--ele_edge_dim', type=int, default=2, help='电力边特征维度')
    parser.add_argument('--embed_dim', type=int, default=12, help='嵌入层输出维度')
    parser.add_argument('--mid_dim', type=int, default=64, help='中间层维度')
    parser.add_argument('--num_heads', type=int, default=4, help='注意力头数')
    parser.add_argument('--time_dim', type=int, default=12, help='时间编码维度')
    parser.add_argument('--time_intvl', type=int, default=10, help='时间间隔（分钟）')
    
    # 模型结构参数
    parser.add_argument('--Kt', type=int, default=3, help='时间卷积核大小')
    parser.add_argument('--stblock_num', type=int, default=2, help='时空卷积块数量')
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'], 
                        help='激活函数类型：glu（门控线性单元）或gtu（门控tanh单元）')
    parser.add_argument('--enable_bias', type=bool, default=True, help='是否启用偏置项，默认为True')
    parser.add_argument('--droprate', type=float, default=0.05, help='Dropout比率')
    parser.add_argument('--pregraph_conv_type', type=str, default='preGraphLayer_nomid',
                        choices=['preGraphLayer', 'preGraphLayer_nomid'], help='预图卷积层类型')
    parser.add_argument('--weight_type', type=int, default=3, help='边权重类型：1-无权重，2-乘法，3-除法')
    
    # 训练参数
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    parser.add_argument('--weight_decay_rate', type=float, default=0.1, help='权重衰减率（L2正则化）')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--epochs', type=int, default=40, help='训练轮数，默认为40')
    parser.add_argument('--opt', type=str, default='adam', help='优化器类型：adam, rmsprop, adamw')
    parser.add_argument('--step_size', type=int, default=5, help='学习率调度器步长')
    parser.add_argument('--gamma', type=float, default=0.95, help='学习率衰减系数')
    parser.add_argument('--patience', type=int, default=30, help='早停机制的耐心值')
    parser.add_argument('--save_interval', type=int, default=5, help='模型保存间隔（epoch）')
    
    # 物理参数（用于PDE损失计算）
    parser.add_argument('--lam', type=float, default=0.05, help='摩擦系数')
    parser.add_argument('--Tb', type=float, default=549.9658495560284, help='基准温度（K）')
    parser.add_argument('--Ta', type=float, default=288.15, help='空气温度（K）')
    parser.add_argument('--R', type=float, default=0.4615, help='蒸汽气体常数（kJ/(kg·K)）')
    parser.add_argument('--R_air', type=float, default=0.287, help='空气气体常数（kJ/(kg·K)）')
    parser.add_argument('--T0', type=float, default=273.15, help='参考温度（K）')
    
    # 其他参数
    parser.add_argument('--exp_data', type=bool, default=False, help='是否导出实验数据')

    args = parser.parse_args()
    print('训练配置: {}'.format(args))

    # 设置随机种子以确保实验结果可重复
    set_env(args.seed)

    # 选择计算设备：NVIDIA GPU (CUDA) 或 CPU
    if args.enable_cuda and torch.cuda.is_available():
        # 设置可用的CUDA设备
        # 此选项对于多GPU环境至关重要
        # 'cuda' 等价于 'cuda:0'
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 计算输出块的时间维度
    # 每个时空卷积块会减少 (Kt-1)*2 个时间步
    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # 构建模型各层的通道数配置
    # blocks[i] 表示第i层的通道数列表
    blocks = []
    blocks.append([args.embed_dim])  # 嵌入层输出维度
    for l in range(args.stblock_num):
        # 每个时空卷积块包含3个卷积层
        blocks.append([args.mid_dim, args.mid_dim, args.mid_dim])
    if Ko == 0:
        # 如果时间维度被完全压缩，使用单层全连接
        blocks.append([128])
    elif Ko > 0:
        # 如果还有时间维度，使用两层全连接
        blocks.append([128, 128])
    blocks.append([2])  # 输出层：2个特征（流量G和压力P）

    return args, device, blocks

def get_iter(G_df, P_df, len_train, len_val):
    """
    将数据分割为训练集、验证集和测试集，并进行标准化处理
    
    参数:
        G_df: 流量（G）数据DataFrame
        P_df: 压力（P）数据DataFrame
        len_train: 训练集长度
        len_val: 验证集长度
    
    返回:
        x_train, y_train: 训练集输入和标签
        x_val, y_val: 验证集输入和标签
        x_test, y_test: 测试集输入和标签
        zscore: 标准化器列表，用于后续反标准化
    """
    # 分割数据为训练集、验证集和测试集
    train_G, val_G, test_G = get_data.load_data(G_df, len_train, len_val)
    train_P, val_P, test_P = get_data.load_data(P_df, len_train, len_val)
    
    # 创建标准化器（Z-score标准化）
    zscore_G = preprocessing.StandardScaler()
    zscore_P = preprocessing.StandardScaler()
    
    # 对训练集进行拟合和转换，对验证集和测试集只进行转换
    train_G = zscore_G.fit_transform(train_G)
    val_G = zscore_G.transform(val_G)
    test_G = zscore_G.transform(test_G)
    train_P = zscore_P.fit_transform(train_P)
    val_P = zscore_P.transform(val_P)
    test_P = zscore_P.transform(test_P)
    zscore = [zscore_G, zscore_P]

    # 将时间序列数据转换为滑动窗口格式
    x_train_G, y_train_G = get_data.data_transform(train_G, args.n_his, args.n_pred, device, args)
    x_val_G, y_val_G = get_data.data_transform(val_G, args.n_his, args.n_pred, device, args)
    x_test_G, y_test_G = get_data.data_transform(test_G, args.n_his, args.n_pred, device, args)
    x_train_P, y_train_P = get_data.data_transform(train_P, args.n_his, args.n_pred, device, args)
    x_val_P, y_val_P = get_data.data_transform(val_P, args.n_his, args.n_pred, device, args)
    x_test_P, y_test_P = get_data.data_transform(test_P, args.n_his, args.n_pred, device, args)
    
    # 将流量（G）和压力（P）特征在特征维度上拼接
    x_train = torch.cat((x_train_G, x_train_P), dim=1)
    y_train = torch.cat((y_train_G, y_train_P), dim=1)
    x_val = torch.cat((x_val_G, x_val_P), dim=1)
    y_val = torch.cat((y_val_G, y_val_P), dim=1)
    x_test = torch.cat((x_test_G, x_test_P), dim=1)
    y_test = torch.cat((y_test_G, y_test_P), dim=1)
    return x_train, y_train, x_val, y_val, x_test, y_test, zscore

def data_preparate(args, GATG1, GATP1, GATG2, GATP2, device):
    """
    从Excel文件加载原始数据，进行预处理并保存为pickle格式
    
    参数:
        args: 配置参数
        GATG1: 蒸汽流量数据Excel文件路径
        GATP1: 蒸汽压力数据Excel文件路径
        GATG2: 电力流量数据Excel文件路径
        GATP2: 电力压力数据Excel文件路径
        device: 计算设备
    
    说明:
        此函数用于首次数据准备，将Excel数据转换为模型可用的格式并保存
    """
    # 从Excel文件加载数据
    G_df, P_df, E1_df, E2_df, edge_index, S1, S2, E1, E2, t = get_data.get_train_data(
        GATG1, GATP1, GATG2, GATP2, device, args)

    # 计算数据集分割长度
    data_col = G_df.shape[0]
    val_and_test_rate = 0.1  # 验证集和测试集各占10%
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    # 处理蒸汽（Steam）数据
    S_x_train, S_y_train, S_x_val, S_y_val, S_x_test, S_y_test, S_score = get_iter(
        G_df, P_df, len_train, len_val)
    # 处理电力（Electricity）数据
    E_x_train, E_y_train, E_x_val, E_y_val, E_x_test, E_y_test, E_score = get_iter(
        E1_df, E2_df, len_train, len_val)

    # 转换为numpy数组以便保存
    S_x_train = S_x_train.cpu().numpy()
    S_y_train = S_y_train.cpu().numpy()
    E_x_train = E_x_train.cpu().numpy()
    E_y_train = E_y_train.cpu().numpy()
    S_x_val = S_x_val.cpu().numpy()
    S_y_val = S_y_val.cpu().numpy()
    E_x_val = E_x_val.cpu().numpy()
    E_y_val = E_y_val.cpu().numpy()
    S_x_test = S_x_test.cpu().numpy()
    S_y_test = S_y_test.cpu().numpy()
    E_x_test = E_x_test.cpu().numpy()
    E_y_test = E_y_test.cpu().numpy()

    # 组织数据字典
    data = {
        'S_x_train': S_x_train,
        'S_y_train': S_y_train,
        'S_x_val': S_x_val,
        'S_y_val': S_y_val,
        'S_x_test': S_x_test,
        'S_y_test': S_y_test,
        'E_x_train': E_x_train,
        'E_y_train': E_y_train,
        'E_x_val': E_x_val,
        'E_y_val': E_y_val,
        'E_x_test': E_x_test,
        'E_y_test': E_y_test
    }

    # 使用pickle将数据保存到文件
    with open('data1.pickle', 'wb') as f:
        pickle.dump(data, f)
    with open('S_score1.pkl', 'wb') as f:
        pickle.dump(S_score, f)
    with open('E_score1.pkl', 'wb') as f:
        pickle.dump(E_score, f)
def load_data(args, data_path, Sscore_path, Escore_path):
    """
    从pickle文件加载预处理后的数据，并创建数据加载器
    
    参数:
        args: 配置参数
        data_path: 数据pickle文件路径
        Sscore_path: 蒸汽数据标准化器pickle文件路径
        Escore_path: 电力数据标准化器pickle文件路径
    
    返回:
        S_score: 蒸汽数据标准化器
        E_score: 电力数据标准化器
        train_iter: 训练集数据加载器
        val_iter: 验证集数据加载器
        test_iter: 测试集数据加载器
    """
    # 加载图结构和物理参数
    edge_index, S1, S2, E1, E2, t = get_data.get_load_data(args)
    device = torch.device("cuda" if args.enable_cuda else "cpu")
    
    # 加载预处理后的数据
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # 将numpy数组转换为PyTorch张量并移动到指定设备
    S_x_train = torch.from_numpy(data['S_x_train']).to(device)
    S_y_train = torch.from_numpy(data['S_y_train']).to(device)
    S_x_val = torch.from_numpy(data['S_x_val']).to(device)
    S_y_val = torch.from_numpy(data['S_y_val']).to(device)
    S_x_test = torch.from_numpy(data['S_x_test']).to(device)
    S_y_test = torch.from_numpy(data['S_y_test']).to(device)
    E_x_train = torch.from_numpy(data['E_x_train']).to(device)
    E_y_train = torch.from_numpy(data['E_y_train']).to(device)
    E_x_val = torch.from_numpy(data['E_x_val']).to(device)
    E_y_val = torch.from_numpy(data['E_y_val']).to(device)
    E_x_test = torch.from_numpy(data['E_x_test']).to(device)
    E_y_test = torch.from_numpy(data['E_y_test']).to(device)
    
    # 加载标准化器（用于后续反标准化）
    with open(Sscore_path, 'rb') as f:
        S_score = pickle.load(f)
    with open(Escore_path, 'rb') as f:
        E_score = pickle.load(f)
    
    # 将数据转换为图数据格式
    data_train = get_data.data_split(S_x_train, S_y_train, E_x_train, E_y_train, 
                                     edge_index, S1, S2, E1, E2, t)
    data_val = get_data.data_split(S_x_val, S_y_val, E_x_val, E_y_val, 
                                   edge_index, S1, S2, E1, E2, t)
    data_test = get_data.data_split(S_x_test, S_y_test, E_x_test, E_y_test, 
                                    edge_index, S1, S2, E1, E2, t)
    
    # 创建数据加载器
    # 使用自定义的collate函数来处理图数据批次
    train_iter = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, 
                           collate_fn=utility.custom_collate)
    val_iter = DataLoader(data_val, batch_size=args.batch_size, shuffle=True, 
                         collate_fn=utility.custom_collate)
    test_iter = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, 
                          collate_fn=utility.custom_collate)

    return S_score, E_score, train_iter, val_iter, test_iter

def prepare_model(args, blocks):
    """
    初始化模型、损失函数、优化器和学习率调度器
    
    参数:
        args: 配置参数
        blocks: 模型各层的通道数配置列表
    
    返回:
        loss: 损失函数（MSE）
        es: 早停机制对象
        model: 初始化后的模型
        optimizer: 优化器
        scheduler: 学习率调度器
    """
    # 使用均方误差损失函数
    loss = nn.MSELoss()
    
    # 初始化早停机制（监控验证损失，模式为最小化）
    es = earlystopping.EarlyStopping(mode='min', min_delta=0.0, patience=args.patience)

    # 创建模型并移动到指定设备
    model = my_model.STGCNGraphConv(args, blocks).to(device)

    # 根据参数选择优化器
    if args.opt == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate)
    elif args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    elif args.opt == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_rate, amsgrad=False)
    else:
        raise NotImplementedError(f'错误: 优化器 {args.opt} 未实现。')

    # 创建学习率调度器（按步长衰减）
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    return loss, es, model, optimizer, scheduler


def train(S_score, E_score, loss, args, optimizer, scheduler, es, model, train_iter, val_iter):
    """
    训练模型
    
    参数:
        S_score: 蒸汽数据标准化器
        E_score: 电力数据标准化器
        loss: 损失函数
        args: 配置参数
        optimizer: 优化器
        scheduler: 学习率调度器
        es: 早停机制对象
        model: 模型
        train_iter: 训练集数据加载器
        val_iter: 验证集数据加载器
    """
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0  # l_sum: 一个epoch的总损失, n: 一个epoch的样本数
        model.train()  # 设置为训练模式
        
        for batch in tqdm(train_iter):
            # 从批次中提取数据
            t = batch.t  # 时间信息
            xs = batch.xs  # 蒸汽节点特征
            xe = batch.xe  # 电力节点特征
            edge_index = batch.edge_index  # 图的边索引
            S1 = batch.S1  # 蒸汽管道长度（直径）
            S2 = batch.S2  # 蒸汽管道直径
            E1 = batch.E1  # 电力管道长度（直径）
            E2 = batch.E2  # 电力管道直径
            ys = batch.ys  # 蒸汽标签
            ye = batch.ye  # 电力标签
            y = torch.cat((ys, ye), dim=-1)  # 拼接蒸汽和电力标签

            # 前向传播
            y_pred = model(xs, xe, t, edge_index, S1, S2, E1, E2, False)  # [batch_size, 2, 1, num_nodes]
            y_pred = y_pred.squeeze(2)  # 移除时间维度 [batch_size, 2, num_nodes]
            
            # 选择需要评估的节点索引
            indices = torch.tensor([0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23, 24, 25, 28, 29, 30], device=device)
            
            if args.need_grad:
                # 如果需要计算PDE损失（物理约束）
                # 计算蒸汽和空气的梯度损失
                l_grad_steam = utility.caculate_grad(args, y_pred, S1, t, S_score, True)
                l_grad_air = utility.caculate_grad(args, y_pred, E1, t, E_score, False)
                
                # 选择特定节点进行评估
                y = torch.index_select(y.reshape(-1, args.n_vertex), 1, indices).reshape(
                    y.shape[0], 2, len(indices))
                y_pred = torch.index_select(y_pred.reshape(-1, args.n_vertex), 1, indices).reshape(
                    y_pred.shape[0], 2, len(indices))
                
                # 计算预测损失
                l_p = loss(y_pred, y)
                
                # 总损失 = PDE损失（缩放后）+ 预测损失
                l = (l_grad_steam[0] / 1000000 + l_grad_steam[1] / 10000 + 
                     l_grad_air[0] / 1000000 + l_grad_air[1] / 10000 + l_p)
            else:
                # 仅使用预测损失
                y = torch.index_select(y.reshape(-1, args.n_vertex), 1, indices).reshape(
                    y.shape[0], 2, len(indices))
                y_pred = torch.index_select(y_pred.reshape(-1, args.n_vertex), 1, indices).reshape(
                    y_pred.shape[0], 2, len(indices))
                l = loss(y_pred, y)

            # 反向传播和优化
            optimizer.zero_grad()  # 清零梯度
            l.backward()  # 反向传播
            optimizer.step()  # 更新参数
            
            # 累计损失和样本数
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        
        # 更新学习率
        scheduler.step()
        
        # 在验证集上评估
        val_loss = val(model, val_iter, loss)

        # 打印GPU内存使用情况
        gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0

        # 打印训练信息
        print('Epoch: {:03d} | Lr: {:.20f} | Train loss: {:.6f} | Val loss: {:.6f} | GPU占用: {:.6f} MiB'. \
              format(epoch + 1, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

        # 检查早停条件
        if es.step(val_loss):
            print('早停触发。')
            break

        # 定期保存模型
        if (epoch + 1) % args.save_interval == 0:
            save_path = f"model_nofno_notime_{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"模型已保存至 {save_path}")


@torch.no_grad()
def val(model, val_iter, loss):
    """
    在验证集上评估模型
    
    参数:
        model: 模型
        val_iter: 验证集数据加载器
        loss: 损失函数
    
    返回:
        验证集平均损失
    """
    model.eval()  # 设置为评估模式
    l_sum, n = 0.0, 0
    
    for batch in val_iter:
        # 从批次中提取数据
        xs = batch.xs
        xe = batch.xe
        edge_index = batch.edge_index
        S1 = batch.S1  # 蒸汽管道长度（直径）
        S2 = batch.S2
        E1 = batch.E1
        E2 = batch.E2
        t = batch.t
        ys = batch.ys
        ye = batch.ye
        y = torch.cat((ys, ye), dim=-1)

        # 前向传播（不计算梯度）
        y_pred = model(xs, xe, t, edge_index, S1, S2, E1, E2, False)  # [batch_size, 2, 1, num_nodes]
        y_pred = y_pred.squeeze(2)  # [batch_size, 2, num_nodes]
        
        # 选择需要评估的节点索引
        indices = torch.tensor([0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23, 24, 25, 28, 29, 30], device=device)
        y = torch.index_select(y.reshape(-1, args.n_vertex), 1, indices).reshape(
            y.shape[0], 2, len(indices))
        y_pred = torch.index_select(y_pred.reshape(-1, args.n_vertex), 1, indices).reshape(
            y_pred.shape[0], 2, len(indices))
        
        # 计算损失
        l = loss(y_pred, y)
        
        # 累计损失和样本数
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    
    return torch.tensor(l_sum / n)

@torch.no_grad()
def test(S_score, E_score, loss, model, test_iter, args):
    """
    在测试集上评估模型并打印详细指标
    
    参数:
        S_score: 蒸汽数据标准化器
        E_score: 电力数据标准化器
        loss: 损失函数
        model: 模型
        test_iter: 测试集数据加载器
        args: 配置参数
    
    打印的指标说明:
        - MAE: 平均绝对误差 (Mean Absolute Error)
        - RMSE: 均方根误差 (Root Mean Square Error)
        - WMAPE: 加权平均绝对百分比误差 (Weighted Mean Absolute Percentage Error)
        - G: 流量 (Flow)
        - P: 压力 (Pressure)
    """
    model.eval()  # 设置为评估模式
    
    # 计算测试集MSE损失
    test_MSE = utility.evaluate_model(model, loss, test_iter, args)
    
    # 计算详细的评估指标（MAE, RMSE, WMAPE）
    steam_MAE_G, steam_RMSE_G, steam_WMAPE_G, steam_MAE_P, steam_RMSE_P, steam_WMAPE_P, \
    air_MAE_G, air_RMSE_G, air_WMAPE_G, air_MAE_P, air_RMSE_P, air_WMAPE_P = utility.evaluate_metric(
        model, test_iter, S_score, E_score, args)
    
    # 打印测试结果
    print(
        f'测试损失: {test_MSE:.6f} \n'
        f'蒸汽 - 流量G: MAE={steam_MAE_G:.6f} | RMSE={steam_RMSE_G:.6f} | WMAPE={steam_WMAPE_G:.8f}\n'
        f'蒸汽 - 压力P: MAE={steam_MAE_P:.6f} | RMSE={steam_RMSE_P:.6f} | WMAPE={steam_WMAPE_P:.8f}\n'
        f'空气 - 流量G: MAE={air_MAE_G:.6f} | RMSE={air_RMSE_G:.6f} | WMAPE={air_WMAPE_G:.8f}\n'
        f'空气 - 压力P: MAE={air_MAE_P:.6f} | RMSE={air_RMSE_P:.6f} | WMAPE={air_WMAPE_P:.8f}')



if __name__ == "__main__":
    """
    主程序入口
    
    执行流程:
    1. 配置日志
    2. 解析参数并设置环境
    3. 加载数据
    4. 准备模型
    5. 训练模型
    6. 测试模型
    """
    # 配置日志系统
    # writer = SummaryWriter('logs')  # TensorBoard日志（可选）
    # logger = logging.getLogger('stgcn')
    # logging.basicConfig(filename='stgcn.log', level=logging.INFO)  # 文件日志（可选）
    logging.basicConfig(level=logging.INFO)
    
    # 获取参数、设备和模型结构配置
    args, device, blocks = get_parameters()
    print(f'使用设备: {device}')

    # 首次运行需要从Excel文件准备数据（取消注释以下代码）
    '''
    GATG = 'data/steam_G.xlsx'  # 蒸汽流量数据
    GATP = "data/steam_P.xlsx"  # 蒸汽压力数据
    GATE1 = 'data/preair_G.xlsx'  # 电力流量数据
    GATE2 = "data/preair_P.xlsx"  # 电力压力数据
    data_preparate(args, GATG, GATP, GATE1, GATE2, device)
    '''

    # 从pickle文件加载预处理后的数据
    S_score, E_score, train_iter, val_iter, test_iter = load_data(
        args, 'data/data.pickle', 'data/S_score.pkl', 'data/E_score.pkl')

    # 准备模型、损失函数、优化器等
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks)

    # 训练模型
    train(S_score, E_score, loss, args, optimizer, scheduler, es, model, train_iter, val_iter)
    
    # 在测试集上评估模型
    test(S_score, E_score, loss, model, test_iter, args)


