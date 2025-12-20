"""
XGBoost方法训练和测试模块

本文件实现了基于XGBoost的能源系统预测模型，用于与深度学习模型进行性能对比。
主要功能包括：
1. 数据加载和特征工程（将图数据转换为表格格式）
2. XGBoost模型训练
3. 模型验证和测试
4. 评估指标计算（MAE, RMSE, MAPE, WMAPE）
"""

import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
from typing import Tuple, List, Dict
import os
import time
from tqdm import tqdm

# 尝试导入早停回调，如果失败则使用旧版本API
try:
    from xgboost.callback import EarlyStopping
    USE_NEW_CALLBACK = True
except ImportError:
    USE_NEW_CALLBACK = False


def load_data(data_path: str, Sscore_path: str, Escore_path: str) -> Tuple:
    """
    从pickle文件加载预处理后的数据
    
    参数:
        data_path: 数据pickle文件路径
        Sscore_path: 蒸汽数据标准化器pickle文件路径
        Escore_path: 电力数据标准化器pickle文件路径
    
    返回:
        S_score: 蒸汽数据标准化器
        E_score: 电力数据标准化器
        训练集、验证集、测试集数据（numpy格式）
    """
    # 加载预处理后的数据
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    
    # 加载标准化器（用于后续反标准化）
    with open(Sscore_path, "rb") as f:
        S_score = pickle.load(f)
    with open(Escore_path, "rb") as f:
        E_score = pickle.load(f)
    
    return (
        S_score,
        E_score,
        data["S_x_train"],
        data["S_y_train"],
        data["S_x_val"],
        data["S_y_val"],
        data["S_x_test"],
        data["S_y_test"],
        data["E_x_train"],
        data["E_y_train"],
        data["E_x_val"],
        data["E_y_val"],
        data["E_x_test"],
        data["E_y_test"],
    )


def prepare_features(x_data: np.ndarray, n_his: int = 12, n_vertex: int = 31) -> np.ndarray:
    """
    将图数据转换为XGBoost可用的表格特征格式
    
    将形状为 [num_samples, 2, n_his, n_vertex] 的数据转换为
    形状为 [num_samples, 2 * n_his * n_vertex] 的扁平特征向量
    
    参数:
        x_data: 输入数据，形状为 [num_samples, 2, n_his, n_vertex] 或 [num_samples, 1, n_his, n_vertex]
        n_his: 历史时间步数，默认为12
        n_vertex: 节点数量，默认为31
    
    返回:
        features: 扁平化的特征矩阵，形状为 [num_samples, 2 * n_his * n_vertex] 或 [num_samples, n_his * n_vertex]
    """
    num_samples = x_data.shape[0]
    # 将数据重塑为扁平特征向量
    features = x_data.reshape(num_samples, -1)
    return features


def prepare_targets(y_data: np.ndarray, indices: List[int] = None) -> np.ndarray:
    """
    准备目标变量（标签）
    
    参数:
        y_data: 标签数据，形状为 [num_samples, 2, n_vertex]
        indices: 要评估的节点索引列表，如果为None则使用所有节点
    
    返回:
        targets: 目标变量，形状为 [num_samples, 2 * len(indices)]
    """
    if indices is not None:
        # 只选择指定的节点
        y_selected = y_data[:, :, indices]  # [num_samples, 2, len(indices)]
    else:
        y_selected = y_data  # [num_samples, 2, n_vertex]
    
    num_samples = y_selected.shape[0]
    # 将数据重塑为 [num_samples, 2 * len(indices)]
    targets = y_selected.reshape(num_samples, -1)
    return targets


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_outputs: int,
    params: Dict = None,
    early_stopping_rounds: int = 10,
    verbose: bool = True,
) -> xgb.XGBRegressor:
    """
    训练XGBoost回归模型（多输出）
    
    为每个输出维度训练一个独立的XGBoost模型
    
    参数:
        X_train: 训练集特征，形状为 [num_train, num_features]
        y_train: 训练集标签，形状为 [num_train, n_outputs]
        X_val: 验证集特征，形状为 [num_val, num_features]
        y_val: 验证集标签，形状为 [num_val, n_outputs]
        n_outputs: 输出维度数量
        params: XGBoost超参数字典
        early_stopping_rounds: 早停轮数
        verbose: 是否打印训练信息
    
    返回:
        models: XGBoost模型列表，每个模型对应一个输出维度
    """
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
    
    models = []
    
    print(f"开始训练XGBoost模型，共 {n_outputs} 个输出维度...")
    for i in tqdm(range(n_outputs), desc="训练模型"):
        # 为每个输出维度训练一个模型
        model = xgb.XGBRegressor(**params)
        
        # 根据XGBoost版本选择不同的早停方式
        if USE_NEW_CALLBACK:
            # 新版本XGBoost (>=2.0) 使用callbacks
            try:
                early_stop = EarlyStopping(
                    rounds=early_stopping_rounds,
                    save_best=True,
                    maximize=False,
                )
                # 训练模型，使用验证集进行早停
                model.fit(
                    X_train,
                    y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    callbacks=[early_stop],
                    verbose=False,
                )
            except Exception as e:
                # 如果callbacks方式失败，回退到不使用早停
                print(f"警告: 早停回调失败 ({e})，将不使用早停")
                model.fit(
                    X_train,
                    y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    verbose=False,
                )
        else:
            # 旧版本XGBoost，尝试使用early_stopping_rounds参数
            try:
                model.fit(
                    X_train,
                    y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            except (TypeError, ValueError) as e:
                # 如果early_stopping_rounds也不支持，则不使用早停
                print(f"警告: 早停参数不支持 ({e})，将不使用早停")
                model.fit(
                    X_train,
                    y_train[:, i],
                    eval_set=[(X_val, y_val[:, i])],
                    verbose=False,
                )
        
        models.append(model)
    
    return models


def predict_with_models(models: List[xgb.XGBRegressor], X: np.ndarray) -> np.ndarray:
    """
    使用多个XGBoost模型进行预测
    
    参数:
        models: XGBoost模型列表
        X: 输入特征，形状为 [num_samples, num_features]
    
    返回:
        predictions: 预测结果，形状为 [num_samples, n_outputs]
    """
    predictions = []
    for model in models:
        pred = model.predict(X)
        predictions.append(pred)
    
    return np.column_stack(predictions)


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scaler: List[StandardScaler],
    indices: List[int],
    n_steam: int = 24,
    is_steam: bool = True,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    计算评估指标（MAE, RMSE, MAPE, WMAPE）
    
    参数:
        y_true: 真实值，形状为 [num_samples, 2 * len(indices)]
        y_pred: 预测值，形状为 [num_samples, 2 * len(indices)]
        scaler: 标准化器列表，用于反标准化
        indices: 节点索引列表
        n_steam: 蒸汽节点数量，默认为24
        is_steam: 是否为蒸汽数据，True表示蒸汽，False表示空气
    
    返回:
        MAE_G, RMSE_G, MAPE_G, WMAPE_G: 流量G的指标
        MAE_P, RMSE_P, MAPE_P, WMAPE_P: 压力P的指标
    """
    # 将预测值和真实值重塑回原始形状
    num_samples = y_true.shape[0]
    y_true_reshaped = y_true.reshape(num_samples, 2, len(indices))
    y_pred_reshaped = y_pred.reshape(num_samples, 2, len(indices))
    
    # 分离流量G和压力P
    y_true_G = y_true_reshaped[:, 0, :]  # [num_samples, len(indices)]
    y_true_P = y_true_reshaped[:, 1, :]  # [num_samples, len(indices)]
    y_pred_G = y_pred_reshaped[:, 0, :]  # [num_samples, len(indices)]
    y_pred_P = y_pred_reshaped[:, 1, :]  # [num_samples, len(indices)]
    
    # 反标准化
    # 需要将数据扩展到完整的节点维度才能反标准化
    # 创建一个临时数组，只填充选中的节点
    if is_steam:
        n_nodes = n_steam
    else:
        n_nodes = 7  # 空气节点数量
    
    # 创建完整维度的数组（使用标准化后的值）
    y_true_G_full = np.zeros((num_samples, n_nodes))
    y_true_P_full = np.zeros((num_samples, n_nodes))
    y_pred_G_full = np.zeros((num_samples, n_nodes))
    y_pred_P_full = np.zeros((num_samples, n_nodes))
    
    # 填充选中的节点（标准化后的值）
    for idx, node_idx in enumerate(indices):
        y_true_G_full[:, node_idx] = y_true_G[:, idx]
        y_true_P_full[:, node_idx] = y_true_P[:, idx]
        y_pred_G_full[:, node_idx] = y_pred_G[:, idx]
        y_pred_P_full[:, node_idx] = y_pred_P[:, idx]
    
    # 反标准化到原始尺度
    y_true_G_orig = scaler[0].inverse_transform(y_true_G_full)
    y_true_P_orig = scaler[1].inverse_transform(y_true_P_full)
    y_pred_G_orig = scaler[0].inverse_transform(y_pred_G_full)
    y_pred_P_orig = scaler[1].inverse_transform(y_pred_P_full)
    
    # 提取选中的节点（原始尺度）
    y_true_G_selected = y_true_G_orig[:, indices].flatten()
    y_true_P_selected = y_true_P_orig[:, indices].flatten()
    y_pred_G_selected = y_pred_G_orig[:, indices].flatten()
    y_pred_P_selected = y_pred_P_orig[:, indices].flatten()
    
    # 计算MAE
    mae_G = mean_absolute_error(y_true_G_selected, y_pred_G_selected)
    mae_P = mean_absolute_error(y_true_P_selected, y_pred_P_selected)
    
    # 计算RMSE
    rmse_G = np.sqrt(mean_squared_error(y_true_G_selected, y_pred_G_selected))
    rmse_P = np.sqrt(mean_squared_error(y_true_P_selected, y_pred_P_selected))
    
    # 计算MAPE（避免除零）
    epsilon = 1e-8
    mape_G = 100.0 * np.mean(np.abs((y_true_G_selected - y_pred_G_selected) / (np.abs(y_true_G_selected) + epsilon)))
    mape_P = 100.0 * np.mean(np.abs((y_true_P_selected - y_pred_P_selected) / (np.abs(y_true_P_selected) + epsilon)))
    
    # 计算WMAPE
    wmape_G = np.sum(np.abs(y_true_G_selected - y_pred_G_selected)) / (np.sum(np.abs(y_true_G_selected)) + epsilon)
    wmape_P = np.sum(np.abs(y_true_P_selected - y_pred_P_selected)) / (np.sum(np.abs(y_true_P_selected)) + epsilon)
    
    return mae_G, rmse_G, mape_G, wmape_G, mae_P, rmse_P, mape_P, wmape_P


def train_and_evaluate(
    data_path: str = "data/data.pickle",
    Sscore_path: str = "data/S_score.pkl",
    Escore_path: str = "data/E_score.pkl",
    n_his: int = 12,
    n_steam: int = 24,
    n_air: int = 7,
    params: Dict = None,
    early_stopping_rounds: int = 10,
):
    """
    完整的训练和评估流程
    
    参数:
        data_path: 数据文件路径
        Sscore_path: 蒸汽标准化器路径
        Escore_path: 电力标准化器路径
        n_his: 历史时间步数
        n_steam: 蒸汽节点数量
        n_air: 空气节点数量
        params: XGBoost超参数
        early_stopping_rounds: 早停轮数
    """
    print("=" * 80)
    print("XGBoost能源系统预测模型")
    print("=" * 80)
    
    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    (
        S_score,
        E_score,
        S_x_train,
        S_y_train,
        S_x_val,
        S_y_val,
        S_x_test,
        S_y_test,
        E_x_train,
        E_y_train,
        E_x_val,
        E_y_val,
        E_x_test,
        E_y_test,
    ) = load_data(data_path, Sscore_path, Escore_path)
    
    # 确保数据是numpy数组
    if not isinstance(S_x_train, np.ndarray):
        S_x_train = np.array(S_x_train)
    if not isinstance(S_y_train, np.ndarray):
        S_y_train = np.array(S_y_train)
    if not isinstance(E_x_train, np.ndarray):
        E_x_train = np.array(E_x_train)
    if not isinstance(E_y_train, np.ndarray):
        E_y_train = np.array(E_y_train)
    
    print(f"训练集大小: {S_x_train.shape[0]}")
    print(f"验证集大小: {S_x_val.shape[0]}")
    print(f"测试集大小: {S_x_test.shape[0]}")
    
    # 2. 准备特征和目标
    print("\n[2/5] 准备特征和目标变量...")
    print(f"原始数据形状 - S_x_train: {S_x_train.shape}, S_y_train: {S_y_train.shape}")
    print(f"原始数据形状 - E_x_train: {E_x_train.shape}, E_y_train: {E_y_train.shape}")
    
    # 蒸汽数据
    S_X_train = prepare_features(S_x_train, n_his, n_steam)
    S_X_val = prepare_features(S_x_val, n_his, n_steam)
    S_X_test = prepare_features(S_x_test, n_his, n_steam)
    
    # 定义要评估的节点索引（与深度学习模型保持一致）
    steam_indices = [0, 3, 4, 6, 10, 12, 14, 17, 19, 21, 23]
    air_indices = [0, 1, 4, 5, 6]
    
    S_y_train_selected = prepare_targets(S_y_train, steam_indices)
    S_y_val_selected = prepare_targets(S_y_val, steam_indices)
    S_y_test_selected = prepare_targets(S_y_test, steam_indices)
    
    # 空气数据
    E_X_train = prepare_features(E_x_train, n_his, n_air)
    E_X_val = prepare_features(E_x_val, n_his, n_air)
    E_X_test = prepare_features(E_x_test, n_his, n_air)
    
    E_y_train_selected = prepare_targets(E_y_train, air_indices)
    E_y_val_selected = prepare_targets(E_y_val, air_indices)
    E_y_test_selected = prepare_targets(E_y_test, air_indices)
    
    print(f"特征维度: {S_X_train.shape[1]}")
    print(f"蒸汽输出维度: {S_y_train_selected.shape[1]}")
    print(f"空气输出维度: {E_y_train_selected.shape[1]}")
    
    # 3. 训练蒸汽模型
    print("\n[3/5] 训练蒸汽XGBoost模型...")
    start_time = time.time()
    S_models = train_xgboost_model(
        S_X_train,
        S_y_train_selected,
        S_X_val,
        S_y_val_selected,
        S_y_train_selected.shape[1],
        params=params,
        early_stopping_rounds=early_stopping_rounds,
    )
    steam_train_time = time.time() - start_time
    print(f"蒸汽模型训练完成，耗时: {steam_train_time:.2f} 秒")
    
    # 4. 训练空气模型
    print("\n[4/5] 训练空气XGBoost模型...")
    start_time = time.time()
    E_models = train_xgboost_model(
        E_X_train,
        E_y_train_selected,
        E_X_val,
        E_y_val_selected,
        E_y_train_selected.shape[1],
        params=params,
        early_stopping_rounds=early_stopping_rounds,
    )
    air_train_time = time.time() - start_time
    print(f"空气模型训练完成，耗时: {air_train_time:.2f} 秒")
    
    # 5. 在验证集上评估
    print("\n[5/5] 在验证集和测试集上评估模型...")
    
    # 验证集预测
    S_y_val_pred = predict_with_models(S_models, S_X_val)
    E_y_val_pred = predict_with_models(E_models, E_X_val)
    
    # 测试集预测
    S_y_test_pred = predict_with_models(S_models, S_X_test)
    E_y_test_pred = predict_with_models(E_models, E_X_test)
    
    # 计算验证集指标
    print("\n" + "=" * 80)
    print("验证集结果:")
    print("=" * 80)
    (
        steam_MAE_G_val,
        steam_RMSE_G_val,
        steam_MAPE_G_val,
        steam_WMAPE_G_val,
        steam_MAE_P_val,
        steam_RMSE_P_val,
        steam_MAPE_P_val,
        steam_WMAPE_P_val,
    ) = evaluate_metrics(S_y_val_selected, S_y_val_pred, S_score, steam_indices, n_steam, is_steam=True)
    
    (
        air_MAE_G_val,
        air_RMSE_G_val,
        air_MAPE_G_val,
        air_WMAPE_G_val,
        air_MAE_P_val,
        air_RMSE_P_val,
        air_MAPE_P_val,
        air_WMAPE_P_val,
    ) = evaluate_metrics(E_y_val_selected, E_y_val_pred, E_score, air_indices, n_steam, is_steam=False)
    
    print(
        f"蒸汽 - 流量G: MAE={steam_MAE_G_val:.6f} | RMSE={steam_RMSE_G_val:.6f} | "
        f"MAPE={steam_MAPE_G_val:.6f}% | WMAPE={steam_WMAPE_G_val:.8f}"
    )
    print(
        f"蒸汽 - 压力P: MAE={steam_MAE_P_val:.6f} | RMSE={steam_RMSE_P_val:.6f} | "
        f"MAPE={steam_MAPE_P_val:.6f}% | WMAPE={steam_WMAPE_P_val:.8f}"
    )
    print(
        f"空气 - 流量G: MAE={air_MAE_G_val:.6f} | RMSE={air_RMSE_G_val:.6f} | "
        f"MAPE={air_MAPE_G_val:.6f}% | WMAPE={air_WMAPE_G_val:.8f}"
    )
    print(
        f"空气 - 压力P: MAE={air_MAE_P_val:.6f} | RMSE={air_RMSE_P_val:.6f} | "
        f"MAPE={air_MAPE_P_val:.6f}% | WMAPE={air_WMAPE_P_val:.8f}"
    )
    
    # 计算测试集指标
    print("\n" + "=" * 80)
    print("测试集结果:")
    print("=" * 80)
    (
        steam_MAE_G_test,
        steam_RMSE_G_test,
        steam_MAPE_G_test,
        steam_WMAPE_G_test,
        steam_MAE_P_test,
        steam_RMSE_P_test,
        steam_MAPE_P_test,
        steam_WMAPE_P_test,
    ) = evaluate_metrics(S_y_test_selected, S_y_test_pred, S_score, steam_indices, n_steam, is_steam=True)
    
    (
        air_MAE_G_test,
        air_RMSE_G_test,
        air_MAPE_G_test,
        air_WMAPE_G_test,
        air_MAE_P_test,
        air_RMSE_P_test,
        air_MAPE_P_test,
        air_WMAPE_P_test,
    ) = evaluate_metrics(E_y_test_selected, E_y_test_pred, E_score, air_indices, n_steam, is_steam=False)
    
    print(
        f"蒸汽 - 流量G: MAE={steam_MAE_G_test:.6f} | RMSE={steam_RMSE_G_test:.6f} | "
        f"MAPE={steam_MAPE_G_test:.6f}% | WMAPE={steam_WMAPE_G_test:.8f}"
    )
    print(
        f"蒸汽 - 压力P: MAE={steam_MAE_P_test:.6f} | RMSE={steam_RMSE_P_test:.6f} | "
        f"MAPE={steam_MAPE_P_test:.6f}% | WMAPE={steam_WMAPE_P_test:.8f}"
    )
    print(
        f"空气 - 流量G: MAE={air_MAE_G_test:.6f} | RMSE={air_RMSE_G_test:.6f} | "
        f"MAPE={air_MAPE_G_test:.6f}% | WMAPE={air_WMAPE_G_test:.8f}"
    )
    print(
        f"空气 - 压力P: MAE={air_MAE_P_test:.6f} | RMSE={air_RMSE_P_test:.6f} | "
        f"MAPE={air_MAPE_P_test:.6f}% | WMAPE={air_WMAPE_P_test:.8f}"
    )
    
    # 计算测试集MSE
    test_MSE_steam = mean_squared_error(S_y_test_selected.flatten(), S_y_test_pred.flatten())
    test_MSE_air = mean_squared_error(E_y_test_selected.flatten(), E_y_test_pred.flatten())
    test_MSE = (test_MSE_steam + test_MSE_air) / 2
    
    print(f"\n测试集MSE损失: {test_MSE:.6f}")
    print(f"总训练时间: {steam_train_time + air_train_time:.2f} 秒")
    print("=" * 80)
    
    return {
        "steam_models": S_models,
        "air_models": E_models,
        "validation_metrics": {
            "steam_MAE_G": steam_MAE_G_val,
            "steam_RMSE_G": steam_RMSE_G_val,
            "steam_MAPE_G": steam_MAPE_G_val,
            "steam_WMAPE_G": steam_WMAPE_G_val,
            "steam_MAE_P": steam_MAE_P_val,
            "steam_RMSE_P": steam_RMSE_P_val,
            "steam_MAPE_P": steam_MAPE_P_val,
            "steam_WMAPE_P": steam_WMAPE_P_val,
            "air_MAE_G": air_MAE_G_val,
            "air_RMSE_G": air_RMSE_G_val,
            "air_MAPE_G": air_MAPE_G_val,
            "air_WMAPE_G": air_WMAPE_G_val,
            "air_MAE_P": air_MAE_P_val,
            "air_RMSE_P": air_RMSE_P_val,
            "air_MAPE_P": air_MAPE_P_val,
            "air_WMAPE_P": air_WMAPE_P_val,
        },
        "test_metrics": {
            "test_MSE": test_MSE,
            "steam_MAE_G": steam_MAE_G_test,
            "steam_RMSE_G": steam_RMSE_G_test,
            "steam_MAPE_G": steam_MAPE_G_test,
            "steam_WMAPE_G": steam_WMAPE_G_test,
            "steam_MAE_P": steam_MAE_P_test,
            "steam_RMSE_P": steam_RMSE_P_test,
            "steam_MAPE_P": steam_MAPE_P_test,
            "steam_WMAPE_P": steam_WMAPE_P_test,
            "air_MAE_G": air_MAE_G_test,
            "air_RMSE_G": air_RMSE_G_test,
            "air_MAPE_G": air_MAPE_G_test,
            "air_WMAPE_G": air_WMAPE_G_test,
            "air_MAE_P": air_MAE_P_test,
            "air_RMSE_P": air_RMSE_P_test,
            "air_MAPE_P": air_MAPE_P_test,
            "air_WMAPE_P": air_WMAPE_P_test,
        },
    }


if __name__ == "__main__":
    """
    主程序入口
    
    执行流程:
    1. 解析命令行参数（可选）
    2. 加载数据
    3. 训练XGBoost模型
    4. 评估模型性能
    """
    parser = argparse.ArgumentParser(description="XGBoost能源系统预测模型")
    parser.add_argument("--data_path", type=str, default="data/data.pickle", help="数据文件路径")
    parser.add_argument("--Sscore_path", type=str, default="data/S_score.pkl", help="蒸汽标准化器路径")
    parser.add_argument("--Escore_path", type=str, default="data/E_score.pkl", help="电力标准化器路径")
    parser.add_argument("--n_his", type=int, default=12, help="历史时间步数")
    parser.add_argument("--n_steam", type=int, default=24, help="蒸汽节点数量")
    parser.add_argument("--n_air", type=int, default=7, help="空气节点数量")
    parser.add_argument("--max_depth", type=int, default=6, help="树的最大深度")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="学习率")
    parser.add_argument("--n_estimators", type=int, default=1000, help="树的数量")
    parser.add_argument("--early_stopping_rounds", type=int, default=10, help="早停轮数")
    
    args = parser.parse_args()
    
    # 设置XGBoost参数
    xgb_params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
    }
    
    # 执行训练和评估
    results = train_and_evaluate(
        data_path=args.data_path,
        Sscore_path=args.Sscore_path,
        Escore_path=args.Escore_path,
        n_his=args.n_his,
        n_steam=args.n_steam,
        n_air=args.n_air,
        params=xgb_params,
        early_stopping_rounds=args.early_stopping_rounds,
    )
    
    print("\n训练和评估完成！")

