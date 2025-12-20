"""
XGBoost方法训练和测试模块

本文件实现了基于XGBoost的能源系统预测模型，用于与深度学习模型进行性能对比。
主要功能包括：
1. 数据加载和特征工程（将图数据转换为表格格式）
2. XGBoost模型训练（合并蒸汽和空气数据统一训练）
3. 模型验证和测试
4. 评估指标计算（MAE, RMSE, MAPE, WMAPE）
5. 超参数自动优化（基于Optuna框架）

注意：
- 蒸汽和空气数据合并训练，使用统一的模型集合
- 已移除早停机制，使用固定轮数训练
- 合并训练时模型规模会适当增大（max_depth=8, n_estimators=1500）
- 超参数优化功能需要安装Optuna：uv pip install optuna plotly

使用方法：
- 普通训练：python xgboost_method_v3.py
- 超参数优化：python xgboost_method_v3.py --optimize_params --n_trials 50
"""

import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
from typing import Tuple, List, Dict, Optional, Any
import os
import time
from tqdm import tqdm
import sys

# 导入Optuna用于超参数优化
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("警告: Optuna未安装，无法使用超参数优化功能。")
    print("安装命令: uv pip install optuna")

# 导入报告生成器
try:
    from script.xgboost_report_generator import XGBoostReportGenerator
except ImportError:
    # 如果直接导入失败，尝试添加路径后导入
    current_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(current_dir, 'script')
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from xgboost_report_generator import XGBoostReportGenerator  # type: ignore

# 已移除早停机制


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
        x_data: 输入数据，形状为 [num_samples, 2, n_his, n_vertex]
        n_his: 历史时间步数，默认为12
        n_vertex: 节点数量，默认为31
    
    返回:
        features: 扁平化的特征矩阵，形状为 [num_samples, 2 * n_his * n_vertex]
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
    verbose: bool = True,
    report_gen: Optional[XGBoostReportGenerator] = None,
) -> List[xgb.XGBRegressor]:
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
        verbose: 是否打印训练信息
        report_gen: 报告生成器对象（可选）
    
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
        # 将eval_metric添加到模型初始化参数中（新版本XGBoost要求）
        model_params = params.copy()
        if 'eval_metric' not in model_params:
            model_params['eval_metric'] = 'rmse'
        
        model = xgb.XGBRegressor(**model_params)
        
        # 训练模型，记录评估结果
        try:
            model.fit(
                X_train,
                y_train[:, i],
                eval_set=[(X_train, y_train[:, i]), (X_val, y_val[:, i])],
                verbose=False,
            )
        except TypeError:
            # 如果eval_set不支持，尝试不使用eval_set
            model.fit(
                X_train,
                y_train[:, i],
                verbose=False,
            )
        
        # 提取训练历史
        if hasattr(model, 'evals_result_') and model.evals_result_:
            # 尝试获取训练和验证损失
            try:
                train_losses = model.evals_result_['validation_0']['rmse']
                val_losses = model.evals_result_['validation_1']['rmse']
                boosting_rounds = list(range(1, len(train_losses) + 1))
                
                # 记录到报告生成器
                if report_gen is not None:
                    report_gen.record_training_history(i, train_losses, val_losses, boosting_rounds)
            except KeyError:
                # 如果键不存在，尝试其他可能的键名
                try:
                    # 尝试不同的键名格式
                    keys = list(model.evals_result_.keys())
                    if len(keys) >= 2:
                        train_key = keys[0]
                        val_key = keys[1]
                        metric_key = list(model.evals_result_[train_key].keys())[0]
                        train_losses = model.evals_result_[train_key][metric_key]
                        val_losses = model.evals_result_[val_key][metric_key]
                        boosting_rounds = list(range(1, len(train_losses) + 1))
                        
                        # 记录到报告生成器
                        if report_gen is not None:
                            report_gen.record_training_history(i, train_losses, val_losses, boosting_rounds)
                except Exception:
                    # 如果无法提取训练历史，跳过记录
                    pass
        
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


def merge_features_and_targets(
    S_X: np.ndarray,
    S_y: np.ndarray,
    E_X: np.ndarray,
    E_y: np.ndarray,
    n_his: int = 12,
    n_steam: int = 24,
    n_air: int = 7,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    合并蒸汽和空气的特征和目标
    
    通过padding将特征维度统一，并添加类型标识特征
    
    参数:
        S_X: 蒸汽特征，形状为 [num_samples, 2 * n_his * n_steam]
        S_y: 蒸汽目标，形状为 [num_samples, n_steam_outputs]
        E_X: 空气特征，形状为 [num_samples, 2 * n_his * n_air]
        E_y: 空气目标，形状为 [num_samples, n_air_outputs]
        n_his: 历史时间步数
        n_steam: 蒸汽节点数量
        n_air: 空气节点数量
    
    返回:
        merged_X: 合并后的特征，形状为 [num_total, max_features + 1]
        merged_y: 合并后的目标，形状为 [num_total, n_steam_outputs + n_air_outputs]
        steam_output_dim: 蒸汽输出维度
        air_output_dim: 空气输出维度
    """
    # 计算最大特征维度
    max_features = max(S_X.shape[1], E_X.shape[1])
    
    # Padding特征到相同维度，并添加类型标识（0表示蒸汽，1表示空气）
    S_X_padded = np.zeros((S_X.shape[0], max_features + 1))
    S_X_padded[:, :S_X.shape[1]] = S_X
    S_X_padded[:, -1] = 0  # 蒸汽类型标识
    
    E_X_padded = np.zeros((E_X.shape[0], max_features + 1))
    E_X_padded[:, :E_X.shape[1]] = E_X
    E_X_padded[:, -1] = 1  # 空气类型标识
    
    # 合并特征
    merged_X = np.vstack([S_X_padded, E_X_padded])
    
    # 合并目标（将蒸汽和空气的输出拼接在一起）
    # 方案：将目标扩展为 [steam_outputs, air_outputs] 的拼接
    # 对于蒸汽样本，前steam_output_dim列有值，后air_output_dim列为0
    # 对于空气样本，前steam_output_dim列为0，后air_output_dim列有值
    steam_output_dim = S_y.shape[1]
    air_output_dim = E_y.shape[1]
    total_output_dim = steam_output_dim + air_output_dim
    
    S_y_extended = np.zeros((S_y.shape[0], total_output_dim))
    S_y_extended[:, :steam_output_dim] = S_y
    
    E_y_extended = np.zeros((E_y.shape[0], total_output_dim))
    E_y_extended[:, steam_output_dim:] = E_y
    
    merged_y = np.vstack([S_y_extended, E_y_extended])
    
    return merged_X, merged_y, steam_output_dim, air_output_dim


def split_predictions(
    predictions: np.ndarray,
    steam_output_dim: int,
    air_output_dim: int,
    steam_indices: np.ndarray,
    air_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从合并的预测结果中分离出蒸汽和空气的预测
    
    参数:
        predictions: 合并的预测结果，形状为 [num_samples, total_output_dim]
        steam_output_dim: 蒸汽输出维度
        air_output_dim: 空气输出维度
        steam_indices: 蒸汽样本的索引数组
        air_indices: 空气样本的索引数组
    
    返回:
        S_predictions: 蒸汽预测结果，形状为 [num_steam, steam_output_dim]
        E_predictions: 空气预测结果，形状为 [num_air, air_output_dim]
    """
    S_predictions = predictions[steam_indices, :steam_output_dim]
    E_predictions = predictions[air_indices, steam_output_dim:]
    return S_predictions, E_predictions


def optimize_hyperparameters(
    merged_X_train: np.ndarray,
    merged_y_train: np.ndarray,
    merged_X_val: np.ndarray,
    merged_y_val: np.ndarray,
    n_outputs: int,
    n_trials: int = 50,
    timeout: Optional[float] = None,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    load_if_exists: bool = False,
) -> Tuple[Dict, Any]:
    """
    使用Optuna进行XGBoost超参数优化
    
    参数:
        merged_X_train: 合并后的训练集特征
        merged_y_train: 合并后的训练集标签
        merged_X_val: 合并后的验证集特征
        merged_y_val: 合并后的验证集标签
        n_outputs: 输出维度数量
        n_trials: 优化试验次数，默认为50
        timeout: 优化超时时间（秒），None表示不限制
        study_name: 研究名称，用于保存优化历史
        storage: 存储路径，用于保存优化历史（SQLite数据库路径）
        load_if_exists: 如果研究已存在，是否加载
    
    返回:
        best_params: 最佳超参数字典
        study: Optuna研究对象（用于后续分析），如果Optuna未安装则返回None
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna未安装，请使用 'uv pip install optuna' 安装")
    
    def objective(trial):
        """Optuna优化目标函数"""
        # 定义超参数搜索空间
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000, step=100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0),
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "rmse",
        }
        
        # 训练模型并评估
        models = []
        val_losses = []
        
        for i in range(n_outputs):
            model = xgb.XGBRegressor(**params)
            try:
                model.fit(
                    merged_X_train,
                    merged_y_train[:, i],
                    eval_set=[(merged_X_train, merged_y_train[:, i]), 
                             (merged_X_val, merged_y_val[:, i])],
                    verbose=False,
                )
            except TypeError:
                model.fit(
                    merged_X_train,
                    merged_y_train[:, i],
                    verbose=False,
                )
            
            models.append(model)
            
            # 获取验证集RMSE
            if hasattr(model, 'evals_result_') and model.evals_result_:
                try:
                    val_rmse = model.evals_result_['validation_1']['rmse'][-1]
                    val_losses.append(val_rmse)
                except KeyError:
                    # 如果无法获取验证损失，使用预测误差
                    y_pred = model.predict(merged_X_val)
                    val_rmse = np.sqrt(mean_squared_error(merged_y_val[:, i], y_pred))
                    val_losses.append(val_rmse)
            else:
                # 如果无法获取验证损失，使用预测误差
                y_pred = model.predict(merged_X_val)
                val_rmse = np.sqrt(mean_squared_error(merged_y_val[:, i], y_pred))
                val_losses.append(val_rmse)
        
        # 返回平均验证RMSE作为优化目标
        mean_val_rmse = np.mean(val_losses)
        
        # 报告中间值（用于剪枝）
        trial.report(mean_val_rmse, step=len(val_losses))
        
        # 检查是否应该剪枝
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        return mean_val_rmse
    
    # 创建或加载研究
    if storage is not None:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
        )
    
    # 执行优化
    print(f"\n开始超参数优化，共 {n_trials} 次试验...")
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
    
    # 输出最佳参数
    print("\n" + "=" * 80)
    print("超参数优化完成！")
    print("=" * 80)
    print(f"最佳验证RMSE: {study.best_value:.6f}")
    print("\n最佳超参数:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # 构建完整的参数字典（包含固定参数）
    best_params = study.best_params.copy()
    best_params.update({
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "random_state": 42,
        "n_jobs": -1,
        "eval_metric": "rmse",
    })
    
    return best_params, study


def train_and_evaluate(
    data_path: str = "data/data.pickle",
    Sscore_path: str = "data/S_score.pkl",
    Escore_path: str = "data/E_score.pkl",
    n_his: int = 12,
    n_steam: int = 24,
    n_air: int = 7,
    params: Dict = None,
    enable_report: bool = True,
    optimize_params: bool = False,
    n_trials: int = 50,
    optimization_timeout: Optional[float] = None,
    optimization_storage: Optional[str] = None,
):
    """
    完整的训练和评估流程（合并蒸汽和空气训练）
    
    参数:
        data_path: 数据文件路径
        Sscore_path: 蒸汽标准化器路径
        Escore_path: 电力标准化器路径
        n_his: 历史时间步数
        n_steam: 蒸汽节点数量
        n_air: 空气节点数量
        params: XGBoost超参数（如果optimize_params=True，此参数将被忽略）
        enable_report: 是否启用报告生成功能
        optimize_params: 是否进行超参数优化，默认为False
        n_trials: 超参数优化试验次数，默认为50
        optimization_timeout: 优化超时时间（秒），None表示不限制
        optimization_storage: 优化历史存储路径（SQLite数据库路径），None表示不保存
    """
    print("=" * 80)
    print("XGBoost能源系统预测模型")
    print("=" * 80)
    
    # 创建报告生成器
    report_gen = None
    if enable_report:
        report_gen = XGBoostReportGenerator(report_dir='reports', prefix='xgboost')
    
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
    
    print(f"蒸汽特征维度: {S_X_train.shape[1]}")
    print(f"空气特征维度: {E_X_train.shape[1]}")
    print(f"蒸汽输出维度: {S_y_train_selected.shape[1]}")
    print(f"空气输出维度: {E_y_train_selected.shape[1]}")
    
    # 3. 合并蒸汽和空气的特征和目标
    print("\n[3/5] 合并蒸汽和空气数据...")
    merged_X_train, merged_y_train, steam_output_dim, air_output_dim = merge_features_and_targets(
        S_X_train, S_y_train_selected, E_X_train, E_y_train_selected, n_his, n_steam, n_air
    )
    merged_X_val, merged_y_val, _, _ = merge_features_and_targets(
        S_X_val, S_y_val_selected, E_X_val, E_y_val_selected, n_his, n_steam, n_air
    )
    merged_X_test, merged_y_test, _, _ = merge_features_and_targets(
        S_X_test, S_y_test_selected, E_X_test, E_y_test_selected, n_his, n_steam, n_air
    )
    
    print(f"合并后特征维度: {merged_X_train.shape[1]}")
    print(f"合并后输出维度: {merged_y_train.shape[1]} (蒸汽: {steam_output_dim}, 空气: {air_output_dim})")
    print(f"合并后训练集大小: {merged_X_train.shape[0]}")
    
    # 记录样本索引，用于后续分离预测结果
    num_steam_train = S_X_train.shape[0]
    num_air_train = E_X_train.shape[0]
    num_steam_val = S_X_val.shape[0]
    num_air_val = E_X_val.shape[0]
    num_steam_test = S_X_test.shape[0]
    num_air_test = E_X_test.shape[0]
    
    # 记录模型和数据信息到报告生成器
    if report_gen is not None:
        report_gen.set_model_info(
            n_outputs=merged_y_train.shape[1],
            n_features=merged_X_train.shape[1],
            steam_output_dim=steam_output_dim,
            air_output_dim=air_output_dim,
            steam_indices=steam_indices,
            air_indices=air_indices,
        )
        report_gen.set_data_info(
            n_train=merged_X_train.shape[0],
            n_val=merged_X_val.shape[0],
            n_test=merged_X_test.shape[0],
            n_steam_train=num_steam_train,
            n_air_train=num_air_train,
        )
    
    steam_train_indices = np.arange(num_steam_train)
    air_train_indices = np.arange(num_steam_train, num_steam_train + num_air_train)
    steam_val_indices = np.arange(num_steam_val)
    air_val_indices = np.arange(num_steam_val, num_steam_val + num_air_val)
    steam_test_indices = np.arange(num_steam_test)
    air_test_indices = np.arange(num_steam_test, num_steam_test + num_air_test)
    
    # 4. 超参数优化（可选）或训练统一的XGBoost模型
    optimization_study = None
    if optimize_params:
        print("\n[4/6] 进行超参数优化...")
        if not OPTUNA_AVAILABLE:
            print("警告: Optuna未安装，跳过超参数优化，使用默认参数")
            optimize_params = False
        else:
            # 生成优化历史存储路径
            if optimization_storage is None and report_gen is not None:
                optimization_storage = os.path.join(
                    report_gen.run_dir, 
                    "optimization_history.db"
                )
            
            best_params, optimization_study = optimize_hyperparameters(
                merged_X_train,
                merged_y_train,
                merged_X_val,
                merged_y_val,
                merged_y_train.shape[1],
                n_trials=n_trials,
                timeout=optimization_timeout,
                study_name="xgboost_optimization",
                storage=f"sqlite:///{optimization_storage}" if optimization_storage else None,
                load_if_exists=False,
            )
            params = best_params
            
            # 保存优化历史可视化
            if report_gen is not None and optimization_study is not None:
                try:
                    import optuna.visualization as vis
                    import plotly
                    
                    # 优化历史图
                    fig_history = vis.plot_optimization_history(optimization_study)
                    history_path = os.path.join(report_gen.run_dir, "images", "optimization_history.html")
                    fig_history.write_html(history_path)
                    
                    # 参数重要性图
                    try:
                        fig_importance = vis.plot_param_importances(optimization_study)
                        importance_path = os.path.join(report_gen.run_dir, "images", "param_importance.html")
                        fig_importance.write_html(importance_path)
                    except Exception:
                        pass  # 如果参数重要性图生成失败，跳过
                    
                    # 参数关系图
                    try:
                        fig_parallel = vis.plot_parallel_coordinate(optimization_study)
                        parallel_path = os.path.join(report_gen.run_dir, "images", "parallel_coordinate.html")
                        fig_parallel.write_html(parallel_path)
                    except Exception:
                        pass  # 如果参数关系图生成失败，跳过
                    
                    print(f"\n优化历史可视化已保存至: {os.path.dirname(history_path)}")
                except ImportError:
                    print("警告: plotly未安装，无法生成优化历史可视化")
                    print("安装命令: uv pip install plotly")
                except Exception as e:
                    print(f"警告: 生成优化历史可视化时出错: {e}")
    
    # 训练统一的XGBoost模型
    if not optimize_params:
        print("\n[4/5] 训练统一的XGBoost模型（合并蒸汽和空气）...")
    else:
        print("\n[5/6] 使用优化后的超参数训练统一的XGBoost模型...")
    
    # 增大模型规模以适应合并后的数据
    if params is None:
        params = {
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "max_depth": 8,  # 增加深度
            "learning_rate": 0.1,
            "n_estimators": 1500,  # 增加树的数量
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "n_jobs": -1,
        }
    else:
        # 如果提供了参数，适当增大规模（仅在未优化时）
        if not optimize_params:
            if params.get("max_depth", 6) < 8:
                params["max_depth"] = 8
            if params.get("n_estimators", 1000) < 1500:
                params["n_estimators"] = 1500
    
    # 记录超参数到报告生成器
    if report_gen is not None:
        hyperparams_to_save = params.copy()
        if optimize_params and optimization_study is not None:
            hyperparams_to_save["optimization_best_value"] = optimization_study.best_value
            hyperparams_to_save["optimization_n_trials"] = len(optimization_study.trials)
        report_gen.set_hyperparameters(hyperparams_to_save)
    
    start_time = time.time()
    unified_models = train_xgboost_model(
        merged_X_train,
        merged_y_train,
        merged_X_val,
        merged_y_val,
        merged_y_train.shape[1],
        params=params,
        report_gen=report_gen,
    )
    train_time = time.time() - start_time
    print(f"统一模型训练完成，耗时: {train_time:.2f} 秒")
    
    # 保存模型权重
    if report_gen is not None:
        report_gen.save_models(unified_models, prefix="xgboost_model")
    
    # 5. 在验证集和测试集上评估
    if not optimize_params:
        print("\n[5/5] 在验证集和测试集上评估模型...")
    else:
        print("\n[6/6] 在验证集和测试集上评估模型...")
    
    # 验证集预测
    merged_y_val_pred = predict_with_models(unified_models, merged_X_val)
    S_y_val_pred, E_y_val_pred = split_predictions(
        merged_y_val_pred, steam_output_dim, air_output_dim, steam_val_indices, air_val_indices
    )
    
    # 测试集预测
    merged_y_test_pred = predict_with_models(unified_models, merged_X_test)
    S_y_test_pred, E_y_test_pred = split_predictions(
        merged_y_test_pred, steam_output_dim, air_output_dim, steam_test_indices, air_test_indices
    )
    
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
    
    # 记录验证集指标到报告生成器
    if report_gen is not None:
        validation_metrics = {
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
        }
        report_gen.set_validation_metrics(validation_metrics)
        
        # 生成验证集可视化
        report_gen.plot_predictions_vs_actual(
            S_y_val_selected, S_y_val_pred,
            title="验证集 - 蒸汽系统预测值 vs 真实值",
            filename="val_steam_predictions_vs_actual.png"
        )
        report_gen.plot_predictions_vs_actual(
            E_y_val_selected, E_y_val_pred,
            title="验证集 - 空气系统预测值 vs 真实值",
            filename="val_air_predictions_vs_actual.png"
        )
        report_gen.plot_residuals(
            S_y_val_selected, S_y_val_pred,
            title="验证集 - 蒸汽系统残差图",
            filename="val_steam_residuals.png"
        )
        report_gen.plot_residuals(
            E_y_val_selected, E_y_val_pred,
            title="验证集 - 空气系统残差图",
            filename="val_air_residuals.png"
        )
    
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
    print(f"总训练时间: {train_time:.2f} 秒")
    print("=" * 80)
    
    # 记录测试集指标到报告生成器
    if report_gen is not None:
        test_metrics = {
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
        }
        report_gen.set_test_metrics(test_metrics)
        
        # 生成测试集可视化
        report_gen.plot_predictions_vs_actual(
            S_y_test_selected, S_y_test_pred,
            title="测试集 - 蒸汽系统预测值 vs 真实值",
            filename="test_steam_predictions_vs_actual.png"
        )
        report_gen.plot_predictions_vs_actual(
            E_y_test_selected, E_y_test_pred,
            title="测试集 - 空气系统预测值 vs 真实值",
            filename="test_air_predictions_vs_actual.png"
        )
        report_gen.plot_residuals(
            S_y_test_selected, S_y_test_pred,
            title="测试集 - 蒸汽系统残差图",
            filename="test_steam_residuals.png"
        )
        report_gen.plot_residuals(
            E_y_test_selected, E_y_test_pred,
            title="测试集 - 空气系统残差图",
            filename="test_air_residuals.png"
        )
        
        # 生成特征重要性图
        report_gen.plot_feature_importance(unified_models, top_n=30)
        
        # 生成并保存报告
        report_path = report_gen.generate_markdown_report()
        print(f"\n训练报告已生成: {report_path}")
    
    return {
        "unified_models": unified_models,
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
    parser.add_argument("--max_depth", type=int, default=8, help="树的最大深度（合并训练建议使用8）")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="学习率")
    parser.add_argument("--n_estimators", type=int, default=1500, help="树的数量（合并训练建议使用1500）")
    parser.add_argument("--optimize_params", action="store_true", help="是否进行超参数优化")
    parser.add_argument("--n_trials", type=int, default=50, help="超参数优化试验次数（默认50）")
    parser.add_argument("--optimization_timeout", type=float, default=None, help="优化超时时间（秒），None表示不限制")
    parser.add_argument("--optimization_storage", type=str, default=None, help="优化历史存储路径（SQLite数据库路径）")
    
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
        optimize_params=args.optimize_params,
        n_trials=args.n_trials,
        optimization_timeout=args.optimization_timeout,
        optimization_storage=args.optimization_storage,
    )
    
    print("\n训练和评估完成！")

