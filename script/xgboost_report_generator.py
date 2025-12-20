"""
XGBoost报告生成器模块

本模块提供了XGBoost训练报告生成功能，包括：
1. 收集训练过程中的各种指标
2. 生成可视化图表（损失曲线、特征重要性、预测对比等）
3. 生成markdown格式的完整报告
4. 保存模型权重
"""

import os
import json
import time
import pickle
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适用于无GUI环境
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional
import xgboost as xgb

# 设置matplotlib中文字体支持
def set_plt_chinese():
    """设置matplotlib的中文显示，避免中文乱码。"""
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    except Exception as e:
        pass  # 若发生异常则忽略

set_plt_chinese()


class XGBoostReportGenerator:
    """
    XGBoost训练报告生成器

    用于收集XGBoost训练过程中的各种信息，并生成完整的markdown报告。
    """

    def __init__(self, report_dir='reports', prefix='xgboost'):
        """
        初始化报告生成器

        参数:
            report_dir: 报告存储目录
            prefix: 报告文件夹前缀，默认为'xgboost'
        """
        self.report_dir = report_dir
        self.start_time = time.time()
        self.end_time = None

        # 创建带时间戳的报告目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(report_dir, f"{prefix}_run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "models"), exist_ok=True)

        # 训练历史数据（每个输出维度的训练过程）
        self.train_losses = []  # List[List[float]]，每个输出维度的训练损失历史
        self.val_losses = []    # List[List[float]]，每个输出维度的验证损失历史
        self.boosting_rounds = []  # List[List[int]]，每个输出维度的boosting轮数

        # 模型信息
        self.model_info = {}
        self.hyperparameters = {}
        self.data_info = {}

        # 验证和测试结果
        self.validation_metrics = {}
        self.test_metrics = {}

        # 保存的模型路径
        self.saved_models = []

        print(f"XGBoost报告将保存至: {self.run_dir}")

    def record_training_history(self, output_idx: int, train_losses: List[float], 
                                val_losses: List[float], boosting_rounds: List[int]):
        """
        记录一个输出维度的训练历史

        参数:
            output_idx: 输出维度索引
            train_losses: 训练损失历史
            val_losses: 验证损失历史
            boosting_rounds: boosting轮数历史
        """
        # 确保列表足够长
        while len(self.train_losses) <= output_idx:
            self.train_losses.append([])
            self.val_losses.append([])
            self.boosting_rounds.append([])
        
        self.train_losses[output_idx] = train_losses
        self.val_losses[output_idx] = val_losses
        self.boosting_rounds[output_idx] = boosting_rounds

    def set_model_info(self, n_outputs: int, n_features: int, 
                       steam_output_dim: int, air_output_dim: int,
                       steam_indices: List[int], air_indices: List[int]):
        """
        记录模型信息

        参数:
            n_outputs: 总输出维度数
            n_features: 特征维度数
            steam_output_dim: 蒸汽输出维度数
            air_output_dim: 空气输出维度数
            steam_indices: 蒸汽节点索引列表
            air_indices: 空气节点索引列表
        """
        self.model_info = {
            'model_type': 'XGBoost',
            'n_outputs': n_outputs,
            'n_features': n_features,
            'steam_output_dim': steam_output_dim,
            'air_output_dim': air_output_dim,
            'steam_indices': steam_indices,
            'air_indices': air_indices,
            'model_count': n_outputs,  # 每个输出维度一个模型
        }

    def set_hyperparameters(self, params: Dict):
        """
        记录超参数

        参数:
            params: XGBoost超参数字典
        """
        self.hyperparameters = params.copy()

    def set_data_info(self, n_train: int, n_val: int, n_test: int,
                      n_steam_train: int, n_air_train: int):
        """
        记录数据信息

        参数:
            n_train: 训练集总样本数
            n_val: 验证集总样本数
            n_test: 测试集总样本数
            n_steam_train: 蒸汽训练样本数
            n_air_train: 空气训练样本数
        """
        self.data_info = {
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'n_steam_train': n_steam_train,
            'n_air_train': n_air_train,
        }

    def set_validation_metrics(self, metrics: Dict):
        """
        记录验证集指标

        参数:
            metrics: 验证集指标字典
        """
        self.validation_metrics = metrics.copy()

    def set_test_metrics(self, metrics: Dict):
        """
        记录测试集指标

        参数:
            metrics: 测试集指标字典
        """
        self.test_metrics = metrics.copy()

    def add_saved_model(self, model_path: str):
        """
        记录保存的模型路径

        参数:
            model_path: 模型文件路径
        """
        self.saved_models.append(model_path)

    def save_models(self, models: List[xgb.XGBRegressor], prefix: str = "model"):
        """
        保存XGBoost模型

        参数:
            models: XGBoost模型列表
            prefix: 模型文件名前缀
        """
        for i, model in enumerate(models):
            model_path = os.path.join(self.run_dir, "models", f"{prefix}_{i}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            self.add_saved_model(model_path)
            print(f"模型 {i} 已保存至: {model_path}")

    def plot_loss_curves(self):
        """
        绘制训练和验证损失曲线

        返回:
            图片保存路径列表
        """
        if not self.train_losses or not any(self.train_losses):
            return None

        set_plt_chinese()

        # 计算平均损失（跨所有输出维度）
        max_rounds = max(len(losses) for losses in self.train_losses if losses)
        if max_rounds == 0:
            return None

        avg_train_losses = []
        avg_val_losses = []
        rounds = []

        for round_idx in range(max_rounds):
            train_vals = []
            val_vals = []
            for output_losses in self.train_losses:
                if round_idx < len(output_losses):
                    train_vals.append(output_losses[round_idx])
            for output_losses in self.val_losses:
                if round_idx < len(output_losses):
                    val_vals.append(output_losses[round_idx])
            
            if train_vals:
                avg_train_losses.append(np.mean(train_vals))
                rounds.append(round_idx + 1)
            if val_vals:
                avg_val_losses.append(np.mean(val_vals))

        if not rounds:
            return None

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(rounds, avg_train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(rounds, avg_val_losses, 'r-', label='验证损失', linewidth=2)
        plt.xlabel('Boosting轮数', fontsize=12)
        plt.ylabel('损失', fontsize=12)
        plt.title('训练和验证损失曲线', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(rounds, avg_train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(rounds, avg_val_losses, 'r-', label='验证损失', linewidth=2)
        plt.xlabel('Boosting轮数', fontsize=12)
        plt.ylabel('损失 (对数尺度)', fontsize=12)
        plt.title('训练和验证损失曲线（对数尺度）', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = os.path.join(self.run_dir, "images", "loss_curves.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_feature_importance(self, models: List[xgb.XGBRegressor], top_n: int = 20):
        """
        绘制特征重要性图

        参数:
            models: XGBoost模型列表
            top_n: 显示前N个重要特征

        返回:
            图片保存路径
        """
        if not models:
            return None

        set_plt_chinese()

        # 计算平均特征重要性
        n_features = models[0].n_features_in_
        importance_sum = np.zeros(n_features)

        for model in models:
            importance = model.feature_importances_
            importance_sum += importance

        avg_importance = importance_sum / len(models)

        # 获取top_n特征
        top_indices = np.argsort(avg_importance)[-top_n:][::-1]
        top_importance = avg_importance[top_indices]

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_indices)), top_importance, color='steelblue')
        plt.yticks(range(len(top_indices)), [f'特征{idx}' for idx in top_indices])
        plt.xlabel('重要性', fontsize=12)
        plt.ylabel('特征索引', fontsize=12)
        plt.title(f'特征重要性（Top {top_n}）', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        save_path = os.path.join(self.run_dir, "images", "feature_importance.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    title: str = "预测值 vs 真实值", 
                                    filename: str = "predictions_vs_actual.png"):
        """
        绘制预测值vs真实值散点图

        参数:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            filename: 保存文件名

        返回:
            图片保存路径
        """
        set_plt_chinese()

        plt.figure(figsize=(10, 10))
        
        # 展平数组
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()

        # 计算R²
        ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
        ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # 绘制散点图
        plt.scatter(y_true_flat, y_pred_flat, alpha=0.5, s=10)
        
        # 绘制对角线
        min_val = min(np.min(y_true_flat), np.min(y_pred_flat))
        max_val = max(np.max(y_true_flat), np.max(y_pred_flat))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='完美预测线')

        plt.xlabel('真实值', fontsize=12)
        plt.ylabel('预测值', fontsize=12)
        plt.title(f'{title}\nR² = {r2:.4f}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.run_dir, "images", filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                       title: str = "残差图", filename: str = "residuals.png"):
        """
        绘制残差图

        参数:
            y_true: 真实值
            y_pred: 预测值
            title: 图表标题
            filename: 保存文件名

        返回:
            图片保存路径
        """
        set_plt_chinese()

        residuals = y_true.flatten() - y_pred.flatten()

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.scatter(y_pred.flatten(), residuals, alpha=0.5, s=10)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('预测值', fontsize=12)
        plt.ylabel('残差', fontsize=12)
        plt.title(f'{title} - 残差vs预测值', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        plt.xlabel('残差', fontsize=12)
        plt.ylabel('频数', fontsize=12)
        plt.title(f'{title} - 残差分布', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        save_path = os.path.join(self.run_dir, "images", filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def finish(self):
        """
        完成报告生成，记录结束时间
        """
        self.end_time = time.time()

    def generate_markdown_report(self):
        """
        生成完整的markdown格式报告

        返回:
            报告文件路径
        """
        self.finish()

        # 计算运行时间
        total_time = self.end_time - self.start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        time_str = f"{hours}小时 {minutes}分钟 {seconds}秒"

        # 生成可视化图表
        loss_curve_path = self.plot_loss_curves()
        feature_importance_path = None  # 需要模型对象，稍后在主函数中调用

        # 生成markdown内容
        md_content = []

        # 标题
        md_content.append("# XGBoost训练报告\n")
        md_content.append(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        md_content.append("---\n\n")

        # 1. 运行概览
        md_content.append("## 1. 运行概览\n\n")
        md_content.append(f"- **总运行时间**: {time_str}\n")
        md_content.append(f"- **模型数量**: {self.model_info.get('model_count', 'N/A')} 个（每个输出维度一个模型）\n")
        md_content.append(f"- **特征维度**: {self.model_info.get('n_features', 'N/A')}\n")
        md_content.append(f"- **输出维度**: {self.model_info.get('n_outputs', 'N/A')} (蒸汽: {self.model_info.get('steam_output_dim', 'N/A')}, 空气: {self.model_info.get('air_output_dim', 'N/A')})\n")
        if self.test_metrics:
            md_content.append(f"- **测试MSE**: {self.test_metrics.get('test_MSE', 'N/A'):.6f}\n")
        md_content.append("\n")

        # 2. 数据信息
        md_content.append("## 2. 数据信息\n\n")
        md_content.append(f"- **训练集大小**: {self.data_info.get('n_train', 'N/A')} (蒸汽: {self.data_info.get('n_steam_train', 'N/A')}, 空气: {self.data_info.get('n_air_train', 'N/A')})\n")
        md_content.append(f"- **验证集大小**: {self.data_info.get('n_val', 'N/A')}\n")
        md_content.append(f"- **测试集大小**: {self.data_info.get('n_test', 'N/A')}\n")
        md_content.append(f"- **蒸汽节点索引**: {self.model_info.get('steam_indices', [])}\n")
        md_content.append(f"- **空气节点索引**: {self.model_info.get('air_indices', [])}\n")
        md_content.append("\n")

        # 3. 模型结构
        md_content.append("## 3. 模型结构\n\n")
        md_content.append(f"- **模型类型**: {self.model_info.get('model_type', 'N/A')}\n")
        md_content.append(f"- **模型数量**: {self.model_info.get('model_count', 'N/A')} 个独立的XGBoost回归模型\n")
        md_content.append(f"- **每个模型对应一个输出维度**: 采用多输出回归策略\n")
        md_content.append("\n")

        # 4. 超参数配置
        md_content.append("## 4. 超参数配置\n\n")
        md_content.append("### 4.1 XGBoost参数\n\n")
        for key, value in self.hyperparameters.items():
            md_content.append(f"- **{key}**: {value}\n")
        md_content.append("\n")

        md_content.append("### 4.2 完整参数列表\n\n")
        md_content.append("<details>\n<summary>点击展开完整参数列表</summary>\n\n")
        md_content.append("```json\n")
        md_content.append(json.dumps(self.hyperparameters, indent=2, ensure_ascii=False))
        md_content.append("\n```\n\n")
        md_content.append("</details>\n\n")

        # 5. 训练过程
        md_content.append("## 5. 训练过程\n\n")
        if loss_curve_path:
            md_content.append("### 5.1 损失曲线\n\n")
            rel_path = os.path.relpath(loss_curve_path, self.run_dir)
            md_content.append(f"![损失曲线]({rel_path})\n\n")
        else:
            md_content.append("训练过程未记录损失历史。\n\n")

        # 6. 验证过程
        md_content.append("## 6. 验证过程\n\n")
        if self.validation_metrics:
            md_content.append("### 6.1 验证集指标\n\n")
            md_content.append("#### 蒸汽系统\n\n")
            md_content.append("| 指标 | 流量G | 压力P |\n")
            md_content.append("|------|-------|-------|\n")
            md_content.append(f"| MAE | {self.validation_metrics.get('steam_MAE_G', 'N/A'):.6f} | "
                            f"{self.validation_metrics.get('steam_MAE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| RMSE | {self.validation_metrics.get('steam_RMSE_G', 'N/A'):.6f} | "
                            f"{self.validation_metrics.get('steam_RMSE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| MAPE | {self.validation_metrics.get('steam_MAPE_G', 'N/A'):.6f}% | "
                            f"{self.validation_metrics.get('steam_MAPE_P', 'N/A'):.6f}% |\n")
            md_content.append(f"| WMAPE | {self.validation_metrics.get('steam_WMAPE_G', 'N/A'):.8f} | "
                            f"{self.validation_metrics.get('steam_WMAPE_P', 'N/A'):.8f} |\n")
            md_content.append("\n")

            md_content.append("#### 空气系统\n\n")
            md_content.append("| 指标 | 流量G | 压力P |\n")
            md_content.append("|------|-------|-------|\n")
            md_content.append(f"| MAE | {self.validation_metrics.get('air_MAE_G', 'N/A'):.6f} | "
                            f"{self.validation_metrics.get('air_MAE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| RMSE | {self.validation_metrics.get('air_RMSE_G', 'N/A'):.6f} | "
                            f"{self.validation_metrics.get('air_RMSE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| MAPE | {self.validation_metrics.get('air_MAPE_G', 'N/A'):.6f}% | "
                            f"{self.validation_metrics.get('air_MAPE_P', 'N/A'):.6f}% |\n")
            md_content.append(f"| WMAPE | {self.validation_metrics.get('air_WMAPE_G', 'N/A'):.8f} | "
                            f"{self.validation_metrics.get('air_WMAPE_P', 'N/A'):.8f} |\n")
            md_content.append("\n")
        else:
            md_content.append("验证结果未记录。\n\n")

        # 7. 测试过程
        md_content.append("## 7. 测试过程\n\n")
        if self.test_metrics:
            md_content.append("### 7.1 测试损失\n\n")
            md_content.append(f"- **测试MSE**: {self.test_metrics.get('test_MSE', 'N/A'):.6f}\n\n")

            md_content.append("### 7.2 详细评估指标\n\n")
            md_content.append("#### 蒸汽系统\n\n")
            md_content.append("| 指标 | 流量G | 压力P |\n")
            md_content.append("|------|-------|-------|\n")
            md_content.append(f"| MAE | {self.test_metrics.get('steam_MAE_G', 'N/A'):.6f} | "
                            f"{self.test_metrics.get('steam_MAE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| RMSE | {self.test_metrics.get('steam_RMSE_G', 'N/A'):.6f} | "
                            f"{self.test_metrics.get('steam_RMSE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| MAPE | {self.test_metrics.get('steam_MAPE_G', 'N/A'):.6f}% | "
                            f"{self.test_metrics.get('steam_MAPE_P', 'N/A'):.6f}% |\n")
            md_content.append(f"| WMAPE | {self.test_metrics.get('steam_WMAPE_G', 'N/A'):.8f} | "
                            f"{self.test_metrics.get('steam_WMAPE_P', 'N/A'):.8f} |\n")
            md_content.append("\n")

            md_content.append("#### 空气系统\n\n")
            md_content.append("| 指标 | 流量G | 压力P |\n")
            md_content.append("|------|-------|-------|\n")
            md_content.append(f"| MAE | {self.test_metrics.get('air_MAE_G', 'N/A'):.6f} | "
                            f"{self.test_metrics.get('air_MAE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| RMSE | {self.test_metrics.get('air_RMSE_G', 'N/A'):.6f} | "
                            f"{self.test_metrics.get('air_RMSE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| MAPE | {self.test_metrics.get('air_MAPE_G', 'N/A'):.6f}% | "
                            f"{self.test_metrics.get('air_MAPE_P', 'N/A'):.6f}% |\n")
            md_content.append(f"| WMAPE | {self.test_metrics.get('air_WMAPE_G', 'N/A'):.8f} | "
                            f"{self.test_metrics.get('air_WMAPE_P', 'N/A'):.8f} |\n")
            md_content.append("\n")
        else:
            md_content.append("测试结果未记录。\n\n")

        # 8. 保存的模型权重
        md_content.append("## 8. 保存的模型权重\n\n")
        if self.saved_models:
            md_content.append("以下模型权重文件已保存：\n\n")
            for model_path in self.saved_models:
                # 获取文件大小
                if os.path.exists(model_path):
                    size_mb = os.path.getsize(model_path) / (1024 * 1024)
                    md_content.append(f"- `{model_path}` ({size_mb:.2f} MB)\n")
                else:
                    md_content.append(f"- `{model_path}` (文件不存在)\n")
        else:
            md_content.append("未保存模型权重文件。\n")
        md_content.append("\n")

        # 9. 模型最终结果总结
        md_content.append("## 9. 模型最终结果总结\n\n")
        if self.test_metrics:
            md_content.append("### 9.1 测试性能\n\n")
            md_content.append("模型在测试集上的表现：\n\n")
            md_content.append("- **蒸汽流量预测**: ")
            md_content.append(f"MAE={self.test_metrics.get('steam_MAE_G', 'N/A'):.6f}, ")
            md_content.append(f"RMSE={self.test_metrics.get('steam_RMSE_G', 'N/A'):.6f}, ")
            md_content.append(f"MAPE={self.test_metrics.get('steam_MAPE_G', 'N/A'):.6f}%, ")
            md_content.append(f"WMAPE={self.test_metrics.get('steam_WMAPE_G', 'N/A'):.8f}\n\n")

            md_content.append("- **蒸汽压力预测**: ")
            md_content.append(f"MAE={self.test_metrics.get('steam_MAE_P', 'N/A'):.6f}, ")
            md_content.append(f"RMSE={self.test_metrics.get('steam_RMSE_P', 'N/A'):.6f}, ")
            md_content.append(f"MAPE={self.test_metrics.get('steam_MAPE_P', 'N/A'):.6f}%, ")
            md_content.append(f"WMAPE={self.test_metrics.get('steam_WMAPE_P', 'N/A'):.8f}\n\n")

            md_content.append("- **空气流量预测**: ")
            md_content.append(f"MAE={self.test_metrics.get('air_MAE_G', 'N/A'):.6f}, ")
            md_content.append(f"RMSE={self.test_metrics.get('air_RMSE_G', 'N/A'):.6f}, ")
            md_content.append(f"MAPE={self.test_metrics.get('air_MAPE_G', 'N/A'):.6f}%, ")
            md_content.append(f"WMAPE={self.test_metrics.get('air_WMAPE_G', 'N/A'):.8f}\n\n")

            md_content.append("- **空气压力预测**: ")
            md_content.append(f"MAE={self.test_metrics.get('air_MAE_P', 'N/A'):.6f}, ")
            md_content.append(f"RMSE={self.test_metrics.get('air_RMSE_P', 'N/A'):.6f}, ")
            md_content.append(f"MAPE={self.test_metrics.get('air_MAPE_P', 'N/A'):.6f}%, ")
            md_content.append(f"WMAPE={self.test_metrics.get('air_WMAPE_P', 'N/A'):.8f}\n\n")

        # 10. 可视化图片
        md_content.append("## 10. 可视化图片\n\n")
        md_content.append("所有生成的可视化图片已保存在 `images/` 目录下。\n\n")
        
        # 检查并添加可视化图片
        image_files = []
        if loss_curve_path:
            image_files.append(("loss_curves.png", "训练损失曲线"))
        
        other_images = [
            ("feature_importance.png", "特征重要性图"),
            ("val_steam_predictions_vs_actual.png", "验证集 - 蒸汽系统预测对比"),
            ("val_air_predictions_vs_actual.png", "验证集 - 空气系统预测对比"),
            ("val_steam_residuals.png", "验证集 - 蒸汽系统残差图"),
            ("val_air_residuals.png", "验证集 - 空气系统残差图"),
            ("test_steam_predictions_vs_actual.png", "测试集 - 蒸汽系统预测对比"),
            ("test_air_predictions_vs_actual.png", "测试集 - 空气系统预测对比"),
            ("test_steam_residuals.png", "测试集 - 蒸汽系统残差图"),
            ("test_air_residuals.png", "测试集 - 空气系统残差图"),
        ]
        
        for filename, title in other_images:
            img_path = os.path.join(self.run_dir, "images", filename)
            if os.path.exists(img_path):
                image_files.append((filename, title))
        
        # 添加所有存在的图片
        for idx, (filename, title) in enumerate(image_files, 1):
            img_path = os.path.join(self.run_dir, "images", filename)
            md_content.append(f"### 10.{idx} {title}\n\n")
            md_content.append(f"![{title}]({os.path.relpath(img_path, self.run_dir)})\n\n")

        # 保存报告
        report_path = os.path.join(self.run_dir, "report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(''.join(md_content))

        print(f"报告已生成: {report_path}")
        return report_path

