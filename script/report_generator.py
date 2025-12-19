"""
报告生成器模块

本模块提供了训练报告生成功能，包括：
1. 收集训练过程中的各种指标
2. 生成可视化图表（loss曲线、学习率曲线等）
3. 生成markdown格式的完整报告
"""

import os
import json
import time
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适用于无GUI环境
import matplotlib.pyplot as plt
import numpy as np
import torch

# 设置matplotlib中文字体支持
def set_plt_chinese():
    """设置matplotlib的中文显示，避免中文乱码。"""
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    except Exception as e:
        pass  # 若发生异常则忽略

set_plt_chinese()

class ReportGenerator:
    """
    训练报告生成器

    用于收集训练过程中的各种信息，并生成完整的markdown报告。
    """

    def __init__(self, report_dir='reports'):
        """
        初始化报告生成器

        参数:
            report_dir: 报告存储目录
        """
        self.report_dir = report_dir
        self.start_time = time.time()
        self.end_time = None

        # 创建带时间戳的报告目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(report_dir, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "images"), exist_ok=True)

        # 训练历史数据
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.epochs_completed = []
        self.gpu_memories = []

        # 模型信息
        self.model_info = {}
        self.hyperparameters = {}

        # 测试结果
        self.test_results = {}

        # 保存的模型路径
        self.saved_models = []

        print(f"报告将保存至: {self.run_dir}")

    def record_epoch(self, epoch, train_loss, val_loss, lr, gpu_mem=0):
        """
        记录一个epoch的训练信息

        参数:
            epoch: 当前epoch编号
            train_loss: 训练损失
            val_loss: 验证损失
            lr: 当前学习率
            gpu_mem: GPU内存使用量（MB）
        """
        self.epochs_completed.append(epoch + 1)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.learning_rates.append(lr)
        self.gpu_memories.append(gpu_mem)

    def set_model_info(self, model, args, blocks):
        """
        记录模型信息

        参数:
            model: PyTorch模型
            args: 配置参数对象
            blocks: 模型结构配置
        """
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.model_info = {
            'model_name': model.__class__.__name__,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_structure': str(model),
            'blocks': blocks
        }

        # 记录所有超参数
        self.hyperparameters = vars(args) if hasattr(args, '__dict__') else args

    def set_test_results(self, test_mse, metrics):
        """
        记录测试结果

        参数:
            test_mse: 测试集MSE损失
            metrics: 详细指标字典，包含MAE、RMSE、WMAPE等
        """
        self.test_results = {
            'test_mse': test_mse,
            **metrics
        }

    def add_saved_model(self, model_path):
        """
        记录保存的模型路径

        参数:
            model_path: 模型文件路径
        """
        self.saved_models.append(model_path)

    def plot_loss_curves(self):
        """
        绘制训练和验证损失曲线

        返回:
            图片保存路径
        """
        if len(self.train_losses) == 0:
            return None

        set_plt_chinese()  # 保证每次都设置中文支持

        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.plot(self.epochs_completed, self.train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(self.epochs_completed, self.val_losses, 'r-', label='验证损失', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('损失', fontsize=12)  # 支持中文
        plt.title('训练和验证损失曲线', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(self.epochs_completed, self.train_losses, 'b-', label='训练损失', linewidth=2)
        plt.plot(self.epochs_completed, self.val_losses, 'r-', label='验证损失', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
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

    def plot_learning_rate(self):
        """
        绘制学习率变化曲线

        返回:
            图片保存路径
        """
        if len(self.learning_rates) == 0:
            return None

        set_plt_chinese()  # 保证每次都设置中文支持

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs_completed, self.learning_rates, 'g-', linewidth=2, marker='o', markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('学习率', fontsize=12)
        plt.title('学习率变化曲线', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.run_dir, "images", "learning_rate.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        return save_path

    def plot_gpu_memory(self):
        """
        绘制GPU内存使用曲线

        返回:
            图片保存路径
        """
        if len(self.gpu_memories) == 0 or all(m == 0 for m in self.gpu_memories):
            return None

        set_plt_chinese()  # 保证每次都设置中文支持

        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs_completed, self.gpu_memories, 'm-', linewidth=2, marker='s', markersize=4)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('GPU内存 (MB)', fontsize=12)
        plt.title('GPU内存使用情况', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        save_path = os.path.join(self.run_dir, "images", "gpu_memory.png")
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
        lr_curve_path = self.plot_learning_rate()
        gpu_mem_path = self.plot_gpu_memory()

        # 生成markdown内容
        md_content = []

        # 标题
        md_content.append("# 训练报告\n")
        md_content.append(f"**生成时间**: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n\n")
        md_content.append("---\n\n")

        # 1. 运行概览
        md_content.append("## 1. 运行概览\n\n")
        md_content.append(f"- **总运行时间**: {time_str}\n")
        md_content.append(f"- **训练轮数**: {len(self.epochs_completed)} epochs\n")
        md_content.append(f"- **最终训练损失**: {self.train_losses[-1]:.6f}\n")
        md_content.append(f"- **最终验证损失**: {self.val_losses[-1]:.6f}\n")
        if self.test_results:
            md_content.append(f"- **测试损失**: {self.test_results.get('test_mse', 'N/A'):.6f}\n")
        md_content.append("\n")

        # 2. 模型结构
        md_content.append("## 2. 模型结构\n\n")
        md_content.append(f"### 2.1 模型基本信息\n\n")
        md_content.append(f"- **模型名称**: {self.model_info.get('model_name', 'N/A')}\n")
        md_content.append(f"- **总参数量**: {self.model_info.get('total_parameters', 0):,}\n")
        md_content.append(f"- **可训练参数量**: {self.model_info.get('trainable_parameters', 0):,}\n")
        md_content.append("\n")

        md_content.append("### 2.2 模型结构配置\n\n")
        md_content.append("```python\n")
        md_content.append(f"blocks = {self.model_info.get('blocks', [])}\n")
        md_content.append("```\n\n")

        md_content.append("### 2.3 完整模型结构\n\n")
        md_content.append("```\n")
        model_structure = self.model_info.get('model_structure', 'N/A')
        # 限制模型结构输出长度，避免报告过长
        if len(model_structure) > 5000:
            model_structure = model_structure[:5000] + "\n... (模型结构过长，已截断)"
        md_content.append(model_structure)
        md_content.append("\n```\n\n")

        # 3. 超参数配置
        md_content.append("## 3. 超参数配置\n\n")
        md_content.append("### 3.1 训练参数\n\n")
        train_params = ['epochs', 'batch_size', 'lr', 'opt', 'weight_decay_rate',
                       'step_size', 'gamma', 'patience', 'save_interval']
        for param in train_params:
            if param in self.hyperparameters:
                value = self.hyperparameters[param]
                md_content.append(f"- **{param}**: {value}\n")
        md_content.append("\n")

        md_content.append("### 3.2 模型结构参数\n\n")
        model_params = ['n_vertex', 'n_steam', 'n_air', 'n_his', 'n_pred',
                       'steam_dim', 'electricity_dim', 'embed_dim', 'mid_dim',
                       'Kt', 'stblock_num', 'act_func', 'droprate', 'weight_type']
        for param in model_params:
            if param in self.hyperparameters:
                value = self.hyperparameters[param]
                md_content.append(f"- **{param}**: {value}\n")
        md_content.append("\n")

        md_content.append("### 3.3 完整参数列表\n\n")
        md_content.append("<details>\n<summary>点击展开完整参数列表</summary>\n\n")
        md_content.append("```json\n")
        md_content.append(json.dumps(self.hyperparameters, indent=2, ensure_ascii=False))
        md_content.append("\n```\n\n")
        md_content.append("</details>\n\n")

        # 4. 训练过程
        md_content.append("## 4. 训练过程\n\n")
        md_content.append("### 4.1 训练历史\n\n")
        md_content.append("| Epoch | 训练损失 | 验证损失 | 学习率 | GPU内存(MB) |\n")
        md_content.append("|-------|----------|----------|--------|-------------|\n")
        for i, epoch in enumerate(self.epochs_completed):
            md_content.append(f"| {epoch} | {self.train_losses[i]:.6f} | "
                            f"{self.val_losses[i]:.6f} | {self.learning_rates[i]:.10f} | "
                            f"{self.gpu_memories[i]:.2f} |\n")
        md_content.append("\n")

        md_content.append("### 4.2 损失曲线\n\n")
        if loss_curve_path:
            rel_path = os.path.relpath(loss_curve_path, self.run_dir)
            md_content.append(f"![损失曲线]({rel_path})\n\n")

        md_content.append("### 4.3 学习率变化\n\n")
        if lr_curve_path:
            rel_path = os.path.relpath(lr_curve_path, self.run_dir)
            md_content.append(f"![学习率曲线]({rel_path})\n\n")

        md_content.append("### 4.4 GPU内存使用\n\n")
        if gpu_mem_path:
            rel_path = os.path.relpath(gpu_mem_path, self.run_dir)
            md_content.append(f"![GPU内存使用]({rel_path})\n\n")

        # 5. 验证过程
        md_content.append("## 5. 验证过程\n\n")
        md_content.append("验证损失在训练过程中持续监控，用于早停机制和模型选择。\n\n")
        md_content.append(f"- **最佳验证损失**: {min(self.val_losses):.6f} (Epoch {self.epochs_completed[np.argmin(self.val_losses)]})\n")
        md_content.append(f"- **最终验证损失**: {self.val_losses[-1]:.6f}\n")
        md_content.append("\n")

        # 6. 测试过程
        md_content.append("## 6. 测试过程\n\n")
        if self.test_results:
            md_content.append("### 6.1 测试损失\n\n")
            md_content.append(f"- **测试MSE**: {self.test_results.get('test_mse', 'N/A'):.6f}\n\n")

            md_content.append("### 6.2 详细评估指标\n\n")
            md_content.append("#### 蒸汽系统\n\n")
            md_content.append("| 指标 | 流量G | 压力P |\n")
            md_content.append("|------|-------|-------|\n")
            md_content.append(f"| MAE | {self.test_results.get('steam_MAE_G', 'N/A'):.6f} | "
                            f"{self.test_results.get('steam_MAE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| RMSE | {self.test_results.get('steam_RMSE_G', 'N/A'):.6f} | "
                            f"{self.test_results.get('steam_RMSE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| MAPE | {self.test_results.get('steam_MAPE_G', 'N/A'):.6f} | "
                            f"{self.test_results.get('steam_MAPE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| WMAPE | {self.test_results.get('steam_WMAPE_G', 'N/A'):.8f} | "
                            f"{self.test_results.get('steam_WMAPE_P', 'N/A'):.8f} |\n")
            md_content.append("\n")

            md_content.append("#### 空气系统\n\n")
            md_content.append("| 指标 | 流量G | 压力P |\n")
            md_content.append("|------|-------|-------|\n")
            md_content.append(f"| MAE | {self.test_results.get('air_MAE_G', 'N/A'):.6f} | "
                            f"{self.test_results.get('air_MAE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| RMSE | {self.test_results.get('air_RMSE_G', 'N/A'):.6f} | "
                            f"{self.test_results.get('air_RMSE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| MAPE | {self.test_results.get('air_MAPE_G', 'N/A'):.6f} | "
                            f"{self.test_results.get('air_MAPE_P', 'N/A'):.6f} |\n")
            md_content.append(f"| WMAPE | {self.test_results.get('air_WMAPE_G', 'N/A'):.8f} | "
                            f"{self.test_results.get('air_WMAPE_P', 'N/A'):.8f} |\n")
            md_content.append("\n")
        else:
            md_content.append("测试结果未记录。\n\n")

        # 7. 保存的模型权重
        md_content.append("## 7. 保存的模型权重\n\n")
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

        # 8. 模型最终结果总结
        md_content.append("## 8. 模型最终结果总结\n\n")
        md_content.append("### 8.1 训练效果\n\n")
        md_content.append(f"- 训练损失从 {self.train_losses[0]:.6f} 降至 {self.train_losses[-1]:.6f} "
                        f"(降低 {((self.train_losses[0] - self.train_losses[-1]) / self.train_losses[0] * 100):.2f}%)\n")
        md_content.append(f"- 验证损失从 {self.val_losses[0]:.6f} 降至 {self.val_losses[-1]:.6f} "
                        f"(降低 {((self.val_losses[0] - self.val_losses[-1]) / self.val_losses[0] * 100):.2f}%)\n")
        md_content.append("\n")

        if self.test_results:
            md_content.append("### 8.2 测试性能\n\n")
            md_content.append("模型在测试集上的表现：\n\n")
            md_content.append("- **蒸汽流量预测**: ")
            md_content.append(f"MAE={self.test_results.get('steam_MAE_G', 'N/A'):.6f}, ")
            md_content.append(f"RMSE={self.test_results.get('steam_RMSE_G', 'N/A'):.6f}, ")
            md_content.append(f"MAPE={self.test_results.get('steam_MAPE_G', 'N/A'):.6f}, ")
            md_content.append(f"WMAPE={self.test_results.get('steam_WMAPE_G', 'N/A'):.8f}\n\n")

            md_content.append("- **蒸汽压力预测**: ")
            md_content.append(f"MAE={self.test_results.get('steam_MAE_P', 'N/A'):.6f}, ")
            md_content.append(f"RMSE={self.test_results.get('steam_RMSE_P', 'N/A'):.6f}, ")
            md_content.append(f"MAPE={self.test_results.get('steam_MAPE_P', 'N/A'):.6f}, ")
            md_content.append(f"WMAPE={self.test_results.get('steam_WMAPE_P', 'N/A'):.8f}\n\n")

            md_content.append("- **空气流量预测**: ")
            md_content.append(f"MAE={self.test_results.get('air_MAE_G', 'N/A'):.6f}, ")
            md_content.append(f"RMSE={self.test_results.get('air_RMSE_G', 'N/A'):.6f}, ")
            md_content.append(f"MAPE={self.test_results.get('air_MAPE_G', 'N/A'):.6f}, ")
            md_content.append(f"WMAPE={self.test_results.get('air_WMAPE_G', 'N/A'):.8f}\n\n")

            md_content.append("- **空气压力预测**: ")
            md_content.append(f"MAE={self.test_results.get('air_MAE_P', 'N/A'):.6f}, ")
            md_content.append(f"RMSE={self.test_results.get('air_RMSE_P', 'N/A'):.6f}, ")
            md_content.append(f"MAPE={self.test_results.get('air_MAPE_P', 'N/A'):.6f}, ")
            md_content.append(f"WMAPE={self.test_results.get('air_WMAPE_P', 'N/A'):.8f}\n\n")

        # 9. 可视化图片
        md_content.append("## 9. 可视化图片\n\n")
        md_content.append("所有生成的可视化图片已保存在 `images/` 目录下。\n\n")
        if loss_curve_path:
            md_content.append(f"![损失曲线]({os.path.relpath(loss_curve_path, self.run_dir)})\n\n")
        if lr_curve_path:
            md_content.append(f"![学习率曲线]({os.path.relpath(lr_curve_path, self.run_dir)})\n\n")
        if gpu_mem_path:
            md_content.append(f"![GPU内存使用]({os.path.relpath(gpu_mem_path, self.run_dir)})\n\n")

        # 保存报告
        report_path = os.path.join(self.run_dir, "report.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(''.join(md_content))

        print(f"报告已生成: {report_path}")
        return report_path

