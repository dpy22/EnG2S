"""
早停机制模块

实现早停（Early Stopping）功能，用于防止模型过拟合。
当验证损失在指定轮数内不再改善时，停止训练。
"""

import torch

class EarlyStopping(object):
    """
    早停机制类
    
    监控验证指标，当指标在指定轮数（patience）内不再改善时，触发早停。
    支持最小化模式（如验证损失）和最大化模式（如验证准确率）。
    """
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        """
        初始化早停机制
        
        参数:
            mode: 'min'表示最小化指标（如损失），'max'表示最大化指标（如准确率）
            min_delta: 最小改善量，只有当改善超过此值时才认为是有效改善
            patience: 耐心值，连续多少轮无改善后触发早停
            percentage: 是否使用百分比模式（min_delta作为百分比）
        """
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None  # 最佳指标值
        self.num_bad_epochs = 0  # 连续无改善的轮数
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            # 如果patience为0，禁用早停
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        """
        执行一步早停检查
        
        参数:
            metrics: 当前轮次的指标值（如验证损失）
        
        返回:
            True: 应该停止训练
            False: 继续训练
        """
        if self.best is None:
            # 第一次调用，记录当前值作为最佳值
            self.best = metrics
            return False

        if torch.isnan(metrics):
            # 如果指标为NaN，立即停止
            return True

        if self.is_better(metrics, self.best):
            # 指标改善，重置计数器并更新最佳值
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            # 指标未改善，增加计数器
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            # 达到耐心值，触发早停
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)