import xgboost as xgb
import numpy as np

print("XGBoost version:", xgb.__version__)

# 构造一点假数据
X = np.random.rand(10000, 50)
y = np.random.randint(0, 2, size=10000)

# 转成 DMatrix
dtrain = xgb.DMatrix(X, label=y)

# 明确指定 GPU
params = {
    "objective": "binary:logistic",
    "tree_method": "hist",   # 关键：GPU
    "device": "cuda",            # 新版本推荐
    "max_depth": 6,
    "eval_metric": "logloss",
}

print("Start training with GPU...")
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=10
)

print("Training finished successfully.")
