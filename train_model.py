# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import xgboost as xgb
from predictor import prepare_features

# 假设数据存储在 CSV 文件中
data = pd.read_csv("data/hospital_data.csv")

# 假设数据有以下列：病情描述, 性别, 年龄, 住院天数
texts = data["病情描述"].values
ages = data["年龄"].values
genders = data["性别"].values
hospital_days = data["住院天数"].values

# 提取特征
X = np.array([prepare_features(text, age, gender) for text, age, gender in zip(texts, ages, genders)])

# 训练数据和测试数据分割
X_train, X_test, y_train, y_test = train_test_split(X, hospital_days, test_size=0.2, random_state=42)

# 使用 XGBoost 训练回归模型（你也可以选择 RandomForestRegressor）
model = xgb.XGBRegressor(objective="reg:squarederror", n_estimators=100, max_depth=6, learning_rate=0.1)
model.fit(X_train, y_train)

# 评估模型
print(f"模型得分: {model.score(X_test, y_test)}")

# 保存训练好的模型
import joblib
joblib.dump(model, 'models/hospital_stay_model.pkl')
