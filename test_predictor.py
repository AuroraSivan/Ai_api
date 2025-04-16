from predictor import prepare_features

# 测试样本
samples = [
    ("患者主诉头痛、发热", 45, "male"),
    ("患者主诉腹痛、腹泻", 32, "female"),
    ("患者出现呼吸困难和咳嗽", 60, "male"),
]

# 测试循环
for i, (text, age, gender) in enumerate(samples, 1):
    print(f"\n===== 测试样本 {i} =====")
    features = prepare_features(text, age, gender)
    print(f"[Test] 特征向量前10项: {features[:10]}")
