from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 加载 ClinicalBERT
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


# 将文本编码为向量
def encode_text(text):
    print(f"\n[INFO] 原始文本: {text}")
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    print(f"[DEBUG] Tokenizer 输出 input_ids 形状: {inputs['input_ids'].shape}")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    print(f"[DEBUG] CLS 向量 shape: {cls_embedding.shape}")
    return cls_embedding.numpy()


# 处理病情描述，并结合年龄、性别特征
def prepare_features(text, age, gender):
    # 提取文本特征
    text_features = encode_text(text)

    # 将年龄和性别转化为数值
    age_tensor = torch.tensor([[float(age)]])
    gender_tensor = torch.tensor([[1.0 if gender.lower() == "female" else 0.0]])

    print(f"[INFO] 年龄: {age} -> Tensor: {age_tensor.numpy().flatten()}")
    print(f"[INFO] 性别: {gender} -> Tensor: {gender_tensor.numpy().flatten()}")

    # 拼接特征
    features = np.concatenate([
        text_features.flatten(),
        age_tensor.numpy().flatten(),
        gender_tensor.numpy().flatten()
    ])

    print(f"[RESULT] 最终特征向量维度: {features.shape}")
    return features
