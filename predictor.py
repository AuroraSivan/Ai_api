# predictor.py

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# 加载 ClinicalBERT
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)


# 将文本编码为向量
def encode_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    # 取 [CLS] token 的嵌入向量作为文本特征
    cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.numpy()


# 处理病情描述，并结合年龄、性别特征
def prepare_features(text, age, gender):
    # 先提取文本特征
    text_features = encode_text(text)

    # 将年龄和性别转化为数值
    age_tensor = torch.tensor([[float(age)]])
    gender_tensor = torch.tensor([[1.0 if gender.lower() == "female" else 0.0]])

    # 将文本特征与年龄、性别特征拼接
    features = np.concatenate([text_features.flatten(), age_tensor.numpy().flatten(), gender_tensor.numpy().flatten()])
    return features
