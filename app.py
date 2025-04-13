# app.py

from flask import Flask, request, jsonify
import joblib
from predictor import prepare_features

app = Flask(__name__)

# 加载训练好的模型
model = joblib.load('hospital_stay_predictor_model.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    description = data.get("病情描述", "")
    gender = data.get("性别", "male")
    age = data.get("年龄", 30)

    try:
        # 提取特征并预测住院天数
        features = prepare_features(description, age, gender)
        predicted_days = model.predict([features])[0]

        return jsonify({"predicted_hospital_days": round(predicted_days, 2)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
