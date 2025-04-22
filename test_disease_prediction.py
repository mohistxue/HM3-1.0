import requests
import json
import random
from datetime import datetime, timedelta

def generate_test_data(num_records=20):
    """生成测试用的健康记录数据"""
    records = []
    base_date = datetime.now()
    
    for i in range(num_records):
        record_date = base_date - timedelta(days=i)
        record = {
            "heart_rate": random.randint(60, 100),
            "blood_pressure": f"{random.randint(90, 140)}/{random.randint(60, 90)}",
            "blood_sugar": round(random.uniform(3.9, 6.1), 1),
            "weight": round(random.uniform(50, 80), 1),
            "height": round(random.uniform(160, 180), 1),
            "sleep_hours": round(random.uniform(6, 9), 1),
            "mood_score": random.randint(1, 10),
            "record_date": record_date.strftime("%Y-%m-%d")
        }
        records.append(record)
    return records

def test_disease_prediction():
    # 1. 登录获取token
    login_url = "http://localhost:5000/api/auth/login"
    login_data = {
        "username": "test_user",
        "password": "test123"
    }
    
    response = requests.post(login_url, json=login_data)
    print(f"登录状态码: {response.status_code}")
    print(f"登录响应: {response.json()}")
    
    if response.status_code == 200:
        token = response.json().get('token')
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        # 2. 生成并添加测试数据
        test_records = generate_test_data()
        for record in test_records:
            add_record_url = "http://localhost:5000/api/health/records"
            response = requests.post(add_record_url, headers=headers, json=record)
            print(f"添加健康记录状态码: {response.status_code}")
        
        # 3. 训练模型
        train_url = "http://localhost:5000/api/disease/train"
        response = requests.post(train_url, headers=headers)
        print(f"训练模型状态码: {response.status_code}")
        print(f"训练响应: {response.json()}")
        
        # 4. 预测疾病风险
        if response.status_code == 200:
            predict_url = "http://localhost:5000/api/disease/predict"
            test_data = {
                "heart_rate": 75,
                "blood_pressure": "120/80",
                "blood_sugar": 5.0,
                "weight": 65,
                "height": 170,
                "sleep_hours": 7,
                "mood_score": 8
            }
            response = requests.post(predict_url, headers=headers, json=test_data)
            print(f"预测状态码: {response.status_code}")
            print(f"预测响应: {response.json()}")

if __name__ == "__main__":
    test_disease_prediction()