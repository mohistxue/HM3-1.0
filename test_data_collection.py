import requests
import json
import pandas as pd
import numpy as np

def test_data_collection():
    # 登录获取token
    login_url = "http://localhost:5000/api/auth/login"
    login_data = {
        "username": "test_user",
        "password": "test123"
    }
    print("正在发送登录请求...")
    login_response = requests.post(login_url, json=login_data)
    print("登录响应状态码:", login_response.status_code)
    print("登录响应内容:", login_response.text)
    
    if login_response.status_code == 200:
        login_data = login_response.json()
        if 'token' in login_data:
            token = login_data['token']
            headers = {
                "Authorization": f"Bearer {token}"
            }
            
            # 1. 测试数据预处理
            print("\n1. 测试数据预处理")
            preprocess_url = "http://localhost:5000/api/data/preprocess"
            test_data = {
                "heart_rate": [80, 85, None, 90, 95],
                "blood_pressure": ["120/80", "130/85", "140/90", None, "150/95"],
                "blood_sugar": [5.5, 6.0, 6.5, 7.0, None],
                "weight": [65, 66, 67, 68, 69],
                "sleep_hours": [7, 7.5, 8, 8.5, 9],
                "mood_score": [8, 7, 6, 5, 4]
            }
            preprocess_response = requests.post(preprocess_url, json=test_data, headers=headers)
            print("预处理响应状态码:", preprocess_response.status_code)
            print("预处理响应内容:", preprocess_response.text)
            
            # 2. 测试情感分析
            print("\n2. 测试情感分析")
            sentiment_url = "http://localhost:5000/api/data/sentiment"
            sentiment_data = {
                "text": "今天感觉心情很好，睡眠质量也不错"
            }
            sentiment_response = requests.post(sentiment_url, json=sentiment_data, headers=headers)
            print("情感分析响应状态码:", sentiment_response.status_code)
            print("情感分析响应内容:", sentiment_response.text)
            
            # 3. 测试医院数据获取
            print("\n3. 测试医院数据获取")
            hospital_url = "http://localhost:5000/api/data/hospital"
            hospital_data = {
                "api_url": "http://example.com/api/hospital",
                "params": {
                    "patient_id": "12345",
                    "start_date": "2023-01-01",
                    "end_date": "2023-12-31"
                }
            }
            hospital_response = requests.post(hospital_url, json=hospital_data, headers=headers)
            print("医院数据响应状态码:", hospital_response.status_code)
            print("医院数据响应内容:", hospital_response.text)
            
            # 4. 测试可穿戴设备数据获取
            print("\n4. 测试可穿戴设备数据获取")
            wearable_url = "http://localhost:5000/api/data/wearable"
            wearable_data = {
                "device_type": "huawei_band"
            }
            wearable_response = requests.post(wearable_url, json=wearable_data, headers=headers)
            print("可穿戴设备数据响应状态码:", wearable_response.status_code)
            print("可穿戴设备数据响应内容:", wearable_response.text)
        else:
            print("登录响应中没有token")
    else:
        print("登录失败")

if __name__ == "__main__":
    test_data_collection() 