import requests
import json

def test_algorithm_analysis():
    # 登录获取token
    login_url = 'http://localhost:5000/api/auth/login'
    login_data = {
        'email': 'test_user',
        'password': 'test123'
    }
    
    print("正在登录...")
    login_response = requests.post(login_url, json=login_data)
    print(f"登录响应状态码: {login_response.status_code}")
    print(f"登录响应内容: {login_response.text}")
    
    if login_response.status_code == 200:
        token = login_response.json().get('token')
        headers = {'Authorization': f'Bearer {token}'}
        
        # 测试糖尿病预测模型训练
        print("\n测试糖尿病预测模型训练...")
        diabetes_url = 'http://localhost:5000/api/algorithm/train/diabetes'
        diabetes_response = requests.post(diabetes_url, headers=headers)
        print(f"糖尿病模型训练响应: {diabetes_response.text}")
        
        # 测试高血压预测模型训练
        print("\n测试高血压预测模型训练...")
        hypertension_url = 'http://localhost:5000/api/algorithm/train/hypertension'
        hypertension_response = requests.post(hypertension_url, headers=headers)
        print(f"高血压模型训练响应: {hypertension_response.text}")
        
        # 测试疾病风险预测
        print("\n测试疾病风险预测...")
        risk_url = 'http://localhost:5000/api/algorithm/predict/risk'
        risk_data = {
            'heart_rate': 75,
            'blood_pressure': '120/80',
            'blood_sugar': 5.5,
            'weight': 65,
            'sleep_hours': 7,
            'mood_score': 8
        }
        risk_response = requests.post(risk_url, headers=headers, json=risk_data)
        print(f"疾病风险预测响应: {risk_response.text}")
        
        # 测试健康状态评估
        print("\n测试健康状态评估...")
        health_url = 'http://localhost:5000/api/algorithm/assess/health'
        health_data = {
            'heart_rate': 75,
            'blood_pressure': '120/80',
            'blood_sugar': 5.5,
            'weight': 65,
            'sleep_hours': 7,
            'mood_score': 8
        }
        health_response = requests.post(health_url, headers=headers, json=health_data)
        print(f"健康状态评估响应: {health_response.text}")

if __name__ == '__main__':
    test_algorithm_analysis() 