import requests
import json

def test_federated_learning():
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
            
            # 测试训练本地模型
            print("\n1. 测试训练本地模型")
            train_url = "http://localhost:5000/api/fl/train"
            train_response = requests.post(train_url, headers=headers)
            print("训练响应状态码:", train_response.status_code)
            print("训练响应内容:", train_response.text)
            
            if train_response.status_code == 200:
                local_model_params = train_response.json()
                
                # 测试更新全局模型
                print("\n2. 测试更新全局模型")
                update_url = "http://localhost:5000/api/fl/update"
                update_response = requests.post(update_url, json=local_model_params, headers=headers)
                print("更新响应状态码:", update_response.status_code)
                print("更新响应内容:", update_response.text)
                
                # 测试预测健康状态
                print("\n3. 测试预测健康状态")
                predict_url = "http://localhost:5000/api/fl/predict"
                predict_data = {
                    "heart_rate": 75,
                    "blood_pressure": "120/80",
                    "blood_sugar": 5.5,
                    "weight": 65,
                    "sleep_hours": 7.5,
                    "mood_score": 8
                }
                predict_response = requests.post(predict_url, json=predict_data, headers=headers)
                print("预测响应状态码:", predict_response.status_code)
                print("预测响应内容:", predict_response.text)
        else:
            print("登录响应中没有token")
    else:
        print("登录失败")

if __name__ == "__main__":
    test_federated_learning() 