import requests
import json

def test_recommendation():
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
            user_id = login_data['user']['id']
            headers = {
                "Authorization": f"Bearer {token}"
            }
            
            # 添加健康记录
            health_data = {
                "heart_rate": 75,
                "blood_pressure": "120/80",
                "blood_sugar": 5.0,
                "weight": 65.0,
                "sleep_hours": 7.5,
                "mood_score": 8
            }
            add_record_url = "http://localhost:5000/api/health/records"
            print("\n正在添加健康记录...")
            add_record_response = requests.post(add_record_url, headers=headers, json=health_data)
            print("添加健康记录响应状态码:", add_record_response.status_code)
            print("添加健康记录响应内容:", add_record_response.text)
            
            # 测试健康分析API
            analysis_url = f"http://localhost:5000/api/recommendation/analysis/{user_id}"
            print("\n正在发送健康分析请求...")
            analysis_response = requests.get(analysis_url, headers=headers)
            print("健康分析响应状态码:", analysis_response.status_code)
            print("健康分析响应内容:", analysis_response.text)
            
            # 测试健康推荐API
            recommendation_url = f"http://localhost:5000/api/recommendation/recommendation/{user_id}"
            print("\n正在发送健康推荐请求...")
            recommendation_response = requests.get(recommendation_url, headers=headers)
            print("健康推荐响应状态码:", recommendation_response.status_code)
            print("健康推荐响应内容:", recommendation_response.text)
        else:
            print("登录响应中没有token")
    else:
        print("登录失败")

if __name__ == "__main__":
    test_recommendation() 