# 测试用户API
import requests
import json

BASE_URL = 'http://localhost:5000'

def test_register():
    """测试用户注册"""
    url = f"{BASE_URL}/api/auth/register"
    data = {
        "username": "test_user",
        "password": "test123",
        "email": "test@example.com"
    }
    response = requests.post(url, json=data)
    print("注册响应:", response.json())
    return response.json()

def test_login():
    """测试用户登录"""
    url = f"{BASE_URL}/api/auth/login"
    data = {
        "username": "test_user",
        "password": "test123"
    }
    response = requests.post(url, json=data)
    print("登录响应:", response.json())
    return response.json()

def test_add_health_record(user_id):
    """测试添加健康记录"""
    url = f"{BASE_URL}/api/health/record"
    data = {
        "user_id": user_id,
        "heart_rate": 75,
        "blood_pressure": "120/80",
        "blood_sugar": 5.5,
        "weight": 65,
        "sleep_hours": 7.5,
        "mood_score": 8
    }
    response = requests.post(url, json=data)
    print("添加健康记录响应:", response.json())
    return response.json()

def test_get_health_analysis(user_id):
    """测试获取健康分析"""
    url = f"{BASE_URL}/api/health/analysis/{user_id}"
    response = requests.get(url)
    print("健康分析响应:", json.dumps(response.json(), ensure_ascii=False, indent=2))
    return response.json()

if __name__ == '__main__':
    print("=== 开始测试 API ===")
    
    # 测试注册
    print("\n1. 测试用户注册")
    register_result = test_register()
    
    # 测试登录
    print("\n2. 测试用户登录")
    login_result = test_login()
    
    if 'user' in login_result:
        user_id = login_result['user']['id']
        
        # 测试添加健康记录
        print("\n3. 测试添加健康记录")
        test_add_health_record(user_id)
        
        # 测试获取健康分析
        print("\n4. 测试获取健康分析")
        test_get_health_analysis(user_id)
    
    print("\n=== 测试完成 ===") 