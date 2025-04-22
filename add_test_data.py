# 添加个性化推荐测试数据
from app import create_app, db
from app.models.user import User, HealthRecord
from datetime import datetime, timedelta
import random

def add_test_data():
    app = create_app()
    with app.app_context():
        # 检查测试用户是否存在
        test_user = User.query.filter_by(username='test_user').first()
        if not test_user:
            # 创建测试用户
            test_user = User(
                username='test_user',
                email='test@example.com',
                password='test123'
            )
            db.session.add(test_user)
            db.session.commit()
        
        # 删除旧的健康记录
        HealthRecord.query.filter_by(user_id=test_user.id).delete()
        
        # 添加测试健康记录
        for i in range(30):
            record = HealthRecord(
                user_id=test_user.id,
                heart_rate=random.randint(60, 100),
                blood_pressure=f"{random.randint(90, 140)}/{random.randint(60, 90)}",
                blood_sugar=random.uniform(3.9, 6.1),
                weight=random.uniform(50, 80),
                sleep_hours=random.uniform(6, 9),
                mood_score=random.randint(1, 10),
                recorded_at=datetime.now() - timedelta(days=i)
            )
            db.session.add(record)
        
        db.session.commit()
        print("测试数据添加成功！")

if __name__ == '__main__':
    add_test_data() 