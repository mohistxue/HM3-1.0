# 用户模型
from app import db
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

class User(db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 健康数据关联
    health_records = db.relationship('HealthRecord', backref='user', lazy=True)
    
    @property
    def password(self):
        raise AttributeError('密码不可读')
        
    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def set_password(self, password):
        """设置密码的辅助方法"""
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def __repr__(self):
        return f'<User {self.username}>'

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class HealthRecord(db.Model):
    __tablename__ = 'health_records'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    heart_rate = db.Column(db.Integer)
    blood_pressure = db.Column(db.String(20))
    blood_sugar = db.Column(db.Float)
    weight = db.Column(db.Float)
    sleep_hours = db.Column(db.Float)
    mood_score = db.Column(db.Integer)  # 1-10分
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<HealthRecord {self.id}>'
        
    # 将健康记录转换为字典
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'heart_rate': self.heart_rate,
            'blood_pressure': self.blood_pressure,
            'blood_sugar': self.blood_sugar,
            'weight': self.weight,
            'sleep_hours': self.sleep_hours,
            'mood_score': self.mood_score,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None
        } 