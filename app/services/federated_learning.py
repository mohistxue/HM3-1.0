# 联邦学习服务
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

class FederatedLearning:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.model_path = 'app/models/federated_model.pkl'
        self.scaler_path = 'app/models/federated_scaler.pkl'
        
        # 加载或初始化模型
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
    
    def prepare_data(self, health_records):
        """准备训练数据"""
        if not health_records:
            return None, None
            
        # 提取特征
        X = []
        y = []
        
        for record in health_records:
            features = []
            # 心率
            if record.get('heart_rate'):
                features.append(record['heart_rate'])
            else:
                features.append(0)
                
            # 血压
            if record.get('blood_pressure'):
                systolic, diastolic = map(float, record['blood_pressure'].split('/'))
                features.extend([systolic, diastolic])
            else:
                features.extend([0, 0])
                
            # 血糖
            if record.get('blood_sugar'):
                features.append(record['blood_sugar'])
            else:
                features.append(0)
                
            # 体重
            if record.get('weight'):
                features.append(record['weight'])
            else:
                features.append(0)
                
            # 睡眠时长
            if record.get('sleep_hours'):
                features.append(record['sleep_hours'])
            else:
                features.append(0)
                
            # 情绪评分
            if record.get('mood_score'):
                features.append(record['mood_score'])
            else:
                features.append(0)
                
            X.append(features)
            
            # 标签：根据健康指标综合评分
            health_score = self._calculate_health_score(record)
            y.append(1 if health_score >= 0.7 else 0)  # 1表示健康，0表示需要关注
            
        return np.array(X), np.array(y)
    
    def _calculate_health_score(self, record):
        """计算健康评分"""
        score = 0
        count = 0
        
        # 心率评分
        if record.get('heart_rate'):
            heart_rate = record['heart_rate']
            if 60 <= heart_rate <= 100:
                score += 1
            count += 1
            
        # 血压评分
        if record.get('blood_pressure'):
            systolic, diastolic = map(float, record['blood_pressure'].split('/'))
            if 90 <= systolic <= 140 and 60 <= diastolic <= 90:
                score += 1
            count += 1
            
        # 血糖评分
        if record.get('blood_sugar'):
            blood_sugar = record['blood_sugar']
            if 3.9 <= blood_sugar <= 6.1:
                score += 1
            count += 1
            
        # 睡眠评分
        if record.get('sleep_hours'):
            sleep_hours = record['sleep_hours']
            if 7 <= sleep_hours <= 9:
                score += 1
            count += 1
            
        # 情绪评分
        if record.get('mood_score'):
            mood_score = record['mood_score']
            if mood_score >= 6:
                score += 1
            count += 1
            
        return score / count if count > 0 else 0
    
    def train_local_model(self, health_records):
        """训练本地模型"""
        X, y = self.prepare_data(health_records)
        if X is None or len(X) == 0:
            return None
            
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 训练模型
        self.model.fit(X_scaled, y)
        
        # 保存模型
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        
        return {
            'model_weights': self.model.coef_.tolist(),
            'intercept': self.model.intercept_.tolist(),
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist()
        }
    
    def update_global_model(self, global_weights, global_intercept, global_scaler_mean, global_scaler_scale):
        """更新全局模型参数"""
        self.model.coef_ = np.array(global_weights)
        self.model.intercept_ = np.array(global_intercept)
        self.scaler.mean_ = np.array(global_scaler_mean)
        self.scaler.scale_ = np.array(global_scaler_scale)
        
        # 保存更新后的模型
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
    
    def predict_health_status(self, health_record):
        """预测健康状态"""
        X, _ = self.prepare_data([health_record])
        if X is None:
            return None
            
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0][1]
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'health_score': self._calculate_health_score(health_record)
        } 