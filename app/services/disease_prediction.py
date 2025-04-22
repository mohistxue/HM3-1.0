import xgboost as xgb
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
from datetime import datetime

class DiseasePrediction:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            scale_pos_weight=1.0,  # 处理类别不平衡
            base_score=0.5  # 设置初始预测值
        )
        self.scaler = StandardScaler()
        self.model_path = 'app/models/disease_model.pkl'
        self.scaler_path = 'app/models/disease_scaler.pkl'
        
        # 确保模型目录存在
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
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
            # 基础生理指标
            if record.get('heart_rate'):
                features.append(record['heart_rate'])
            else:
                features.append(0)
                
            if record.get('blood_pressure'):
                systolic, diastolic = map(float, record['blood_pressure'].split('/'))
                features.extend([systolic, diastolic])
            else:
                features.extend([0, 0])
                
            if record.get('blood_sugar'):
                features.append(record['blood_sugar'])
            else:
                features.append(0)
                
            # 生活习惯指标
            if record.get('weight'):
                features.append(record['weight'])
            else:
                features.append(0)
                
            if record.get('sleep_hours'):
                features.append(record['sleep_hours'])
            else:
                features.append(0)
                
            if record.get('mood_score'):
                features.append(record['mood_score'])
            else:
                features.append(0)
                
            # 添加BMI特征
            if record.get('weight') and record.get('height'):
                bmi = record['weight'] / ((record['height'] / 100) ** 2)
                features.append(bmi)
            else:
                features.append(0)
                
            X.append(features)
            
            # 标签：根据健康指标综合评分
            health_score = self._calculate_health_score(record)
            y.append(0 if health_score >= 0.7 else 1)  # 1表示需要关注，0表示健康
            
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
            
        # BMI评分
        if record.get('weight') and record.get('height'):
            bmi = record['weight'] / ((record['height'] / 100) ** 2)
            if 18.5 <= bmi <= 24:
                score += 1
            count += 1
            
        # 睡眠评分
        if record.get('sleep_hours'):
            sleep_hours = record['sleep_hours']
            if 7 <= sleep_hours <= 9:
                score += 1
            count += 1
            
        return score / count if count > 0 else 0
    
    def train_model(self, health_records):
        """训练疾病风险预测模型"""
        try:
            X, y = self.prepare_data(health_records)
            if X is None or len(X) < 10:  # 确保有足够的训练数据
                return False, "训练数据不足"
                
            # 数据标准化
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X_scaled, y)
            
            # 保存模型和标准化器
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            return True, "模型训练成功"
        except Exception as e:
            return False, f"模型训练失败: {str(e)}"
    
    def predict_risk(self, health_record):
        """预测疾病风险"""
        try:
            # 准备特征
            features = []
            if health_record.get('heart_rate'):
                features.append(health_record['heart_rate'])
            else:
                features.append(0)
                
            if health_record.get('blood_pressure'):
                systolic, diastolic = map(float, health_record['blood_pressure'].split('/'))
                features.extend([systolic, diastolic])
            else:
                features.extend([0, 0])
                
            if health_record.get('blood_sugar'):
                features.append(health_record['blood_sugar'])
            else:
                features.append(0)
                
            if health_record.get('weight'):
                features.append(health_record['weight'])
            else:
                features.append(0)
                
            if health_record.get('sleep_hours'):
                features.append(health_record['sleep_hours'])
            else:
                features.append(0)
                
            if health_record.get('mood_score'):
                features.append(health_record['mood_score'])
            else:
                features.append(0)
                
            # 计算BMI
            if health_record.get('weight') and health_record.get('height'):
                bmi = health_record['weight'] / ((health_record['height'] / 100) ** 2)
                features.append(bmi)
            else:
                features.append(0)
                
            # 标准化特征
            X = np.array([features])
            X_scaled = self.scaler.transform(X)
            
            # 预测风险概率
            risk_prob = self.model.predict_proba(X_scaled)[0][1]
            
            # 根据概率确定风险等级
            if risk_prob < 0.3:
                risk_level = "低风险"
            elif risk_prob < 0.7:
                risk_level = "中风险"
            else:
                risk_level = "高风险"
                
            return {
                "risk_probability": float(risk_prob),
                "risk_level": risk_level,
                "suggestions": self._generate_suggestions(health_record, risk_prob)
            }
        except Exception as e:
            return {
                "error": f"预测失败: {str(e)}"
            }
    
    def _generate_suggestions(self, health_record, risk_prob):
        """生成健康建议"""
        suggestions = []
        
        # 根据风险概率添加通用建议
        if risk_prob > 0.5:
            suggestions.append("建议定期进行健康检查")
            suggestions.append("保持规律作息，避免熬夜")
            suggestions.append("注意饮食均衡，控制糖分摄入")
        
        # 根据具体指标添加针对性建议
        if health_record.get('blood_pressure'):
            systolic, diastolic = map(float, health_record['blood_pressure'].split('/'))
            if systolic > 140 or diastolic > 90:
                suggestions.append("血压偏高，建议减少盐分摄入，适当运动")
                
        if health_record.get('blood_sugar'):
            if health_record['blood_sugar'] > 6.1:
                suggestions.append("血糖偏高，建议控制饮食，减少糖分摄入")
                
        if health_record.get('sleep_hours'):
            if health_record['sleep_hours'] < 7:
                suggestions.append("睡眠不足，建议保证每天7-8小时睡眠")
                
        if health_record.get('mood_score'):
            if health_record['mood_score'] < 5:
                suggestions.append("情绪评分较低，建议适当放松，保持积极心态")
                
        return suggestions 