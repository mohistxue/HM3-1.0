# 算法分析服务
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

class AlgorithmAnalysisService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.models_dir = 'app/models'
        self._ensure_models_dir()
        
        # 加载或初始化模型
        self.diabetes_model = self._load_model('diabetes_model.pkl')
        self.hypertension_model = self._load_model('hypertension_model.pkl')
        self.health_assessment_model = self._load_model('health_assessment_model.pkl')
        
    def _ensure_models_dir(self):
        """确保模型目录存在"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
            
    def _load_model(self, model_name: str):
        """加载模型"""
        model_path = os.path.join(self.models_dir, model_name)
        if os.path.exists(model_path):
            return joblib.load(model_path)
        return None
        
    def prepare_training_data(self, health_records: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """准备训练数据"""
        try:
            # 转换为DataFrame
            df = pd.DataFrame(health_records)
            
            # 提取特征
            features = []
            for record in health_records:
                feature = []
                # 心率
                feature.append(record.get('heart_rate', 0))
                # 血压
                if 'blood_pressure' in record:
                    systolic, diastolic = map(float, record['blood_pressure'].split('/'))
                    feature.extend([systolic, diastolic])
                else:
                    feature.extend([0, 0])
                # 血糖
                feature.append(record.get('blood_sugar', 0))
                # 体重
                feature.append(record.get('weight', 0))
                # 睡眠时长
                feature.append(record.get('sleep_hours', 0))
                # 情绪评分
                feature.append(record.get('mood_score', 0))
                features.append(feature)
                
            X = np.array(features)
            y = np.array([self._calculate_health_label(record) for record in health_records])
            
            return X, y
        except Exception as e:
            self.logger.error(f"准备训练数据失败: {str(e)}")
            return np.array([]), np.array([])
            
    def _calculate_health_label(self, record: Dict) -> int:
        """计算健康标签"""
        score = 0
        count = 0
        
        # 心率评分
        if 'heart_rate' in record:
            heart_rate = record['heart_rate']
            if 60 <= heart_rate <= 100:
                score += 1
            count += 1
            
        # 血压评分
        if 'blood_pressure' in record:
            systolic, diastolic = map(float, record['blood_pressure'].split('/'))
            if 90 <= systolic <= 140 and 60 <= diastolic <= 90:
                score += 1
            count += 1
            
        # 血糖评分
        if 'blood_sugar' in record:
            blood_sugar = record['blood_sugar']
            if 3.9 <= blood_sugar <= 6.1:
                score += 1
            count += 1
            
        return 1 if score / count >= 0.7 else 0
        
    def train_diabetes_model(self, health_records: List[Dict]) -> Dict:
        """训练糖尿病预测模型"""
        try:
            X, y = self.prepare_training_data(health_records)
            if len(X) == 0:
                return {'error': '没有足够的训练数据'}
                
            # 数据标准化
            X = self.scaler.fit_transform(X)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 训练模型
            model = GradientBoostingClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # 保存模型
            self.diabetes_model = model
            joblib.dump(model, os.path.join(self.models_dir, 'diabetes_model.pkl'))
            
            return {
                'message': '模型训练成功',
                'metrics': metrics
            }
        except Exception as e:
            self.logger.error(f"训练糖尿病模型失败: {str(e)}")
            return {'error': str(e)}
            
    def train_hypertension_model(self, health_records: List[Dict]) -> Dict:
        """训练高血压预测模型"""
        try:
            X, y = self.prepare_training_data(health_records)
            if len(X) == 0:
                return {'error': '没有足够的训练数据'}
                
            # 数据标准化
            X = self.scaler.fit_transform(X)
            
            # 划分训练集和测试集
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 训练模型
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            # 评估模型
            y_pred = model.predict(X_test)
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # 保存模型
            self.hypertension_model = model
            joblib.dump(model, os.path.join(self.models_dir, 'hypertension_model.pkl'))
            
            return {
                'message': '模型训练成功',
                'metrics': metrics
            }
        except Exception as e:
            self.logger.error(f"训练高血压模型失败: {str(e)}")
            return {'error': str(e)}
            
    def predict_disease_risk(self, health_record: Dict) -> Dict:
        """预测疾病风险"""
        try:
            if self.diabetes_model is None or self.hypertension_model is None:
                return {'error': '模型未训练'}
                
            # 准备特征
            features = []
            # 心率
            features.append(health_record.get('heart_rate', 0))
            # 血压
            if 'blood_pressure' in health_record:
                systolic, diastolic = map(float, health_record['blood_pressure'].split('/'))
                features.extend([systolic, diastolic])
            else:
                features.extend([0, 0])
            # 血糖
            features.append(health_record.get('blood_sugar', 0))
            # 体重
            features.append(health_record.get('weight', 0))
            # 睡眠时长
            features.append(health_record.get('sleep_hours', 0))
            # 情绪评分
            features.append(health_record.get('mood_score', 0))
            
            X = np.array([features])
            X = self.scaler.transform(X)
            
            # 预测
            diabetes_prob = self.diabetes_model.predict_proba(X)[0][1]
            hypertension_prob = self.hypertension_model.predict_proba(X)[0][1]
            
            return {
                'diabetes_risk': float(diabetes_prob),
                'hypertension_risk': float(hypertension_prob),
                'recommendations': self._generate_risk_recommendations(
                    diabetes_prob, hypertension_prob
                )
            }
        except Exception as e:
            self.logger.error(f"预测疾病风险失败: {str(e)}")
            return {'error': str(e)}
            
    def _generate_risk_recommendations(self, 
                                     diabetes_prob: float,
                                     hypertension_prob: float) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        # 糖尿病风险建议
        if diabetes_prob > 0.7:
            recommendations.append("建议进行血糖监测")
            recommendations.append("控制饮食，减少糖分摄入")
        elif diabetes_prob > 0.4:
            recommendations.append("注意饮食健康")
            recommendations.append("保持适度运动")
            
        # 高血压风险建议
        if hypertension_prob > 0.7:
            recommendations.append("建议定期测量血压")
            recommendations.append("减少盐分摄入")
        elif hypertension_prob > 0.4:
            recommendations.append("保持健康饮食")
            recommendations.append("适当运动")
            
        return recommendations
        
    def assess_health_status(self, health_records: List[Dict]) -> Dict:
        """评估健康状态"""
        try:
            # 计算健康评分
            health_scores = []
            for record in health_records:
                score = self._calculate_health_score(record)
                health_scores.append(score)
                
            # 计算趋势
            if len(health_scores) > 1:
                trend = np.polyfit(range(len(health_scores)), health_scores, 1)[0]
            else:
                trend = 0
                
            # 生成建议
            recommendations = self._generate_health_recommendations(
                np.mean(health_scores), trend
            )
            
            return {
                'average_score': float(np.mean(health_scores)),
                'trend': float(trend),
                'recommendations': recommendations
            }
        except Exception as e:
            self.logger.error(f"评估健康状态失败: {str(e)}")
            return {'error': str(e)}
            
    def _calculate_health_score(self, record: Dict) -> float:
        """计算健康评分"""
        score = 0
        count = 0
        
        # 心率评分
        if 'heart_rate' in record:
            heart_rate = record['heart_rate']
            if 60 <= heart_rate <= 100:
                score += 1
            count += 1
            
        # 血压评分
        if 'blood_pressure' in record:
            systolic, diastolic = map(float, record['blood_pressure'].split('/'))
            if 90 <= systolic <= 140 and 60 <= diastolic <= 90:
                score += 1
            count += 1
            
        # 血糖评分
        if 'blood_sugar' in record:
            blood_sugar = record['blood_sugar']
            if 3.9 <= blood_sugar <= 6.1:
                score += 1
            count += 1
            
        # 睡眠评分
        if 'sleep_hours' in record:
            sleep_hours = record['sleep_hours']
            if 7 <= sleep_hours <= 9:
                score += 1
            count += 1
            
        # 情绪评分
        if 'mood_score' in record:
            mood_score = record['mood_score']
            if mood_score >= 6:
                score += 1
            count += 1
            
        return score / count if count > 0 else 0
        
    def _generate_health_recommendations(self, 
                                       average_score: float,
                                       trend: float) -> List[str]:
        """生成健康建议"""
        recommendations = []
        
        # 基于平均分
        if average_score < 0.6:
            recommendations.append("建议进行全面体检")
            recommendations.append("调整生活方式")
        elif average_score < 0.8:
            recommendations.append("保持健康习惯")
            recommendations.append("注意休息")
            
        # 基于趋势
        if trend < -0.1:
            recommendations.append("健康状况有下降趋势，建议及时调整")
        elif trend > 0.1:
            recommendations.append("健康状况良好，继续保持")
            
        return recommendations 