# 数据收集服务
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from snownlp import SnowNLP
import jieba
import json
from datetime import datetime
import requests
from typing import Dict, List, Union, Optional
import logging

class DataCollectionService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.minmax_scaler = MinMaxScaler()
        self.isolation_forest = IsolationForest(contamination=0.1)
        
    def fetch_hospital_data(self, api_url: str, params: Dict) -> pd.DataFrame:
        """从医院HIS系统获取数据"""
        try:
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            data = response.json()
            return pd.DataFrame(data)
        except Exception as e:
            self.logger.error(f"获取医院数据失败: {str(e)}")
            return pd.DataFrame()

    def fetch_wearable_data(self, device_type: str, user_id: str) -> pd.DataFrame:
        """获取可穿戴设备数据"""
        # 这里需要根据具体设备类型实现不同的数据获取逻辑
        try:
            if device_type == "apple_watch":
                # 实现Apple Watch数据获取
                pass
            elif device_type == "huawei_band":
                # 实现华为手环数据获取
                pass
            return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"获取可穿戴设备数据失败: {str(e)}")
            return pd.DataFrame()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        try:
            # 1. 处理缺失值
            df = self._handle_missing_values(df)
            
            # 2. 检测异常值
            df = self._detect_outliers(df)
            
            # 3. 特征归一化
            df = self._normalize_features(df)
            
            return df
        except Exception as e:
            self.logger.error(f"数据预处理失败: {str(e)}")
            return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理缺失值"""
        # 数值型特征使用均值填充
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        
        # 分类特征使用众数填充
        categorical_cols = df.select_dtypes(include=['object']).columns
        df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
        
        return df

    def _detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测异常值"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # 使用箱线图方法
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 将异常值替换为边界值
            df[col] = df[col].clip(lower_bound, upper_bound)
            
            # 使用孤立森林方法
            outliers = self.isolation_forest.fit_predict(df[[col]])
            df.loc[outliers == -1, col] = df[col].mean()
        
        return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """特征归一化"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        # 使用Z-Score标准化
        df[numeric_cols] = self.scaler.fit_transform(df[numeric_cols])
        
        # 使用Min-Max归一化
        # df[numeric_cols] = self.minmax_scaler.fit_transform(df[numeric_cols])
        
        return df

    def analyze_sentiment(self, text: str) -> Dict:
        """情感分析"""
        try:
            # 使用SnowNLP进行情感分析
            s = SnowNLP(text)
            sentiment_score = s.sentiments
            
            # 根据情感分数判断情感
            if sentiment_score > 0.6:
                sentiment = "positive"
            elif sentiment_score < 0.4:
                sentiment = "negative"
            else:
                sentiment = "neutral"
            
            return {
                "sentiment": sentiment,
                "score": float(sentiment_score)
            }
        except Exception as e:
            self.logger.error(f"情感分析失败: {str(e)}")
            return {"sentiment": "unknown", "score": 0.0}

    def parse_health_record(self, record: Dict) -> Dict:
        """解析健康记录"""
        try:
            # 解析时间
            if isinstance(record.get("recorded_at"), str):
                record["recorded_at"] = datetime.fromisoformat(record["recorded_at"])
            
            # 解析血压
            if isinstance(record.get("blood_pressure"), str):
                systolic, diastolic = map(float, record["blood_pressure"].split("/"))
                record["systolic_bp"] = systolic
                record["diastolic_bp"] = diastolic
            
            # 情感分析
            if record.get("mood_description"):
                sentiment = self.analyze_sentiment(record["mood_description"])
                record["mood_sentiment"] = sentiment["sentiment"]
                record["mood_score"] = sentiment["score"]
            
            return record
        except Exception as e:
            self.logger.error(f"解析健康记录失败: {str(e)}")
            return record

    def assess_mental_health(self, records: List[Dict]) -> Dict:
        """评估心理健康状态"""
        try:
            # 计算情绪稳定性
            mood_scores = [r.get('mood_score', 0) for r in records]
            mood_stability = np.std(mood_scores)
            
            # 分析情绪趋势
            mood_trend = np.polyfit(range(len(mood_scores)), mood_scores, 1)[0]
            
            # 检测压力水平
            stress_level = self._calculate_stress_level(records)
            
            return {
                'mood_stability': float(mood_stability),
                'mood_trend': float(mood_trend),
                'stress_level': stress_level,
                'recommendations': self._generate_mental_health_recommendations(
                    mood_stability, mood_trend, stress_level
                )
            }
        except Exception as e:
            self.logger.error(f"心理健康评估失败: {str(e)}")
            return {}

    def _generate_mental_health_recommendations(self, 
                                             mood_stability: float,
                                             mood_trend: float,
                                             stress_level: str) -> List[str]:
        """生成心理健康建议"""
        recommendations = []
        
        # 基于情绪稳定性
        if mood_stability > 0.3:
            recommendations.append("建议进行正念冥想练习")
            recommendations.append("保持规律的作息时间")
        
        # 基于情绪趋势
        if mood_trend < -0.1:
            recommendations.append("建议与朋友或家人多交流")
            recommendations.append("考虑进行心理咨询")
        
        # 基于压力水平
        if stress_level == "high":
            recommendations.append("建议进行放松训练")
            recommendations.append("适当减少工作/学习压力")
        elif stress_level == "medium":
            recommendations.append("保持适度运动")
            recommendations.append("注意休息和放松")
        
        return recommendations 