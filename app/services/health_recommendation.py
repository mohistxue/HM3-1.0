# 健康推荐服务
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os

class HealthRecommendationService:
    def __init__(self):
        # 定义健康指标的正常范围
        self.normal_ranges = {
            'heart_rate': (60, 100),  # 每分钟心跳次数
            'blood_sugar': (3.9, 6.1),  # mmol/L
            'sleep_hours': (7, 9),  # 小时
            'mood_score': (7, 10),  # 1-10分
            'weight': (18.5, 24.9)  # BMI范围
        }
        
        # 模型相关路径
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        self.model_path = os.path.join(self.model_dir, 'health_model.joblib')
        self.scaler_path = os.path.join(self.model_dir, 'health_scaler.joblib')
        
        # 确保模型目录存在
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 初始化模型和标准化器
        self.model = LogisticRegression()
        self.scaler = StandardScaler()

    def train_model(self, training_data):
        """
        训练健康状况预测模型
        :param training_data: 包含用户健康记录的列表
        :return: 训练结果信息
        """
        try:
            # 准备训练数据
            X = []  # 特征
            y = []  # 标签（健康状态）
            
            for record in training_data:
                # 提取特征
                features = [
                    record['heart_rate'],
                    float(record['blood_pressure'].split('/')[0]),  # 收缩压
                    float(record['blood_pressure'].split('/')[1]),  # 舒张压
                    record['blood_sugar'],
                    record['sleep_hours'],
                    record['mood_score'],
                    record['weight']
                ]
                X.append(features)
                
                # 根据各项指标判断整体健康状态（0表示需要改善，1表示良好）
                health_status = self._evaluate_health_status(record)
                y.append(health_status)
            
            # 转换为numpy数组
            X = np.array(X)
            y = np.array(y)
            
            # 确保有足够的训练样本
            if len(X) < 10:
                return {
                    "status": "error",
                    "message": "需要至少10条健康记录来训练模型"
                }
            
            # 标准化特征
            X_scaled = self.scaler.fit_transform(X)
            
            # 训练模型
            self.model.fit(X_scaled, y)
            
            # 保存模型和标准化器
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            
            return {
                "status": "success",
                "message": "模型训练成功",
                "model_path": self.model_path,
                "training_samples": len(X)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"模型训练失败: {str(e)}"
            }
    
    def _evaluate_health_status(self, record):
        """
        评估健康记录的整体状态
        :param record: 健康记录字典
        :return: 0表示需要改善，1表示良好
        """
        # 检查各项指标是否在正常范围内
        heart_rate_normal = self.normal_ranges['heart_rate'][0] <= record['heart_rate'] <= self.normal_ranges['heart_rate'][1]
        
        systolic, diastolic = map(float, record['blood_pressure'].split('/'))
        blood_pressure_normal = 90 <= systolic <= 140 and 60 <= diastolic <= 90
        
        blood_sugar_normal = self.normal_ranges['blood_sugar'][0] <= record['blood_sugar'] <= self.normal_ranges['blood_sugar'][1]
        sleep_normal = self.normal_ranges['sleep_hours'][0] <= record['sleep_hours'] <= self.normal_ranges['sleep_hours'][1]
        mood_normal = record['mood_score'] >= self.normal_ranges['mood_score'][0]
        weight_normal = self.normal_ranges['weight'][0] <= record['weight'] <= self.normal_ranges['weight'][1]
        
        # 如果大多数指标正常，则认为整体状态良好
        normal_count = sum([heart_rate_normal, blood_pressure_normal, blood_sugar_normal, 
                          sleep_normal, mood_normal, weight_normal])
        return 1 if normal_count >= 4 else 0

    def analyze_health_metrics(self, health_record):
        """分析健康指标，返回每个指标的状态评估"""
        analysis = {}
        
        # 分析心率
        heart_rate = health_record.heart_rate
        analysis['heart_rate'] = {
            'value': heart_rate,
            'status': self._get_status(heart_rate, self.normal_ranges['heart_rate']),
            'description': self._get_heart_rate_description(heart_rate)
        }
        
        # 分析血压
        try:
            systolic, diastolic = map(int, health_record.blood_pressure.split('/'))
            blood_pressure_status = self._analyze_blood_pressure(systolic, diastolic)
            analysis['blood_pressure'] = {
                'value': health_record.blood_pressure,
                'status': blood_pressure_status,
                'description': self._get_blood_pressure_description(systolic, diastolic)
            }
        except:
            analysis['blood_pressure'] = {
                'value': health_record.blood_pressure,
                'status': 'unknown',
                'description': '血压数据格式错误'
            }
        
        # 分析血糖
        blood_sugar = health_record.blood_sugar
        analysis['blood_sugar'] = {
            'value': blood_sugar,
            'status': self._get_status(blood_sugar, self.normal_ranges['blood_sugar']),
            'description': self._get_blood_sugar_description(blood_sugar)
        }
        
        # 分析睡眠时间
        sleep_hours = health_record.sleep_hours
        analysis['sleep_hours'] = {
            'value': sleep_hours,
            'status': self._get_status(sleep_hours, self.normal_ranges['sleep_hours']),
            'description': self._get_sleep_description(sleep_hours)
        }
        
        # 分析心情评分
        mood_score = health_record.mood_score
        analysis['mood_score'] = {
            'value': mood_score,
            'status': self._get_status(mood_score, self.normal_ranges['mood_score']),
            'description': self._get_mood_description(mood_score)
        }
        
        # 分析体重（BMI）
        weight = health_record.weight
        analysis['weight'] = {
            'value': weight,
            'status': self._get_status(weight, self.normal_ranges['weight']),
            'description': self._get_weight_description(weight)
        }
        
        return analysis

    def generate_recommendations(self, analysis):
        """基于健康指标分析生成个性化建议"""
        recommendations = []
        
        # 根据各项指标状态生成建议
        for metric, data in analysis.items():
            if data['status'] == 'low':
                recommendations.extend(self._get_low_recommendations(metric))
            elif data['status'] == 'high':
                recommendations.extend(self._get_high_recommendations(metric))
        
        # 如果所有指标正常，添加保持建议
        if not recommendations:
            recommendations.append("您的各项健康指标都在正常范围内，请继续保持当前的健康生活方式！")
        
        return recommendations

    def _get_status(self, value, normal_range):
        """判断指标值的状态（低、正常、高）"""
        if value < normal_range[0]:
            return 'low'
        elif value > normal_range[1]:
            return 'high'
        return 'normal'

    def _analyze_blood_pressure(self, systolic, diastolic):
        """分析血压状态"""
        if systolic < 90 or diastolic < 60:
            return 'low'
        elif systolic > 140 or diastolic > 90:
            return 'high'
        return 'normal'

    # 各项指标的描述生成方法
    def _get_heart_rate_description(self, value):
        if value < self.normal_ranges['heart_rate'][0]:
            return "心率偏低，可能感觉疲劳或头晕"
        elif value > self.normal_ranges['heart_rate'][1]:
            return "心率偏高，可能感觉心跳加快或焦虑"
        return "心率正常，心脏功能良好"

    def _get_blood_pressure_description(self, systolic, diastolic):
        if systolic < 90 or diastolic < 60:
            return "血压偏低，可能感觉头晕或疲劳"
        elif systolic > 140 or diastolic > 90:
            return "血压偏高，需要注意控制"
        return "血压正常，循环系统功能良好"

    def _get_blood_sugar_description(self, value):
        if value < self.normal_ranges['blood_sugar'][0]:
            return "血糖偏低，可能感觉饥饿或头晕"
        elif value > self.normal_ranges['blood_sugar'][1]:
            return "血糖偏高，需要注意控制"
        return "血糖正常，代谢功能良好"

    def _get_sleep_description(self, value):
        if value < self.normal_ranges['sleep_hours'][0]:
            return "睡眠时间不足，可能影响日间表现"
        elif value > self.normal_ranges['sleep_hours'][1]:
            return "睡眠时间过长，可能影响身体状态"
        return "睡眠时间适中，有助于身体恢复"

    def _get_mood_description(self, value):
        if value < self.normal_ranges['mood_score'][0]:
            return "心情状态欠佳，需要适当调节"
        return "心情状态良好，请继续保持"

    def _get_weight_description(self, value):
        if value < self.normal_ranges['weight'][0]:
            return "体重偏低，需要适当增加营养摄入"
        elif value > self.normal_ranges['weight'][1]:
            return "体重偏高，需要注意控制"
        return "体重正常，身体状态良好"

    # 针对异常指标的建议生成方法
    def _get_low_recommendations(self, metric):
        recommendations = {
            'heart_rate': [
                "适当进行有氧运动，如散步、慢跑等",
                "保持充足的休息和睡眠",
                "如果经常感觉头晕或疲劳，建议咨询医生"
            ],
            'blood_pressure': [
                "适当增加盐分摄入",
                "保持充足的水分补充",
                "避免突然起立或剧烈运动"
            ],
            'blood_sugar': [
                "规律进食，避免长时间空腹",
                "随身携带含糖食物以应对低血糖",
                "注意营养均衡，适量增加碳水化合物摄入"
            ],
            'sleep_hours': [
                "保持规律的作息时间",
                "创造良好的睡眠环境",
                "避免睡前使用电子设备"
            ],
            'mood_score': [
                "尝试进行放松活动，如瑜伽或冥想",
                "与亲朋好友多交流",
                "适当参加户外活动，增加阳光接触"
            ],
            'weight': [
                "适当增加饮食量",
                "增加优质蛋白质的摄入",
                "进行适度的力量训练"
            ]
        }
        return recommendations.get(metric, ["请咨询专业医生获取更详细的建议"])

    def _get_high_recommendations(self, metric):
        recommendations = {
            'heart_rate': [
                "避免剧烈运动和情绪激动",
                "学习放松技巧，如深呼吸",
                "减少咖啡因的摄入"
            ],
            'blood_pressure': [
                "限制盐分摄入",
                "保持规律运动",
                "避免压力和情绪波动"
            ],
            'blood_sugar': [
                "控制碳水化合物的摄入",
                "增加运动量",
                "规律监测血糖水平"
            ],
            'sleep_hours': [
                "适当增加日间活动量",
                "避免日间过长的午睡",
                "保持规律的作息时间"
            ],
            'weight': [
                "控制饮食摄入量",
                "增加运动频率",
                "选择低热量、高营养的食物"
            ]
        }
        return recommendations.get(metric, ["请咨询专业医生获取更详细的建议"]) 