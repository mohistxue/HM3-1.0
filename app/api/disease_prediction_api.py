from flask import Blueprint, jsonify, request
from app.utils.auth import token_required
from app.services.disease_prediction import DiseasePrediction
from app.models.user import User, HealthRecord
from app import db

bp = Blueprint('disease_prediction', __name__)

@bp.route('/train', methods=['POST'])
@token_required
def train_model(current_user):
    """训练疾病风险预测模型"""
    # 获取用户的健康记录
    health_records = HealthRecord.query.filter_by(user_id=current_user.id).all()
    records_data = [record.to_dict() for record in health_records]
    
    # 训练模型
    predictor = DiseasePrediction()
    success, message = predictor.train_model(records_data)
    
    if success:
        return jsonify({"message": message}), 200
    else:
        return jsonify({"error": message}), 400

@bp.route('/predict', methods=['POST'])
@token_required
def predict_risk(current_user):
    """预测疾病风险"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "未提供健康数据"}), 400
        
    # 获取预测结果
    predictor = DiseasePrediction()
    result = predictor.predict_risk(data)
    
    return jsonify(result), 200

@bp.route('/batch_predict', methods=['POST'])
@token_required
def batch_predict(current_user):
    """批量预测疾病风险"""
    data = request.get_json()
    
    if not data or not isinstance(data, list):
        return jsonify({"error": "未提供有效的健康数据列表"}), 400
        
    # 获取预测结果
    predictor = DiseasePrediction()
    results = []
    
    for record in data:
        result = predictor.predict_risk(record)
        results.append(result)
    
    return jsonify({"predictions": results}), 200 