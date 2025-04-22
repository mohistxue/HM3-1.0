# 联邦学习API
from flask import Blueprint, request, jsonify
from app.services.federated_learning import FederatedLearning
from app.models.user import HealthRecord
from app import db
from app.api.auth import token_required

bp = Blueprint('federated_learning', __name__)
fl_service = FederatedLearning()

@bp.route('/api/fl/train', methods=['POST'])
@token_required
def train_local_model(current_user):
    """训练本地模型"""
    health_records = HealthRecord.query.filter_by(user_id=current_user.id).all()
    records_data = [record.to_dict() for record in health_records]
    
    local_model_params = fl_service.train_local_model(records_data)
    if local_model_params is None:
        return jsonify({'error': '没有足够的训练数据'}), 400
        
    return jsonify(local_model_params)

@bp.route('/api/fl/update', methods=['POST'])
@token_required
def update_global_model(current_user):
    """更新全局模型参数"""
    data = request.get_json()
    
    if not all(key in data for key in ['model_weights', 'intercept', 'scaler_mean', 'scaler_scale']):
        return jsonify({'error': '缺少必要的模型参数'}), 400
        
    fl_service.update_global_model(
        data['model_weights'],
        data['intercept'],
        data['scaler_mean'],
        data['scaler_scale']
    )
    
    return jsonify({'message': '全局模型更新成功'})

@bp.route('/api/fl/predict', methods=['POST'])
@token_required
def predict_health_status(current_user):
    """预测健康状态"""
    data = request.get_json()
    
    if not data:
        return jsonify({'error': '请提供健康数据'}), 400
        
    prediction = fl_service.predict_health_status(data)
    if prediction is None:
        return jsonify({'error': '无法进行预测'}), 400
        
    return jsonify(prediction) 