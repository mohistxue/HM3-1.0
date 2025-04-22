from flask import Blueprint, request, jsonify
from app.services.algorithm_analysis import AlgorithmAnalysisService
from app.utils.auth import token_required

bp = Blueprint('algorithm_analysis', __name__)
algorithm_service = AlgorithmAnalysisService()

@bp.route('/api/algorithm/train/diabetes', methods=['POST'])
@token_required
def train_diabetes_model():
    """训练糖尿病预测模型"""
    try:
        data = request.get_json()
        if not data or 'health_records' not in data:
            return jsonify({'error': '缺少健康记录数据'}), 400
            
        result = algorithm_service.train_diabetes_model(data['health_records'])
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/algorithm/train/hypertension', methods=['POST'])
@token_required
def train_hypertension_model():
    """训练高血压预测模型"""
    try:
        data = request.get_json()
        if not data or 'health_records' not in data:
            return jsonify({'error': '缺少健康记录数据'}), 400
            
        result = algorithm_service.train_hypertension_model(data['health_records'])
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/algorithm/predict/risk', methods=['POST'])
@token_required
def predict_disease_risk():
    """预测疾病风险"""
    try:
        data = request.get_json()
        if not data or 'health_record' not in data:
            return jsonify({'error': '缺少健康记录数据'}), 400
            
        result = algorithm_service.predict_disease_risk(data['health_record'])
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/algorithm/assess/health', methods=['POST'])
@token_required
def assess_health_status():
    """评估健康状态"""
    try:
        data = request.get_json()
        if not data or 'health_records' not in data:
            return jsonify({'error': '缺少健康记录数据'}), 400
            
        result = algorithm_service.assess_health_status(data['health_records'])
        if 'error' in result:
            return jsonify(result), 400
            
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500 