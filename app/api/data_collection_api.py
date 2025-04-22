# 数据收集API
from flask import Blueprint, request, jsonify
from app.services.data_collection import DataCollectionService
from app.api.auth import token_required
import pandas as pd
import json

bp = Blueprint('data_collection', __name__)
data_service = DataCollectionService()

@bp.route('/api/data/hospital', methods=['POST'])
@token_required
def fetch_hospital_data(current_user):
    """获取医院数据"""
    try:
        data = request.get_json()
        api_url = data.get('api_url')
        params = data.get('params', {})
        
        if not api_url:
            return jsonify({'error': '缺少API地址'}), 400
            
        df = data_service.fetch_hospital_data(api_url, params)
        if df.empty:
            return jsonify({'error': '获取数据失败'}), 400
            
        # 数据预处理
        df = data_service.preprocess_data(df)
        
        return jsonify({
            'message': '数据获取成功',
            'data': df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/data/wearable', methods=['POST'])
@token_required
def fetch_wearable_data(current_user):
    """获取可穿戴设备数据"""
    try:
        data = request.get_json()
        device_type = data.get('device_type')
        user_id = current_user.id
        
        if not device_type:
            return jsonify({'error': '缺少设备类型'}), 400
            
        df = data_service.fetch_wearable_data(device_type, user_id)
        if df.empty:
            return jsonify({'error': '获取数据失败'}), 400
            
        # 数据预处理
        df = data_service.preprocess_data(df)
        
        return jsonify({
            'message': '数据获取成功',
            'data': df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/data/preprocess', methods=['POST'])
@token_required
def preprocess_data(current_user):
    """数据预处理"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': '缺少数据'}), 400
            
        # 将JSON数据转换为DataFrame
        df = pd.DataFrame(data)
        
        # 数据预处理
        df = data_service.preprocess_data(df)
        
        return jsonify({
            'message': '数据预处理成功',
            'data': df.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/data/sentiment', methods=['POST'])
@token_required
def analyze_sentiment(current_user):
    """情感分析"""
    try:
        data = request.get_json()
        text = data.get('text')
        
        if not text:
            return jsonify({'error': '缺少文本'}), 400
            
        result = data_service.analyze_sentiment(text)
        
        return jsonify({
            'message': '情感分析成功',
            'result': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500 