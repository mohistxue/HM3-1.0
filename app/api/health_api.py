# 健康数据API
from flask import Blueprint, request, jsonify
from app.models.user import User, HealthRecord
from app.services.health_recommendation import HealthRecommendationService
from app import db
from datetime import datetime
from app.api.auth_api import token_required

bp = Blueprint('health', __name__)
recommender = HealthRecommendationService()

@bp.route('/records', methods=['POST'])
@token_required
def add_health_record(current_user):
    data = request.get_json()
    
    if not data:
        return jsonify({'error': '缺少数据'}), 400
        
    record = HealthRecord(
        user_id=current_user.id,
        heart_rate=data.get('heart_rate'),
        blood_pressure=data.get('blood_pressure'),
        blood_sugar=data.get('blood_sugar'),
        weight=data.get('weight'),
        sleep_hours=data.get('sleep_hours'),
        mood_score=data.get('mood_score')
    )
    
    db.session.add(record)
    db.session.commit()
    
    return jsonify(record.to_dict()), 201

@bp.route('/records', methods=['GET'])
@token_required
def get_health_records(current_user):
    records = HealthRecord.query.filter_by(user_id=current_user.id).all()
    return jsonify([record.to_dict() for record in records]), 200

@bp.route('/records/<int:record_id>', methods=['GET'])
@token_required
def get_health_record(current_user, record_id):
    record = HealthRecord.query.filter_by(id=record_id, user_id=current_user.id).first()
    if not record:
        return jsonify({'error': '记录不存在'}), 404
    return jsonify(record.to_dict()), 200

@bp.route('/records/<int:record_id>', methods=['PUT'])
@token_required
def update_health_record(current_user, record_id):
    record = HealthRecord.query.filter_by(id=record_id, user_id=current_user.id).first()
    if not record:
        return jsonify({'error': '记录不存在'}), 404
        
    data = request.get_json()
    if not data:
        return jsonify({'error': '缺少数据'}), 400
        
    record.heart_rate = data.get('heart_rate', record.heart_rate)
    record.blood_pressure = data.get('blood_pressure', record.blood_pressure)
    record.blood_sugar = data.get('blood_sugar', record.blood_sugar)
    record.weight = data.get('weight', record.weight)
    record.sleep_hours = data.get('sleep_hours', record.sleep_hours)
    record.mood_score = data.get('mood_score', record.mood_score)
    
    db.session.commit()
    return jsonify(record.to_dict()), 200

@bp.route('/records/<int:record_id>', methods=['DELETE'])
@token_required
def delete_health_record(current_user, record_id):
    record = HealthRecord.query.filter_by(id=record_id, user_id=current_user.id).first()
    if not record:
        return jsonify({'error': '记录不存在'}), 404
        
    db.session.delete(record)
    db.session.commit()
    return jsonify({'message': '记录已删除'}), 200

@bp.route('/recommendation/<int:user_id>', methods=['GET'])
@token_required
def get_health_recommendation(current_user, user_id):
    if current_user.id != user_id:
        return jsonify({'error': '无权访问其他用户的健康建议'}), 403
        
    try:
        # 获取用户最近的健康记录
        health_record = HealthRecord.query.filter_by(user_id=user_id).order_by(HealthRecord.recorded_at.desc()).first()
        
        if not health_record:
            return jsonify({'error': '未找到健康记录'}), 404
            
        # 分析健康指标
        analysis = recommender.analyze_health_metrics(health_record)
        
        # 生成个性化建议
        recommendations = recommender.generate_recommendations(analysis)
        
        return jsonify({
            'analysis': analysis,
            'recommendations': recommendations
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'获取健康建议失败: {str(e)}'}), 500

@bp.route('/train', methods=['POST'])
@token_required
def train_model(current_user):
    try:
        # 获取用户的所有健康记录
        records = HealthRecord.query.filter_by(user_id=current_user.id).all()
        
        if len(records) < 10:
            return jsonify({'error': '需要至少10条健康记录来训练模型'}), 400
            
        # 准备训练数据
        training_data = []
        for record in records:
            data = {
                'heart_rate': record.heart_rate,
                'blood_pressure': record.blood_pressure,
                'blood_sugar': record.blood_sugar,
                'weight': record.weight,
                'sleep_hours': record.sleep_hours,
                'mood_score': record.mood_score
            }
            training_data.append(data)
            
        # 使用健康推荐服务训练模型
        recommender.train_model(training_data)
        
        return jsonify({
            'message': '模型训练成功',
            'data_points': len(records)
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'模型训练失败: {str(e)}'}), 500 