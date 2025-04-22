# 健康记录API
from flask import Blueprint, request, jsonify
from app.models.user import HealthRecord
from app import db
from app.api.auth_api import token_required

bp = Blueprint('health_record', __name__, url_prefix='/api/health')

@bp.route('/records', methods=['POST'])
@token_required
def add_health_record(current_user):
    data = request.get_json()
    
    # 验证必需的字段
    required_fields = ['heart_rate', 'blood_pressure', 'blood_sugar', 
                      'weight', 'sleep_hours', 'mood_score']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'缺少必需字段: {field}'}), 400
    
    # 创建新的健康记录
    health_record = HealthRecord(
        user_id=current_user.id,
        heart_rate=data['heart_rate'],
        blood_pressure=data['blood_pressure'],
        blood_sugar=data['blood_sugar'],
        weight=data['weight'],
        sleep_hours=data['sleep_hours'],
        mood_score=data['mood_score']
    )
    
    try:
        db.session.add(health_record)
        db.session.commit()
        return jsonify({'message': '健康记录添加成功', 'record': health_record.to_dict()}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'添加健康记录失败: {str(e)}'}), 500

@bp.route('/records', methods=['GET'])
@token_required
def get_health_records(current_user):
    try:
        records = HealthRecord.query.filter_by(user_id=current_user.id).all()
        return jsonify({
            'records': [record.to_dict() for record in records]
        }), 200
    except Exception as e:
        return jsonify({'error': f'获取健康记录失败: {str(e)}'}), 500 