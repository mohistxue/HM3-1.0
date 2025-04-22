from flask import Blueprint, jsonify
from app.models.user import User, HealthRecord
from app.services.health_recommendation import HealthRecommendationService
from app.api.auth_api import token_required

bp = Blueprint('health_recommendation', __name__)
recommender = HealthRecommendationService()

@bp.route('/recommendation/<int:user_id>', methods=['GET'])
@token_required
def get_health_recommendation(current_user, user_id):
    """获取用户的健康建议"""
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
        return jsonify({'error': f'生成健康建议时出错：{str(e)}'}), 500

@bp.route('/analysis/<int:user_id>', methods=['GET'])
@token_required
def get_health_analysis(current_user, user_id):
    """获取用户的健康指标分析"""
    if current_user.id != user_id:
        return jsonify({'error': '无权访问其他用户的健康分析'}), 403
        
    try:
        # 获取用户最近的健康记录
        health_record = HealthRecord.query.filter_by(user_id=user_id).order_by(HealthRecord.recorded_at.desc()).first()
        
        if not health_record:
            return jsonify({'error': '未找到健康记录'}), 404
            
        # 分析健康指标
        analysis = recommender.analyze_health_metrics(health_record)
        
        return jsonify({
            'analysis': analysis
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'分析健康指标时出错：{str(e)}'}), 500 