# 健康建议API
from flask import Blueprint, jsonify
from app.models.user import HealthRecord
from app.api.auth_api import token_required
from app.services.health_recommendation import HealthRecommendationService

bp = Blueprint('recommendation', __name__, url_prefix='/api/recommendation')

@bp.route('/health/<int:user_id>', methods=['GET'])
@token_required
def get_health_recommendation(current_user, user_id):
    if current_user.id != user_id:
        return jsonify({'error': '无权访问其他用户的健康建议'}), 403
        
    try:
        # 获取用户最近的健康记录
        health_record = HealthRecord.query.filter_by(user_id=user_id).order_by(HealthRecord.recorded_at.desc()).first()
        
        if not health_record:
            return jsonify({'error': '未找到健康记录'}), 404
            
        # 创建健康推荐服务实例
        recommendation_service = HealthRecommendationService()
        
        # 分析健康指标
        analysis = recommendation_service.analyze_health_metrics(health_record)
        
        # 生成个性化建议
        recommendations = recommendation_service.generate_recommendations(analysis)
        
        return jsonify({
            'analysis': analysis,
            'recommendations': recommendations
        }), 200
        
    except Exception as e:
        return jsonify({'error': f'获取健康建议失败: {str(e)}'}), 500 