from flask import Blueprint, request, jsonify
import jwt
from datetime import datetime, timedelta
from app.models.user import User, db
from app.config import Config
from app.utils.auth import token_required

bp = Blueprint('auth', __name__)

@bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('email') or not data.get('password'):
        return jsonify({'error': '缺少必要信息'}), 400
        
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': '用户名已存在'}), 400
        
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': '邮箱已存在'}), 400
        
    user = User(
        username=data['username'],
        email=data['email']
    )
    user.password = data['password']
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({'message': '注册成功'}), 201

@bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    
    if not data or not data.get('username') or not data.get('password'):
        return jsonify({'error': '缺少必要信息'}), 400
        
    user = User.query.filter_by(username=data['username']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': '用户名或密码错误'}), 401
        
    token = jwt.encode({
        'user_id': user.id,
        'exp': datetime.utcnow() + timedelta(hours=1)
    }, Config.JWT_SECRET_KEY, algorithm='HS256')
    
    return jsonify({
        'token': token,
        'user': user.to_dict()
    }), 200

@bp.route('/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    user = User.query.get(current_user)
    if not user:
        return jsonify({'error': '用户不存在'}), 404
        
    return jsonify(user.to_dict()), 200 