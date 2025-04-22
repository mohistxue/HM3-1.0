import os
os.environ['FLASK_APP'] = 'run.py'
from app import create_app

app = create_app()

if __name__ == '__main__':
    print("正在启动应用程序...")
    print("访问地址: http://localhost:5000")
    app.run(debug=True, host='127.0.0.1', port=5000) 