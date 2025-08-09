from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pyrebase
import cloudinary
import cloudinary.uploader

app = Flask(__name__)
app.secret_key = 'super_secure_key_123'  # 🔐 Replace with a strong secret

# 🔐 Firebase Configuration
firebase_config = {
    "apiKey": "AIzaSyDu92oz4n6y1anUuampNve5jxrCbPWogdk",
    "authDomain": "video-656cd.firebaseapp.com",
    "databaseURL": "",  # Required by pyrebase even if unused
    "projectId": "video-656cd",
    "storageBucket": "video-656cd.appspot.com",
    "messagingSenderId": "1042620718983",
    "appId": "1:1042620718983:web:your_app_id_here"  # Replace with actual App ID
}

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# 🔑 Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Temporary user store (for session tracking)
user_store = {}

class User(UserMixin):
    def __init__(self, uid, email):
        self.id = uid
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    return user_store.get(user_id)

# 🎥 Cloudinary Configuration
cloudinary.config(
    cloud_name='dmr3w4jgu',
    api_key='321578829594452',
    api_secret='k9XfYOMX-rWPf9kannC39Ja1sIE'
)

videos = []  # Each item: {'url': ..., 'uploader': ...}

@app.route('/my-videos')
@login_required
def my_videos():
    user_videos = [v for v in videos if v['uploader'] == current_user.email]
    return render_template('viewer.html', videos=user_videos)

# 🔐 Signup Route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            return "Missing email or password"
        try:
            auth.create_user_with_email_and_password(email, password)
            return redirect(url_for('login'))
        except Exception as e:
            error_msg = str(e)
            if "EMAIL_EXISTS" in error_msg:
                return "This email is already registered. Please log in instead."
            return f"Signup failed: {error_msg}"
    return render_template('signup.html')

# 🔓 Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            return "Missing email or password"
        try:
            user = auth.sign_in_with_email_and_password(email, password)
            uid = user['localId']
            flask_user = User(uid, email)
            user_store[uid] = flask_user  # ✅ Store user for session
            login_user(flask_user)
            return redirect(url_for('home'))
        except Exception as e:
            return f"Login failed: {e}"
    return render_template('login.html')

# 🚪 Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# 🏠 Home Page
@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            upload_result = cloudinary.uploader.upload_large(
                file.stream,
                resource_type="video"
            )
            video_url = upload_result['secure_url']
            videos.append({
                'url': video_url,
                'uploader': current_user.email
            })
            return redirect(url_for('view_videos'))
    return render_template('upload.html')

# 📺 View Videos
@app.route('/videos')
@login_required
def view_videos():
    return render_template('viewer.html', videos=videos)

import os

port = int(os.environ.get("PORT", 8010))  # Azure will set PORT
app.run(host="0.0.0.0", port=port, debug=True)
