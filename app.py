import os
import json
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import firebase_admin
from firebase_admin import credentials, auth as firebase_auth
import cloudinary
import cloudinary.uploader

app = Flask(__name__)
app.secret_key = 'super_secure_key_123'  # 🔐 Replace with a strong secret

import firebase_admin
from firebase_admin import credentials

cred = credentials.Certificate("firebase_config.json")
firebase_admin.initialize_app(cred)


# 🔑 Firebase Web API Key
FIREBASE_WEB_API_KEY = "AIzaSyDu92oz4n6y1anUuampNve5jxrCbPWogdk"

# 🔑 Flask-Login Setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

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

videos = []

# ✅ Helper function to verify password
def verify_firebase_password(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Password verification error: {e}")
        return None

# Routes
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            flash("Missing email or password", "error")
            return render_template('signup.html')
        try:
            firebase_auth.create_user(email=email, password=password)
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            flash(f"Signup failed: {str(e)}", "error")
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            flash("Missing email or password", "error")
            return render_template('login.html')
        try:
            user_record = firebase_auth.get_user_by_email(email)
            auth_result = verify_firebase_password(email, password)
            if auth_result:
                uid = user_record.uid
                flask_user = User(uid, email)
                user_store[uid] = flask_user
                login_user(flask_user)
                flash("Logged in successfully!", "success")
                return redirect(url_for('home'))
            else:
                flash("Invalid email or password", "error")
        except Exception as e:
            flash(f"Login failed: {str(e)}", "error")
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully!", "info")
    return redirect(url_for('login'))

@app.route('/')
@login_required
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected", "error")
            return render_template('upload.html')
        try:
            upload_result = cloudinary.uploader.upload_large(
                file.stream, resource_type="video", folder="video_uploads"
            )
            video_url = upload_result['secure_url']
            videos.append({
                'url': video_url,
                'uploader': current_user.email,
                'filename': file.filename,
                'upload_time': upload_result.get('created_at', '')
            })
            flash("Video uploaded successfully!", "success")
            return redirect(url_for('view_videos'))
        except Exception as e:
            flash(f"Upload failed: {str(e)}", "error")
    return render_template('upload.html')

@app.route('/videos')
@login_required
def view_videos():
    return render_template('viewer.html', videos=videos)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
