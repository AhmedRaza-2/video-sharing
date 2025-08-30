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

# 🔐 Firebase Admin SDK Configuration with your actual credentials
firebase_config = {
    "type": "service_account",
    "project_id": "video-656cd",
    "private_key_id": "13908bf668393201a1ed9cfd1786486a6f977420",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCbWP5Sp3dPGnS3\n+BfnNVfzvKkonI37pGxehyC5INUN6o4iyLyqNlnZggj+qztH4qFOpahENhZQkMCj\nRXS3XyjJ1BDH6QX8aRZwKIoddTYiUQGjMg1UwAfodd01FZvq4OX2VH+aHRKkNeGK\nH7k1p+D3MRkDHOY4XXuNouywoSecMDCXmd8IHbkmtbPpF3yZmyRr+t81hmO4YvrQ\nIxToMtM0k6j06hUHZJyrsE0kckG6+rq4w/OGALv2HfarpSp9tm4MameNNSjFOBSK\nHLEqn4nwnNd+717i3vP28uEa1ihpKtWqo5xe4jpT9r8taiz4wv0KuLxviejcEfqD\nNGLEaVsZAgMBAAECggEAFtsiyAcVpGYlvW+GF3bKEvVeBLihTIs7hTOSb/UnA54l\nUsmAC0xl2dJU/8syX02LzXt4D1kkiAP+znGIo6DKP9pbVwkE4ByXYuHEDbjSFyhz\nT48sq5x5TWK8rMkaIJZS9J7StgUIWhqyAqrPitfBX1KHoI/0RgoTBJrFAj55CnSd\n1C2hCFopWPiEamaHBxnPtzlFDh7l6CXYwvgfSujelIFl7W1Vb19b2937ckKLvUdg\nsCtGbuLyk28We87sz0cpQLPTqHh4CEjCCpBZY4DuaDZ1ZzqBy8W347Rb1mNRoZEy\n04KaSe+JpUpgFYBPmrFlqPJ8KfuhlqfV+Whs42Bm9QKBgQDUn1g0KCMs2BvlCxKB\n5mMSjwT9on58V9psgvPmJOt0T8mcJ96Y2WDTAzcAkl+TFPwR0OvPsDmRoGXdXgra\nsqq4EXNW+DvOyda1VrzZTJoN6OdO5WY36z0OtKRkpKIf/Hj9EJsGMUcqqt06E9mb\njGkvhzqTaHMv+H1ZpAjhtw4nywKBgQC7Clnm0ws88B1KAaaH8eXlErJ9qq3PvOWq\n7UXHPHtyrk9hIIlQ66n7Za3k9XUH4Jz3jrxqQgvEHh2S1QxFzhvLb58z3ga6XQ/W\nmnTBuFAAsdDfRlcva50pNvjgO6O5/cQlYcpFb3ZPWUX7BMreP7HAkOWBilr5kuli\ncDeqMQyEKwKBgCM4YfwByhHbmoNOWjp6V17zofgBusIOK3heGNi+tOIHdXYQhKb9\nGzTZC3tkw8Axca/h064Lmv2sfDM3KMUhY3YqLdjyNbYDaTWQsKeuMLatJePDzDLn\nHK4a7kBdpR13TPNelM9pykgfFZZRQ3Kox6O/2swgOTRxDRKUQYRiNk4RAoGAZSlm\nEngW3fCohrx5y5FD3C/OpjVIKNFsDpSiZu4Jfq9Uc53bZw7vMu99rBYuJAKSIzrq\nITZzkrEPIbllF+QwGEnY+36cePOYe2OyvovniVijNf+fbCByMjZvSSg4l4HvMqC/\ne+qbPLN2LBwddTNR+mrkFro0FkQlQn6bPMumj10CgYEAnSqAMfbroxJ7UFIYO431\nrYiaRwHKt8Q7LQDO9h6p8FD9y+XG6I67soi897wLZjNriyLC029pnfJyJ1BYel6W\njv0Th4wyCjMXSFrfXuEzz2G+vBTFmmU7l/YSbEROgj9BmW5qKmdIhCmlIrP3kToa\n/sswieHMqUMnwJi2qYl9XiY=\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-fbsvc@video-656cd.iam.gserviceaccount.com",
    "client_id": "102872420138978231133",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc%40video-656cd.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Initialize Firebase Admin SDK
if not firebase_admin._apps:
    cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred)

# Your Firebase Web API Key
FIREBASE_WEB_API_KEY = "AIzaSyDu92oz4n6y1anUuampNve5jxrCbPWogdk"

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

# Helper function to verify password using Firebase Auth REST API
def verify_firebase_password(email, password):
    """Verify password using Firebase Auth REST API"""
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
    
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Password verification error: {e}")
        return None

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
            flash("Missing email or password", "error")
            return render_template('signup.html')
        try:
            # Create user with Firebase Admin SDK
            user_record = firebase_auth.create_user(
                email=email,
                password=password
            )
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for('login'))
        except firebase_auth.EmailAlreadyExistsError:
            flash("This email is already registered. Please log in instead.", "error")
            return render_template('signup.html')
        except Exception as e:
            flash(f"Signup failed: {str(e)}", "error")
            return render_template('signup.html')
    return render_template('signup.html')

# 🔓 Login Route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if not email or not password:
            flash("Missing email or password", "error")
            return render_template('login.html')
        
        try:
            # First verify the user exists
            user_record = firebase_auth.get_user_by_email(email)
            
            # Verify password using Firebase REST API
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
                return render_template('login.html')
                
        except firebase_auth.UserNotFoundError:
            flash("User not found. Please sign up first.", "error")
            return render_template('login.html')
        except Exception as e:
            flash(f"Login failed: {str(e)}", "error")
            return render_template('login.html')
    return render_template('login.html')

# 🚪 Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully!", "info")
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
        file = request.files.get('file')
        if not file or file.filename == '':
            flash("No file selected", "error")
            return render_template('upload.html')
        
        try:
            upload_result = cloudinary.uploader.upload_large(
                file.stream,
                resource_type="video",
                folder="video_uploads"  # Optional: organize uploads in folders
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
    return render_template('upload.html')

# 📺 View Videos
@app.route('/videos')
@login_required
def view_videos():
    return render_template('viewer.html', videos=videos)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
