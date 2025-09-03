import json
import logging
import requests
from flask_session import Session
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
import firebase_admin
from firebase_admin import credentials
import cloudinary
import cloudinary.uploader

# ----------------- LOGGING -----------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
logger.info("Starting CreatorApp...")

# ----------------- APP SETUP -----------------
app = Flask(__name__)
app.secret_key = "super_secure_key_123"

# ----------------- FIREBASE CONFIG (HARDCODED) -----------------
firebase_config = {
    "type": "service_account",
    "project_id": "video-656cd",
    "private_key_id": "8818b8bdfea9c1dc64627d12c99aa68a92898c99",
    "private_key": """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC4nFSHDBInSBYg
X88dxsATpGdsRbGuJPSs7gSGwBu9Fsod11TMPYUJa/Wk7b9E6/Xz4shjNN6RSaVq
WeZSDl4PvqMrJXXu3CEG3Pkb6TNPlitoHVSWuhqNCjUD1+5ooQ7sSyCAgtOfsM8B
bsDfhPp+XAns/SbrvE05tOiUfFnUUwrQ32aZW6R8PLGrVAjrxIWisRQ3g30dm1NQ
bYcipoi63RNM8NX+Qnq6r2pm8+WGJknwBchSabbOOPjwBksOceQIq2iAthMyb2K5
Km1HICbe7VKdd2sSe0suG5DJG+jYNnXOBderAtGFrB0r8qKCZpEUMUJg40xcZPvD
6b8MHbjZAgMBAAECggEAEWMj8O45nXqEHvt3GeJYv+DZntB3miO/6bOnOHvKgQYu
9h2MTooyx/7jjWuY+qhQq24+Gl3l4oAxtLEP6MWSpV/6pTsrfto7wBY63h6aJxJJ
N06f42xWyNbfxGNngHI+4hF3V7M6tE0mSgfA4ax4HUOU6b20FzrOeTNpPmbx3PXm
GsOZcSs7CZAh7JVBuvOpm/MsFHq2xWgW4H88dZqWboZbHQn/wqkO4hTaYWelhlPC
bsyLFBicbMEIl5mMn0hPbL8qVdxAKvOa+D+QXp1h+H+Ef+8khWkwFXWUketPScpa
JtUGQF7J+UiedXVdwNxKJ+PZHfLtpDNWBOQjbtKi4wKBgQDya1YpadW6vQXJpKaj
bTZC7RmLVNVIN6Do2gMdMhXrV40Ha7xGscHfOnGDPS2Izk+rTj4NZrO+31ktTHgi
qab8OzJs5JFGc7qxw57F/cIJT+mWQTKFfTh11wb/mlSzySZfDpb7AIcuodZ+EgEk
tatB1E37O21f/CKXAFlbljtOowKBgQDC8+34s1I6Qc9YeYMZtlKDrJSRbEJc9R9B
BrMz97JPqRTLbtsmR4Mis4kjj02CUM3w1OCj+syyDfX9ltEb2s9605tnMXKVaOz5
oM/qTTbbRFcz7sq3Jn7OLgOVCjPsaZhqzvqF7UaOQOYeQh5Q+jTZaeatsy0nKRQp
UmUYvwV+UwKBgCX+7OJQ1E8QkXepdvTmiTq0LuzHvyYykeXtRc+tqgHZFyGyoS/z
bI+weVo4nIp0y8ft24v+LO4d07xl3+6O6L1gCedHa/2+5eQ25QvjWiZbgCEs1t5V
YiQWL+KgLeaAAKOlhcSRsJ5+f0ADUmqOjukifZaDGgGRY1qHk3nnciRBAoGAW6N5
2wu/vS6uHnKP04hGZSq8c1cmIrf+Rvy1Q9pM8PETm0Syst2uoKMv9Y0o6/a7t1b5
eVss2Q2C8f7wsF08ZgoN5IXzzJOTwQt8cDB3dr47F2hJ1am8tYIfoPE40woX4S0F
yeps3fVXtiVyRrI2IXSMQF4W/W9r0LiwWN+B340CgYEAueQaiLgvgbFPxn1imE25
JXiIerhcjvbL5XOEaMayb1Jpazo88AGX9GaslxE9AE0XyMlw1+vmF5y5Y/xyjBLR
UJi/Nv8rgmRgD9Irihut3f3/X3Ku/JOZmt9c0UeMfoc0F9yMJvOrjINTYOB2n2h9
I7wr4Gh+26oRWUnvjNBNQ0s=
-----END PRIVATE KEY-----""",
    "client_email": "firebase-adminsdk-fbsvc@video-656cd.iam.gserviceaccount.com",
    "client_id": "102872420138978231133",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-fbsvc@video-656cd.iam.gserviceaccount.com"
}

cred = credentials.Certificate(firebase_config)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

FIREBASE_WEB_API_KEY = "AIzaSyDu92oz4n6y1anUuampNve5jxrCbPWogdk"

# Configure server-side session (important for Azure!)
app.config["SESSION_TYPE"] = "filesystem"  # or "redis" if you want faster
app.config["SESSION_FILE_DIR"] = "/tmp/flask_session"  # works in Azure Linux
app.config["SESSION_PERMANENT"] = False
Session(app)

# ----------------- FLASK-LOGIN -----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

user_store = {}  # Global user store

class User(UserMixin):
    def __init__(self, uid, email):
        self.id = uid
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    user = user_store.get(user_id)
    logger.debug(f"Loading user {user_id}: {user}")
    return user

# ----------------- CLOUDINARY -----------------
cloudinary.config(
    cloud_name="dmr3w4jgu",
    api_key="321578829594452",
    api_secret="k9XfYOMX-rWPf9kannC39Ja1sIE"
)

videos = []

# ----------------- ROUTES -----------------
@app.route("/")
@login_required
def home():
    logger.debug(f"Accessing home page: {current_user.email}")
    return render_template("index.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        logger.debug(f"Signup attempt: {email}")

        if not email or not password:
            flash("Missing email or password", "error")
            logger.warning("Signup failed: missing email or password")
            return redirect(url_for("signup"))

        try:
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
            payload = {"email": email, "password": password, "returnSecureToken": True}
            resp = requests.post(url, json=payload)
            data = resp.json()
            logger.debug(f"Firebase signup response: {data}")

            if "error" in data:
                flash(f"Signup failed: {data['error']['message']}", "error")
                return redirect(url_for("signup"))

            uid = data["localId"]
            user = User(uid, email)
            user_store[uid] = user
            login_user(user)
            logger.info(f"User signed up: {email}")
            flash("Account created & logged in!", "success")
            return redirect(url_for("home"))
        except Exception as e:
            logger.exception(f"Signup exception: {e}")
            flash(f"Signup error: {str(e)}", "error")
            return redirect(url_for("signup"))

    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        logger.debug(f"Login attempt: {email}")

        if not email or not password:
            flash("Missing email or password", "error")
            return redirect(url_for("login"))

        try:
            url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
            payload = {"email": email, "password": password, "returnSecureToken": True}
            resp = requests.post(url, json=payload)
            data = resp.json()
            logger.debug(f"Firebase login response: {data}")

            if "error" in data:
                flash(f"Login failed: {data['error']['message']}", "error")
                return redirect(url_for("login"))

            uid = data["localId"]
            user = User(uid, email)
            user_store[uid] = user
            login_user(user)
            logger.info(f"User logged in: {email}")
            flash("Logged in successfully!", "success")
            return redirect(url_for("home"))
        except Exception as e:
            logger.exception(f"Login exception: {e}")
            flash(f"Login error: {str(e)}", "error")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    try:
        email = current_user.email
        logout_user()
        logger.info(f"User logged out: {email}")
        flash("Logged out successfully!", "info")
    except Exception as e:
        logger.exception(f"Logout error: {e}")
    return redirect(url_for("login"))

@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        logger.debug(f"Upload attempt: {file.filename if file else 'No file'} by {current_user.email}")

        if not file or file.filename == "":
            flash("No file selected", "error")
            return redirect(url_for("upload"))

        try:
            result = cloudinary.uploader.upload_large(file.stream, resource_type="video", folder="video_uploads")
            videos.append({
                "url": result["secure_url"],
                "uploader": current_user.email,
                "filename": file.filename,
                "upload_time": result.get("created_at", "")
            })
            logger.info(f"Video uploaded: {file.filename}")
            flash("Video uploaded successfully!", "success")
            return redirect(url_for("view_videos"))
        except Exception as e:
            logger.exception(f"Upload failed: {e}")
            flash(f"Upload failed: {str(e)}", "error")
            return redirect(url_for("upload"))

    return render_template("upload.html")

@app.route("/videos")
@login_required
def view_videos():
    logger.debug(f"Viewing all videos by {current_user.email}")
    return render_template("viewer.html", videos=videos)

@app.route("/my_videos")
@login_required
def my_videos():
    user_videos = [v for v in videos if v["uploader"] == current_user.email]
    logger.debug(f"My videos count: {len(user_videos)}")
    return render_template("viewer.html", videos=user_videos)

# ----------------- RUN APP -----------------
if __name__ == "__main__":
    logger.info("Running Flask app...")
    app.run(debug=True, host="0.0.0.0", port=8000)
