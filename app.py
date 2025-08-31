import os
import json
import requests
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

app = Flask(__name__)
app.secret_key = 'super_secure_key_123'  # replace with env var in prod

# ----------------- FIREBASE CONFIG -----------------
if "FIREBASE_CONFIG" in os.environ:
    cred_dict = json.loads(os.environ["FIREBASE_CONFIG"])
    if "private_key" in cred_dict:
        cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(cred_dict)
else:
    cred = credentials.Certificate("firebase_config.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Firebase Web API key
FIREBASE_WEB_API_KEY = "AIzaSyDu92oz4n6y1anUuampNve5jxrCbPWogdk"

# ----------------- FLASK-LOGIN SETUP -----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

user_store = {}


class User(UserMixin):
    def __init__(self, uid, email):
        self.id = uid
        self.email = email


@login_manager.user_loader
def load_user(user_id):
    return user_store.get(user_id)


# ----------------- CLOUDINARY CONFIG -----------------
cloudinary.config(
    cloud_name="dmr3w4jgu",
    api_key="321578829594452",
    api_secret="k9XfYOMX-rWPf9kannC39Ja1sIE",
)

videos = []


# ----------------- ROUTES -----------------
@app.route("/my_videos")
@login_required
def my_videos():
    user_videos = [v for v in videos if v["uploader"] == current_user.email]
    return render_template("viewer.html", videos=user_videos)

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Missing email or password", "error")
            return redirect(url_for("signup"))

        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={FIREBASE_WEB_API_KEY}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()

        if "error" in data:
            flash(f"Signup failed: {data['error']['message']}", "error")
            return redirect(url_for("signup"))  # ✅ redirect on error
        else:
            uid = data["localId"]
            flask_user = User(uid, email)
            user_store[uid] = flask_user
            login_user(flask_user)
            flash("Account created & logged in successfully!", "success")
            return redirect(url_for("home"))
    return render_template("signup.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        if not email or not password:
            flash("Missing email or password", "error")
            return redirect(url_for("login"))

        url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={FIREBASE_WEB_API_KEY}"
        payload = {"email": email, "password": password, "returnSecureToken": True}
        response = requests.post(url, json=payload, timeout=10)
        data = response.json()

        if "error" in data:
            flash(f"Login failed: {data['error']['message']}", "error")
            return redirect(url_for("login"))  # ✅ redirect on error
        else:
            uid = data["localId"]
            flask_user = User(uid, email)
            user_store[uid] = flask_user
            login_user(flask_user)
            flash("Logged in successfully!", "success")
            return redirect(url_for("home")) # ✅ redirect on success

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully!", "info")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def home():
    return render_template("index.html")


@app.route("/upload", methods=["GET", "POST"])
@login_required
def upload():
    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            flash("No file selected", "error")
            return redirect(url_for("upload"))

        try:
            upload_result = cloudinary.uploader.upload_large(
                file.stream, resource_type="video", folder="video_uploads"
            )
            video_url = upload_result["secure_url"]
            videos.append(
                {
                    "url": video_url,
                    "uploader": current_user.email,
                    "filename": file.filename,
                    "upload_time": upload_result.get("created_at", ""),
                }
            )
            flash("Video uploaded successfully!", "success")
            return redirect(url_for("view_videos"))
        except Exception as e:
            flash(f"Upload failed: {str(e)}", "error")
            return redirect(url_for("upload"))

    return render_template("upload.html")


@app.route("/videos")
@login_required
def view_videos():
    return render_template("viewer.html", videos=videos)


# ----------------- MAIN -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
