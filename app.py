import os
import json
import requests
from flask import Flask, render_template, request, redirect, url_for, flash, session
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

# ----------------- APP SETUP -----------------
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super_secure_key_123")

# ----------------- FIREBASE CONFIG -----------------
if "FIREBASE_CONFIG" in os.environ:
    cred_dict = json.loads(os.environ["FIREBASE_CONFIG"])
    if "private_key" in cred_dict:
        cred_dict["private_key"] = cred_dict["private_key"].replace("\\n", "\n")
    cred = credentials.Certificate(cred_dict)
else:
    # Only fallback for local dev
    cred = credentials.Certificate("firebase_config.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# Firebase Web API key
FIREBASE_WEB_API_KEY = os.environ.get("FIREBASE_WEB_API_KEY", "YOUR_FIREBASE_WEB_API_KEY")

# ----------------- FLASK-LOGIN SETUP -----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"


class User(UserMixin):
    def __init__(self, uid, email):
        self.id = uid
        self.email = email

@login_manager.user_loader
def load_user(user_id):
    if session.get("user_id") == user_id:
        return User(user_id, session.get("user_email"))
    return None



# ----------------- CLOUDINARY CONFIG -----------------
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME", "dmr3w4jgu"),
    api_key=os.environ.get("CLOUDINARY_API_KEY", "321578829594452"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET", "k9XfYOMX-rWPf9kannC39Ja1sIE"),
)

videos = []  # Simple in-memory store; consider DB for production


# ----------------- ROUTES -----------------
@app.route("/")
@login_required
def home():
    return render_template("index.html")


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
            return redirect(url_for("signup"))
        else:
            uid = data["localId"]
            user = User(uid, email)
            session["user_email"] = email
            session["user_id"] = uid
            login_user(user)
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
            return redirect(url_for("login"))
        else:
            uid = data["localId"]
            user = User(uid, email)
            session["user_email"] = email
            session["user_id"] = uid
            login_user(user)
            flash("Logged in successfully!", "success")
            return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    session.pop("user_email", None)
    session.pop("user_id", None)
    flash("Logged out successfully!", "info")
    return redirect(url_for("login"))


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


@app.route("/my_videos")
@login_required
def my_videos():
    user_videos = [v for v in videos if v["uploader"] == current_user.email]
    return render_template("viewer.html", videos=user_videos)


# ----------------- MAIN -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
