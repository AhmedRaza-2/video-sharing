import os
from flask import Flask, render_template, request, redirect, url_for, flash
import cloudinary
import cloudinary.uploader

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "super_secure_key_123")  

# ----------------- CLOUDINARY CONFIG -----------------
cloudinary.config(
    cloud_name="dmr3w4jgu",
    api_key="321578829594452",
    api_secret="k9XfYOMX-rWPf9kannC39Ja1sIE",
)

videos = []

# ----------------- ROUTES -----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["GET", "POST"])
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
                    "uploader": "Anonymous",
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
def view_videos():
    return render_template("viewer.html", videos=videos)

@app.route("/my_videos")
def my_videos():
    user_videos = [v for v in videos if v["uploader"] == "Anonymous"]
    return render_template("viewer.html", videos=user_videos)

# ----------------- MAIN -----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)
