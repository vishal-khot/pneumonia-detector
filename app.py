from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
from keras.models import load_model
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "static/uploads")

app.secret_key = "secret key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(["jpg", "jpeg"])

model = load_model("pneumonia.h5")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    imgs = os.listdir(app.config["UPLOAD_FOLDER"])
    for img in imgs:
        os.remove(os.path.join(app.config["UPLOAD_FOLDER"], img))
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def upload_image():

    if "file" not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files["file"]

    if file.filename == "":
        flash("No image selected for uploading")
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        pred_class, confidence = "", 0
        img = cv2.imread(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.convertScaleAbs(img, alpha=1.3, beta=0)
            img = cv2.resize(img, (128, 128))
            img = np.array(img) / 255
            img = np.reshape(img, (1, 128, 128, 1))

            prediction = model.predict(img)[0][0]
            if prediction < 0.5:
                pred_class = "NORMAL"
                confidence = 1 - prediction
            else:
                pred_class = "PNEUMONIA"
                confidence = prediction

            flash("Image successfully uploaded and displayed below")

        return render_template(
            "upload.html",
            filename=filename,
            pred=pred_class,
            conf=round(confidence * 100, 2),
        )

    else:
        flash("Allowed image types are - jpg, jpeg")
        return redirect(request.url)


@app.route("/display/<filename>")
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for("static", filename="uploads/" + filename), code=301)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
