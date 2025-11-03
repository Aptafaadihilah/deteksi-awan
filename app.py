import os
import json
from flask import Flask, render_template, request, redirect, url_for, session, flash
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

# Konfigurasi
MODEL_PATH = "models/cloud_model.h5"
CLASS_IDX_PATH = "models/class_indices.json"
IMG_SIZE = (224, 224)
USERS_FILE = "users.json"

app = Flask(__name__)
app.secret_key = "rahasia123"  # ubah ke secret key milikmu

# Load model
model = load_model(MODEL_PATH)
with open(CLASS_IDX_PATH, "r") as f:
    class_indices = json.load(f)
idx_to_class = {v: k for k, v in class_indices.items()}


# ---------- FUNGSI TAMBAHAN ----------
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)
    return x


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=4)


# ---------- ROUTE LOGIN / REGISTER ----------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()
        if username in users:
            flash("Username sudah digunakan!", "danger")
            return redirect(url_for("register"))

        users[username] = password
        save_users(users)
        flash("Registrasi berhasil! Silakan login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        users = load_users()
        if username in users and users[username] == password:
            session["username"] = username
            return redirect(url_for("index"))
        else:
            flash("Username atau password salah!", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("username", None)
    flash("Berhasil logout!", "info")
    return redirect(url_for("login"))


# ---------- HALAMAN UTAMA ----------
@app.route("/", methods=["GET", "POST"])
def index():
    if "username" not in session:
        return redirect(url_for("login"))

    label, confidence, image_path = None, None, None
    gray_path, edge_path, resized_path = None, None, None

    if request.method == "POST":
        file = request.files["file"]
        if file and file.filename != "":
            upload_folder = os.path.join("static", "uploads")
            preprocess_folder = os.path.join("static", "preprocess")
            os.makedirs(upload_folder, exist_ok=True)
            os.makedirs(preprocess_folder, exist_ok=True)

            # Simpan gambar upload
            save_path = os.path.join(upload_folder, file.filename)
            file.save(save_path)

            # ---------- Tahapan Preprocessing ----------
            img = cv2.imread(save_path)

            # 1️⃣ Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray_path = os.path.join(preprocess_folder, f"gray_{file.filename}")
            cv2.imwrite(gray_path, gray)

            # 2️⃣ Edge Detection
            edges = cv2.Canny(gray, 100, 200)
            edge_path = os.path.join(preprocess_folder, f"edge_{file.filename}")
            cv2.imwrite(edge_path, edges)

            # 3️⃣ Resize ke 224x224
            resized = cv2.resize(img, IMG_SIZE)
            resized_path = os.path.join(preprocess_folder, f"resized_{file.filename}")
            cv2.imwrite(resized_path, resized)

            # ---------- Prediksi Model ----------
            x = prepare_image(save_path)
            preds = model.predict(x)[0]
            top_idx = np.argmax(preds)
            confidence = float(preds[top_idx]) * 100
            label = idx_to_class[top_idx]

            # Kirim ke HTML
            return render_template(
                "index.html",
                label=label,
                confidence=round(confidence, 2),
                image_path=f"uploads/{file.filename}",
                gray_path=f"preprocess/gray_{file.filename}",
                edge_path=f"preprocess/edge_{file.filename}",
                resized_path=f"preprocess/resized_{file.filename}",
                username=session["username"]
            )

    return render_template(
        "index.html",
        label=None,
        confidence=None,
        image_path=None,
        gray_path=None,
        edge_path=None,
        resized_path=None,
        username=session["username"]
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
