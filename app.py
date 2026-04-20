import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from datetime import datetime
from functools import wraps

app = Flask(__name__)
CORS(app)

# ── CONFIG ──────────────────────────────────────────────────────────────────
app.config['SECRET_KEY'] = 'change-this-to-a-long-random-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///teadiagnosticx.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ── MODELS ───────────────────────────────────────────────────────────────────

class User(db.Model):
    __tablename__ = 'users'
    id             = db.Column(db.Integer, primary_key=True)
    first_name     = db.Column(db.String(80), nullable=False)
    last_name      = db.Column(db.String(80), nullable=False)
    email          = db.Column(db.String(150), unique=True, nullable=False)
    phone          = db.Column(db.String(30))
    location       = db.Column(db.String(150))
    password_hash  = db.Column(db.String(256), nullable=False)
    created_at     = db.Column(db.DateTime, default=datetime.utcnow)
    # Relationship: one user → many predictions
    predictions    = db.relationship('Prediction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Prediction(db.Model):
    __tablename__ = 'predictions'
    id            = db.Column(db.Integer, primary_key=True)
    user_id       = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)  # nullable = allow guests
    image_data    = db.Column(db.LargeBinary, nullable=False)   # raw image bytes
    image_name    = db.Column(db.String(200))
    prediction    = db.Column(db.String(100), nullable=False)
    confidence    = db.Column(db.Float, nullable=False)
    treatment     = db.Column(db.Text)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)


# ── ML MODEL ─────────────────────────────────────────────────────────────────

model = tf.keras.models.load_model("tea_leaf_model.keras")

with open("class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

IMG_SIZE = 128

treatments = {
    "Healthy leaf": "No treatment needed. Maintain proper irrigation and nutrient supply.",
    "Gray Blight":  "Remove infected leaves and spray Copper Oxychloride or Carbendazim fungicide every 7–10 days.",
    "Brown Blight": "Prune infected areas and apply Mancozeb fungicide.",
    "Red Rust":     "Apply Bordeaux mixture or Copper fungicide.",
    "Leaf Spot":    "Use Chlorothalonil fungicide and avoid excess moisture."
}


def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image


# ── AUTH HELPERS ──────────────────────────────────────────────────────────────

def login_required(f):
    """Decorator to protect routes that require authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


# ── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/analyze")
def analyze():
    return render_template("analyze.html")


# ── REGISTER ─────────────────────────────────────────────────────────────────

@app.route("/register", methods=["GET", "POST"])
def register():
    if 'user_id' in session:
        return redirect(url_for('home'))

    if request.method == "POST":
        first_name       = request.form.get("first_name", "").strip()
        last_name        = request.form.get("last_name", "").strip()
        email            = request.form.get("email", "").strip().lower()
        phone            = request.form.get("phone", "").strip()
        location         = request.form.get("location", "").strip()
        password         = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        # Validation
        if not all([first_name, last_name, email, password]):
            flash("Please fill in all required fields.", "error")
            return render_template("register.html")

        if password != confirm_password:
            flash("Passwords do not match.", "error")
            return render_template("register.html")

        if len(password) < 8:
            flash("Password must be at least 8 characters.", "error")
            return render_template("register.html")

        if User.query.filter_by(email=email).first():
            flash("An account with that email already exists.", "error")
            return render_template("register.html")

        user = User(
            first_name=first_name,
            last_name=last_name,
            email=email,
            phone=phone,
            location=location,
        )
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        flash("Account created successfully! Please log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# ── LOGIN ─────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if 'user_id' in session:
        return redirect(url_for('home'))

    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()

        if not user or not user.check_password(password):
            flash("Invalid email or password.", "error")
            return render_template("login.html")

        session['user_id']   = user.id
        session['user_name'] = user.first_name
        session.permanent    = 'remember' in request.form

        flash(f"Welcome back, {user.first_name}!", "success")
        return redirect(url_for("home"))

    return render_template("login.html")


# ── LOGOUT ────────────────────────────────────────────────────────────────────

@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("home"))


# ── PREDICT ───────────────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()                    # read raw bytes for DB storage
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    processed = preprocess_image(image)

    prediction_arr = model.predict(processed)
    predicted_index = np.argmax(prediction_arr)
    predicted_class = class_names[predicted_index].split(".")[-1].strip()
    confidence      = float(np.max(prediction_arr)) * 100
    treatment       = treatments.get(predicted_class, "No treatment information available.")

    # Save to database
    record = Prediction(
        user_id    = session.get('user_id'),   # None if not logged in
        image_data = image_bytes,
        image_name = file.filename or "upload.jpg",
        prediction = predicted_class,
        confidence = round(confidence, 2),
        treatment  = treatment,
    )
    db.session.add(record)
    db.session.commit()

    return jsonify({
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "treatment":  treatment
    })


# ── HISTORY (logged-in users only) ───────────────────────────────────────────

@app.route("/history")
@login_required
def history():
    user_id = session['user_id']
    records = (
        Prediction.query
        .filter_by(user_id=user_id)
        .order_by(Prediction.created_at.desc())
        .all()
    )
    return render_template("history.html", records=records)


# ── DB INIT ───────────────────────────────────────────────────────────────────

with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)