import os
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
from transformers import pipeline

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
DB_NAME = "plants_history.db"

# تهيئة قاعدة بيانات سجل النباتات
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plant_name TEXT NOT NULL,
            disease_name TEXT NOT NULL,
            status TEXT NOT NULL,
            is_healthy INTEGER NOT NULL,
            confidence REAL NOT NULL,
            advice TEXT NOT NULL,
            image_path TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# تحميل نموذج الرؤية الحاسوبية الحديث (Vision Transformer)
print("Loading AI Plant Disease Model...")
classifier = pipeline(
    "image-classification",
    model="wambugu71/crop_leaf_diseases_vit"
)
print("Model loaded successfully!")

def get_advice(label):
    label_lower = label.lower()
    if "healthy" in label_lower:
        return {
            "status": "سليم (Healthy)",
            "is_healthy": 1,
            "advice": "النبات بصحة جيدة ولا تظهر عليه علامات مرضية. استمر في الري المنتظم والتسميد المتوازن."
        }
    elif "blight" in label_lower:
        return {
            "status": "مصاب باللفحة (Blight Disease)",
            "is_healthy": 0,
            "advice": "يُنصح بإزالة الأوراق المصابة فوراً، وتجنب الري العلوي المباشر للأوراق، واستخدام مبيد فطري نحاسي مناسب."
        }
    elif "spot" in label_lower:
        return {
            "status": "مصاب بالتبقع (Leaf Spot)",
            "is_healthy": 0,
            "advice": "تأكد من تهوية النباتات جيداً، قلل الرطوبة حول الأوراق، واستخدم العلاج الفطري أو البكتيري المخصص."
        }
    elif "rust" in label_lower or "mildew" in label_lower:
        return {
            "status": "إصابة فطرية (Rust / Mildew)",
            "is_healthy": 0,
            "advice": "قم بعزل النبات المصاب إن أمكن، واستخدم مبيداً فطرياً كبريتياً مع تحسين تعرض النبات لأشعة الشمس."
        }
    else:
        return {
            "status": "تم اكتشاف حالة مرضية (Disease Detected)",
            "is_healthy": 0,
            "advice": "تظهر علامات إجهاد أو إصابة على الورقة. يُفضل فحص بقية أجزاء النبات واستشارة مختص زراعي."
        }

@app.route("/", methods=["GET", "POST"])
def index():
    latest_result = None

    if request.method == "POST":
        file = request.files.get("leaf_image")
        plant_name = request.form.get("plant_name", "").strip()
        if not plant_name:
            plant_name = "نبات غير مسمى"

        if file and file.filename != "":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
            filename = timestamp + secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            image_path = filepath.replace("\\", "/")

            # تحليل الصورة بالذكاء الاصطناعي
            img = Image.open(filepath).convert("RGB")
            predictions = classifier(img)
            top_pred = predictions[0]

            raw_label = top_pred["label"].replace("___", " - ").replace("_", " ")
            confidence = round(top_pred["score"] * 100, 2)
            details = get_advice(top_pred["label"])
            now_str = datetime.now().strftime("%Y-%m-%d | %I:%M %p")

            # حفظ النتيجة في سجل النباتات
            conn = sqlite3.connect(DB_NAME)
            c = conn.cursor()
            c.execute("""
                INSERT INTO records (plant_name, disease_name, status, is_healthy, confidence, advice, image_path, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (plant_name, raw_label, details["status"], details["is_healthy"], confidence, details["advice"], image_path, now_str))
            conn.commit()
            conn.close()

            latest_result = {
                "plant_name": plant_name,
                "disease_name": raw_label,
                "confidence": confidence,
                "status": details["status"],
                "is_healthy": details["is_healthy"],
                "advice": details["advice"],
                "image_path": image_path
            }

    # جلب سجل جميع النباتات المفحوصة مرتبة من الأحدث للأقدم
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM records ORDER BY id DESC")
    history = c.fetchall()
    conn.close()

    return render_template("index.html", result=latest_result, history=history)

@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("DELETE FROM records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return redirect(url_for("index"))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)