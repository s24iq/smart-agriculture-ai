FROM python:3.11-slim

WORKDIR /app

# إنشاء مجلدات الكاش للنماذج بصلاحيات كتابة كاملة
ENV HF_HOME=/tmp/huggingface
ENV TORCH_HOME=/tmp/torch

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# منح صلاحيات الكتابة لقاعدة البيانات ومجلد الصور المرفوعة
RUN mkdir -p static/uploads && chmod -R 777 /app

EXPOSE 7860

CMD ["python", "app.py"]