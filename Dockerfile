# 荟诊 · 芦荟病虫害智能问答与诊断 Agent
# 适用于魔搭创空间（Docker）、Zeabur、Sealos、Railway 及任意容器平台。

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=7860

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py README.md ./
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','7860')+'/health', timeout=3)"

CMD ["python", "app.py"]
