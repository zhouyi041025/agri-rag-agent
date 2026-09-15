# 荟诊 · 芦荟病虫害智能问答与诊断 Agent
# 适用于魔搭创空间（Docker）、Zeabur、Sealos、Railway 及任意容器平台。

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    PORT=7860 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=5

WORKDIR /app

COPY requirements.txt ./

# 优先走国内镜像源，失败自动回落官方源，避免构建时拉包超时
RUN pip install --no-cache-dir -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        --trusted-host mirrors.aliyun.com \
    || pip install --no-cache-dir -r requirements.txt

COPY app.py README.md ./
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY eval/ ./eval/

EXPOSE 7860

CMD ["python", "app.py"]
