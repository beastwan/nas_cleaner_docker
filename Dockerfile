# 明确使用 bookworm (Debian 12) 稳定版，避免自动升级到测试版导致不兼容
FROM python:3.10-slim-bookworm

# 修改这里：使用 libgl1 替代旧的 libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# 增加国内镜像源参数，加快 Python 库下载速度（可选，如果下载慢加上 -i https://pypi.tuna.tsinghua.edu.cn/simple）
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
RUN chmod 777 /app
