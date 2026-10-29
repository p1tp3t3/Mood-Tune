FROM python:3.13-slim

# Run as a normal user instead of root, for safety
RUN useradd --create-home --uid 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    PORT=5002
WORKDIR /home/user/app

# CPU-only torch first (much smaller than the default CUDA build), then the rest
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

# Download the emotion model (~500 MB) at build time so the container starts without fetching it
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='SamLowe/roberta-base-go_emotions')"

COPY --chown=user . .

# Listens on all interfaces (0.0.0.0), so other devices on the local network can reach it once the port is published:
#   docker run -p 5002:5002 moodtunes   ->   http://<this PC's LAN IP>:5002
EXPOSE 5002

# One worker: each worker loads its own copy of the emotion model (~1 GB RAM).
# Threads handle concurrent requests; the long timeout covers model loading and album-cover fetching.
CMD gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 main:app
