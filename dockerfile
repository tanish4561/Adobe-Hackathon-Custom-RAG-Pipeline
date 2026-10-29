FROM python:3.10-slim
WORKDIR /app
COPY req.txt /app

RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r req.txt && rm -rf ~/.cache/pip

COPY Challenge_1a/ /app/
COPY entry.sh /app/
RUN chmod +x /app/entry.sh

ENTRYPOINT ["/app/entry.sh"]