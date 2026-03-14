FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 443 80

ENV CERT_DIR=/etc/letsencrypt/live/default
CMD ["sh", "-c", \
  "hypercorn api:app \
    --bind 0.0.0.0:443 \
    --certfile ${CERT_DIR}/fullchain.pem \
    --keyfile ${CERT_DIR}/privkey.pem \
    --insecure-bind 0.0.0.0:80"]
