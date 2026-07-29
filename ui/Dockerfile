FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit==1.55.0 requests==2.31.0

COPY ui/ ui/
COPY notebook/*.png notebook/

EXPOSE 8501

CMD exec streamlit run ui/app.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true
