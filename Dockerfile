FROM python:3.9.17-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
CMD [ "streamlit run app.py" ]