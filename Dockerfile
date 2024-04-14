FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /
EXPOSE 8000
WORKDIR /
ENV PYTHONPATH /
CMD streamlit run conversational_prompt_engineering/cpe.py --server.port 8000

