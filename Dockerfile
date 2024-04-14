FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /
EXPOSE 8000
WORKDIR ./conversational_prompt_engineering
CMD streamlit run cpe.py --server.port 8000
