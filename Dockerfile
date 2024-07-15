FROM python:3.10-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /
EXPOSE 8000
WORKDIR /conversational_prompt_engineering
ENV PYTHONPATH /:/conversational_prompt_engineering
ARG watsonx_api_key
ARG watsonx_project_id
ARG config_name
ENV WATSONX_APIKEY=${watsonx_api_key}
ENV PROJECT_ID=${watsonx_project_id}
CMD streamlit run cpe.py --server.port 8000 ${config_name}
