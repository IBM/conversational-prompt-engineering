import pandas as pd
from io import BytesIO
import chardet


def read_user_csv_file(uploaded_file):
    if isinstance(uploaded_file, str):  # our data is correctly formatted
        return pd.read_csv(uploaded_file)
    if uploaded_file and 'csv' in uploaded_file.type:
        bytes_data = uploaded_file.read()
        assert(len(bytes_data) == uploaded_file.size)
        encoding = chardet.detect(bytes_data)['encoding']
        return pd.read_csv(BytesIO(bytes_data), encoding=encoding)
    elif uploaded_file and 'sheet' in uploaded_file.type:
        return pd.read_excel(uploaded_file, engine="calamine") #load xsls


