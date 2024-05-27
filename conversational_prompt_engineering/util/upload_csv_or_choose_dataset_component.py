from conversational_prompt_engineering.util.csv_file_utils import read_user_csv_file
from conversational_prompt_engineering.data.dataset_name_to_dir import dataset_name_to_dir

def add_download_button(st, split_name):
    #this button should be present during all the session.
    selected_file_dir = dataset_name_to_dir.get(st.session_state["selected_dataset"])[split_name]
    with open(selected_file_dir, 'rb') as f:
        st.download_button(f'Download data', f, file_name=f"{selected_file_dir}_{split_name}.csv", )

def rander_component(st, default_value_for_droplist, split_name):
    upload_your_csv = "upload your csv"
    col1, col2 = st.columns(2)
    with col1:
        list_of_datasets = list(dataset_name_to_dir.keys()) + [upload_your_csv]
        if selected_dataset := st.selectbox('Upload your csv or choose from our datasets catalog',
                                            (list_of_datasets),
                                            index=default_value_for_droplist):
            if selected_dataset != upload_your_csv:
                selected_file_dir = dataset_name_to_dir.get(selected_dataset)[split_name]
                uploaded_file = selected_file_dir  # for possible code that is conditioned on the existance of the uploaded file
                st.session_state["selected_dataset"] = selected_dataset
                st.session_state[f"uploaded_file_{split_name}"] = uploaded_file
                st.code(dataset_name_to_dir.get(selected_dataset)['desc'], language="markdown")

    with col2:
        if selected_dataset == upload_your_csv:
            if uploaded_file := st.file_uploader("Upload text examples csv"):
                st.session_state[f"uploaded_file_{split_name}"] = uploaded_file


def create_choose_dataset_component_train(st, manager):
    if manager.enable_upload_file:
        rander_component(st, default_value_for_droplist=None, split_name='train')
        if "uploaded_file_train" in st.session_state:
            manager.process_examples(read_user_csv_file(st.session_state["uploaded_file_train"]))


    if "selected_dataset" in st.session_state:
        add_download_button(st, 'train')


def create_choose_dataset_component_eval(st):
    datasets = list(dataset_name_to_dir.keys())
    selected_index=None
    if "selected_dataset" in st.session_state:
        selected_index = datasets.index(st.session_state["selected_dataset"])
    rander_component(st, default_value_for_droplist=selected_index, split_name='eval')
    if "selected_dataset" in st.session_state:
        add_download_button(st, 'eval')
    if "uploaded_file_eval" in st.session_state:
        return read_user_csv_file(st.session_state["uploaded_file_eval"]).text.tolist()
