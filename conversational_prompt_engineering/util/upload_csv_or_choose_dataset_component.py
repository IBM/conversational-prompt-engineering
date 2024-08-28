# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from enum import Enum

from conversational_prompt_engineering.util.csv_file_utils import read_user_csv_file


def add_download_button(st, split_name):
    # this button should be present during all the session.
    selected_file_dir = st.session_state["dataset_name_to_dir"].get(st.session_state["selected_dataset"])[split_name]
    with open(selected_file_dir, 'rb') as f:
        st.download_button(f'Download data', f, file_name=f"{selected_file_dir}_{split_name}.csv", )


def rander_component(st, default_value_for_droplist, split_name):
    upload_your_csv = "upload your csv"
    dataset_name_to_dir = st.session_state["dataset_name_to_dir"]
    if "selected_dataset" not in st.session_state:
        col1, col2, col3 = st.columns(3)
    else:
        col1, col2 = st.columns(2)
        col3 = None
    with col1:
        list_of_datasets = list(dataset_name_to_dir.keys()) + [upload_your_csv]
        if selected_dataset := st.selectbox('Please select a dataset',
                                            (list_of_datasets),
                                            index=default_value_for_droplist):
            if selected_dataset != upload_your_csv:
                selected_file_dir = dataset_name_to_dir.get(selected_dataset)[split_name]
                uploaded_file = selected_file_dir  # for possible code that is conditioned on the existance of the uploaded file
                st.session_state["selected_dataset"] = selected_dataset
                st.session_state[f"csv_file_{split_name}"] = uploaded_file

    with col2:
        if selected_dataset == upload_your_csv:
            if uploaded_file := st.file_uploader(
                    "Upload text examples csv (comma separated) file.\n\n Your file should contain a column named \"text\""):
                st.session_state[f"csv_file_{split_name}"] = uploaded_file

    if col3 is not None:
        with col3:
            if not st.session_state['existing_chat_loaded']:
                with st.popover("load existing chat (debug)"):
                    st.markdown("Local path to an existing chat 👋")
                    st.session_state["existing_chat_path"] = st.text_input("path")
            else:
                st.session_state["existing_chat_path"] = ""
    if "selected_dataset" in st.session_state and st.session_state["selected_dataset"] != upload_your_csv:
        st.code(dataset_name_to_dir.get(selected_dataset)['desc'], language="markdown")


class StartType(Enum):
    No = 1
    Uploaded = 2
    Loaded = 3


def create_choose_dataset_component_train(st, manager):
    start_type = StartType.No
    if manager.enable_upload_file:
        rander_component(st, default_value_for_droplist=None, split_name='train')
        if "csv_file_train" in st.session_state and st.session_state["csv_file_train"] != None:
            start_type = StartType.Uploaded
        if ("existing_chat_path" in st.session_state and st.session_state["existing_chat_path"] != ""):
            start_type = StartType.Loaded
            # manager.process_examples(read_user_csv_file(st.session_state["csv_file_train"]), st.session_state["selected_dataset"] if "selected_dataset" in st.session_state else "user")
    # if "selected_dataset" in st.session_state:
    #     add_download_button(st, 'train')
    #     st.write(f"Using {st.session_state['selected_dataset']} dataset")
    return start_type


def create_choose_dataset_component_eval(st):
    datasets = list(st.session_state["dataset_name_to_dir"].keys())
    selected_index = None
    if "selected_dataset" in st.session_state:
        selected_index = datasets.index(st.session_state["selected_dataset"])
    rander_component(st, default_value_for_droplist=selected_index, split_name='eval')
    if "selected_dataset" in st.session_state:
        add_download_button(st, 'eval')
    if "csv_file_eval" in st.session_state:
        return read_user_csv_file(st.session_state["csv_file_eval"]).text.tolist()[:10]

