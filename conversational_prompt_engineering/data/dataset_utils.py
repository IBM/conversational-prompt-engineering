import importlib

def load_dataset_mapping(config):
    #setup datasets loading script:
    script_path_name = config.get("Dataset", "ds_script")
    spec = importlib.util.spec_from_file_location('module_name', script_path_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "dataset_name_to_dir")
