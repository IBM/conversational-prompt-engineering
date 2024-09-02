# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import importlib

def load_dataset_mapping(config):
    #setup datasets loading script:
    script_path_name = config.get("UI", "ds_script")
    spec = importlib.util.spec_from_file_location('module_name', script_path_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "dataset_name_to_dir")
