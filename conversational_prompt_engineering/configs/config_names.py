import configparser

config_name_to_file = {"main": "configs/main_config.conf"}

def load_config(config_name):
    if config_name not in config_name_to_file:
        raise ValueError(f"Provided config name is: {config_name} is invalid!")

    config_file_name = config_name_to_file.get(config_name)
    config = configparser.ConfigParser()
    config.read(config_file_name)
    return config