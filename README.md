<p align="center">
   &emsp;
   <a href="https://www.label-sleuth.org/docs/installation.html">Quick Start</a>
   &emsp; | &emsp;
   <a href="https://www.label-sleuth.org/docs/index.html">Documentation</a>
   &emsp; | &emsp; 
   <a href="https://join.slack.com/t/labelsleuth/shared_invite/zt-1j5tpz1jl-W~UaNEKmK0RtzK~lI3Wkxg">Join&nbsp;Slack</a>
   &emsp;
</p>

# CPE: Conversational Prompt Engineering

![license](https://img.shields.io/github/license/label-sleuth/label-sleuth)  ![python](https://img.shields.io/badge/python-3.8%20--%203.9-blue) 

Describe CPE

**Table of contents**

[Installation](#installation)

[System configuration](#system-configuration)


[Reference](#reference)


## Installation
The system requires Python 3.8 or 3.9 (other versions are currently not supported and may cause issues).
1. Clone the repository:

   `git clone git@github.com:IBM/conversational-prompt-engineering.git`
2. cd to the cloned directory: `cd conversational-prompt-engineering`
3. Install the project dependencies using `conda` :
    1. Install Anaconda https://docs.anaconda.com/anaconda/install/index.html
    2. Restart your console
    3. Use the following commands to create a new anaconda environment and install the requirements:
    ```bash
    # Create and activate a virtual environment:
    conda create --yes -n cpe-env python=3.9
    conda activate cpe-env
    # Install requirements
    pip install -r requirements.txt
    ```


4. Start the CPE: run `streamlit run cpe_ui.py`. CPE will be available at http://localhost:PORT, where PORT is the port assigned by Streamlit (usually port from 8501 to 8505). If you wish to specify the port you want to use, run `streamlit run cpe_ui.py --port YOUR_PORT` where YOUR_PORT is the port you wish to use.
5. Different system properties can be set up in configurations files. The default configuration used by the system is main. If you want to use different configuration, run `streamlit run cpe_ui.py CONFIG`, where CONFIG is the name of the desired configuration. Make sure you add a file named CONFIG_config.conf to `conversational_prompt_engineering/configs`
6. Currently, the system supports WatsonX or BAM LLM clients. To use BAM you need an api key, and to use WatsonX you need an api key and a project ID. You can provide your credentials in one of three ways:
   1. Enter your credentials in the UI.
   2. Add environment arguments to you execution. Use BAM_APIKEY for BAM api key, and WATSONX_APIKEY and PROJECT_ID for WatsonX credentials.
   3. Add .env file with your credentials (see the file .env.example for example)

7. By default all project files are written to `<home_directory>/_out`, to change the directory, please update the property OUTPUT_DIR in your config file.

   

### System configuration
The configurable parameters of the system are specified config files that are placed in `conversational_prompt_engineering/configs`.

**Configurable parameters:**

| Section    | Parameter                  | Description                                                                                                                                                                                                                                                                                                                                                                   |
|------------|----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| General    | `llm_api`                  | The list of supported LLM clients. Currently we only support [WatsonXClient](https://github.com/IBM/conversational-prompt-engineering/conversational_prompt_engineering/backend/util/llm_clients/watsonx_client.py#L9) and [BamClient](https://github.com/IBM/conversational-prompt-engineering/conversational_prompt_engineering/backend/util/llm_clients/bam_client.py#L20) |
| General    | `output_dir`               | The output repository where all output files and logs are stored.                                                                                                                                                                                                                                                                                                             |
| UI         | `background_color`         | The background color of the UI.                                                                                                                                                                                                                                                                                                                                               |
| UI         | `ds_script`                | The scripts the load the list of supported dataset in the datasets droplist in the UI.                                                                                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                             |
| Evaluation | `prompt_types`             | The list of prompts that are compared in the evaluation tab. The options are: "baseline", "zero_shot" and "few_shot".                                                                                                                                                                                                                                                         |
| Evaluation | `min_examples_to_evaluate` | In the evaluation tab, the minimal number of examples the user needs to annotate before submitting threir annotations.                                                                                                                                                                                                                                                        |


## Reference
Liat Ein-Dor, Orith Toledo-Ronen, Artem Spector, Shai Gretz, Lena Dankin, Alon Halfon, Yoav Katz, Noam Slonim. [Conversational Prompt Engineering](https://arxiv.org/abs/2408.04560).

Please cite:
```
@misc{eindor2024conversationalpromptengineering,
      title={Conversational Prompt Engineering}, 
      author={Liat Ein-Dor and Orith Toledo-Ronen and Artem Spector and Shai Gretz and Lena Dankin and Alon Halfon and Yoav Katz and Noam Slonim},
      year={2024},
      eprint={2408.04560},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.04560}, 
}
```


# conversational-prompt-engineering

### Setting up the environment:
* Create Conda environment
```conda create --yes -n cpe python=3.10```
* Activate the environment
```conda activate cpe```
* Install the requirements
```pip install -r requirements.txt```

### To start the GUI, use the following *command:
```streamlit run cpe.py```

### How to debug/run in PyCharm
In the run configuration, replace the `script path` with a path to streamlit (it should be in the same directory as `python`)
In the `Parameters` field, type `run cpe.py`. Make sure your `Working directory` is the directory of the project.

#### * To avoid entering BAM's API key on every run, set the BAM_APIKEY environment variable.
