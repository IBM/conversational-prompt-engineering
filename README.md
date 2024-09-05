<p align="center">
   <a href="https://join.slack.com/t/conversationp-0js4284/shared_invite/zt-2q377kmi9-00FL2CZXtaR0TFkg2NwGdw">Join&nbsp;Slack</a>
   &emsp;
</p>

# CPE: Conversational Prompt Engineering

![license](https://img.shields.io/badge/License-Apache_2.0-blue.svg)  ![python](https://img.shields.io/badge/python-3.10%20-green) 

CPE is a prompt building assistant developed in IBM. This service is intended to help users build an effective prompt, tailored to their specific use case, through a simple chat with an LLM. 

This is an overview of the system:

![cpe_overview](./images/cpe_overview.png)

**Table of contents**

[Installation](#installation)

[System configuration](#system-configuration)

[Using the system](#using-the-system)

[Reference](#reference)


## Installation
The system requires Python 3.10 (other versions are currently not supported and may cause issues).
1. Clone the repository:

   `git clone git@github.com:IBM/conversational-prompt-engineering.git`
2. cd to the cloned directory: `cd conversational-prompt-engineering`
3. Install the project dependencies :
    1. Install an environment management tool such as [miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install)
    2. Restart your console
    3. Use the following commands to create a new anaconda environment and install the requirements:
    ```bash
    # Create and activate a virtual environment:
    conda create --yes -n cpe-env python=3.10
    conda activate cpe-env
    # Install requirements
    pip install -r requirements.txt
    ```


4. Start the CPE: run `streamlit run cpe_ui.py`. CPE will be available at http://localhost:PORT, where PORT is the port assigned by Streamlit (usually between 8501 to 8505). If you wish to specify the port you want to use, run `streamlit run cpe_ui.py --port YOUR_PORT` where YOUR_PORT is the port you wish to use.
5. Different system properties can be set up in configurations files. The default configuration used by the system is `main` (config file is [here](https://github.com/IBM/conversational-prompt-engineering/blob/main/conversational_prompt_engineering/configs/main_config.conf). If you want to use different configuration, run `streamlit run cpe_ui.py CONFIG`, where CONFIG is the name of the desired configuration. Make sure you add a file named CONFIG_config.conf to `conversational_prompt_engineering/configs`
6. Currently, the system only supports WatsonX LLM api. If you wish to intergrate another LLM api, you can easily do so by adding a new class that inherits from [AbstLLMClient](https://github.com/IBM/conversational-prompt-engineering/blob/main/conversational_prompt_engineering/backend/util/llm_clients/abst_llm_client.py#L17). To use WatsonX, you need to provide you api key and project id in one of three ways:
   1. Enter your credentials in the UI.
   2. Add environment arguments to you execution environment. The aguments are: WATSONX_APIKEY and PROJECT_ID.
   3. Add .env file with your credentials (see the file .env.example for example)

7. By default all project files are written to `<home_directory>/_out`, to change the directory, please update the property OUTPUT_DIR in your config file.

   

### System configuration
The configurable parameters of the system are specified config files that are placed in `conversational_prompt_engineering/configs`.

**Configurable parameters:**

| Section    | Parameter                  | Description                                                                                                                                                                                                                                                                          |
|------------|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| General    | `llm_api`                  | The list of supported LLM clients. Currently we only support [WatsonXClient](https://github.com/IBM/conversational-prompt-engineering/blob/main/conversational_prompt_engineering/backend/util/llm_clients/watsonx_client.py#L9)                                                               |
| General    | `output_dir`               | The output repository where all output files and logs are stored.                                                                                                                                                                                                                    |
| UI         | `background_color`         | The background color of the UI.                                                                                                                                                                                                                                                      |
| UI         | `ds_script`                | The scripts the load the list of supported dataset in the datasets droplist in the UI.                                                                                                                                                                                               |                                                                                                                                                                                                                                                             |
| Evaluation | `prompt_types`             | The list of prompts that are compared in the evaluation tab. The options are: `baseline`, `zero_shot` and `few_shot`. `baseline` is generated by the LLM after the user briefly explain their task. `zero_shot` and `few_shot` prompts are generated at the end of the conversation. |
| Evaluation | `min_examples_to_evaluate` | In the evaluation tab, the minimal number of examples the user needs to annotate before submitting threir annotations.                                                                                                                                                               |


### Using the system 

To use and evaluate CPE, proceed according to the following steps: 
1. When you run the system, you need to select your `target model`, which is the model you'll be creating prompts for. Currently we support 3 target models:   `llama-3-70b-instruct`, `mixtral-8x7b-instruct-v01` and 
2. Start by generating a cvs file with 3 examples of texts most related to your daily work, and upload it. Your csv file should contain a single column titled "text". Each row is a single input example to CPE. If your input examples should contain multiple sentences/paragraphs, make sure they fit a single row. Alternatively, you can select a dataset from our catalog.
3. Follow the chat with the system. These are the phases of the chat:
![cpe_chat_flow](./images/cpe_flow.png)
   1. Once the system notifies you that the final prompt is ready, you will be presented with the outputs generated by the prompts. 
   2. For each output, you can present your comments and or modifications. The system will generate a new prompt, if needed, initialize another round of outputs confirmation.
   3. Once the prompt is ready, you can download two versions: 
      1. `zero shot prompt`: It includes only the instructions generated during the chat. 
      2. `few shot prompt`: It includes the instructions and also the three text examples and approved outputs. 
   4. After you generate a prompt, you can proceed to evaluation in the Evaluation tab. 
      1. You will be presented with the compared prompts (according to your chosen configuration). You begin with the upload of an examples csv files (similar format to the input csv file you used in the chat phase).
      2. When you click on the "generate outputs" button, the system generates all the output for the compared prompts over the examples in your csv file. 
      3. You will be presented, for each example, the generated output. These outputs are shuffled and source prompt is not disclosed to you. For each example, select the best and worst output.
      4. Once you reach the required minimum of annotated examples, you can submit your annotation and get the analysis per each prompt.

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

