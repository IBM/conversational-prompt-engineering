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