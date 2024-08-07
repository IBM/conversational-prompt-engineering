import json

import numpy as np
import pandas as pd
import os
from scipy.stats import ttest_ind

from nltk.metrics.distance import edit_distance

# survey
survey_files = []
chat_dir = '_out/paper_dirs'
for subdir, dirs, files in os.walk(chat_dir):
    if subdir.endswith('survey'):
        survey_files.append(os.path.join(subdir, 'survey.csv'))
print(f"number of surveys: {len(survey_files)}")
df = pd.concat([pd.read_csv(survey_file) for survey_file in survey_files])
df = df.rename(columns={"q_0": "Satisfaction from baseline prompt", "q_1": "Satisfaction from CPE prompt",
                        "q_2": "Gain from thinking process", "q_3": "Chat pleasantness", "q_4": "Convergence time"})
print(df.describe())
out_dir = os.path.join("_out", "paper_chat_analysis")
os.makedirs(out_dir, exist_ok=True)
df[[c for c in df.columns if c != 'q_5']].mean().to_csv(os.path.join(out_dir, "survey_results.csv"))
pval = ttest_ind(a=df['Satisfaction from baseline prompt'].tolist(), b=df['Satisfaction from CPE prompt'].tolist(), equal_var=False).pvalue
print(f"Pvalue between baseline and CPE prompt: {pval}")

# avg. number of turns
chat_files = []
for subdir, dirs, files in os.walk(chat_dir):
    if subdir.endswith('chat'):
        chat_files.append(os.path.join(subdir, 'user_chat.csv'))
print(f"number of chats: {len(chat_files)}")
number_of_turns = [len(pd.read_csv(uc)['role'].tolist()) for uc in chat_files]
print(f"avg. number of turns: {np.mean(number_of_turns)}, min number of turns: {np.min(number_of_turns)}, max number of turns {np.max(number_of_turns)}, std dev: {np.std(number_of_turns)}")

# avg. number of prompts
prompts = []
chat_results_files = []
chat_dir = '_out/paper_dirs'
for subdir, dirs, files in os.walk(chat_dir):
    if len(files) > 0:
        if os.path.exists(os.path.join(subdir, 'chat_result.json')):
            print(subdir)
            chat_results_files.append(os.path.join(subdir, 'chat_result.json'))
for crf in chat_results_files:
    with open(crf, 'r') as f:
        data = json.load(f)
        prompts.append(data['prompts'])
number_of_prompts = [len(p) for p in prompts]
print(f"number_of_prompts: {number_of_prompts}")
print(f"avg. number of prompts: {np.mean(number_of_prompts)}, std dev: {np.std(number_of_prompts)}")

eds = []
for p_list in prompts:
    ed = edit_distance(p_list[0], p_list[-1])
    eds.append(ed)
print(f"edit distance: {eds}")
print(f"avg. edit distance: {np.mean(eds)}, min: {np.min(eds)}, max: {np.max(eds)}, std dev: {np.std(eds)}")

prompt_data = {'prompt_1': [], 'prompt_2': [], 'prompt_3': [], 'prompt_4': []}
for p_list in prompts:
    # print("*******************************************")
    # print("\nvs. \n".join(p_list))
    if len(p_list) < 3:
        p_list.append('')
        p_list.append('')
    if len(p_list) < 4:
        p_list.append('')
    for i, p in enumerate(p_list):
        prompt_data[f'prompt_{i+1}'].append(p)
pd.DataFrame(prompt_data).to_csv(os.path.join(out_dir, 'prompts.csv'))
