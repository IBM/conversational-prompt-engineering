import os

import pandas as pd
import re
from collections import Counter

# chat_output_path = '/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Orith_BAM/07-07-2024 13:10:27'
# chat_output_path = '/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Artem_BAM/09-07-2024 15:14:31'
chat_output_path = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/oritht/14-07-2024 12:36:46"

target_model = 'llama-3'
offline_eval_results = f'llm_judge/{target_model}/test_full.offline.llm_judge_evaluation.csv'
eval_llm_file = os.path.join(chat_output_path, offline_eval_results)

manual_eval_results = 'eval/eval_results.csv'
eval_manual_file = os.path.join(chat_output_path, manual_eval_results)


def print_normalized_counts(type_name, counts):
    c_norm = {k: v / sum(counts.values()) for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)}
    print(type_name, c_norm, sum(counts.values()))


print(eval_llm_file)
df = pd.read_csv(eval_llm_file)
print("Num samples", len(df))
total_counts = {}

print('\nAbsolute scores (mean, var):')
for col in df.columns:
    if 'llm_judge_abs_result' in col:
        print(col, f'{df[col].mean():.2f}', f'{df[col].var():.2f}')

print('\nRelative scores:')
for col in df.columns:
    if 'llm_judge_rel_result' in col:
        counter = Counter(df[col])
        print(col, dict(counter), 'num_samples:', sum(counter.values()))
        prompt_types = "-".join(sorted(re.findall(r'\<(.*?)\>', col)))
        if prompt_types not in total_counts:
            total_counts[prompt_types] = Counter()
        total_counts[prompt_types] += counter

print('\nTotal relative normalized counts')
overall_counts = Counter()
for pt, c in total_counts.items():
    print_normalized_counts(pt, c)
    overall_counts += c

print_normalized_counts("Overall best prompt", overall_counts)

print('\nCompare with manual evaluation ')
print(eval_manual_file)
df_manual = pd.read_csv(eval_manual_file)
print("Num samples", len(df_manual))
col = "ranked_prompt_('dim1', 'Best')"
print_normalized_counts(col, Counter(df_manual[col]))
col = "ranked_prompt_('dim1', 'Worst')"
print_normalized_counts(col, Counter(df_manual[col]))





