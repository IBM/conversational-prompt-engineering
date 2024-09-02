# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

import pandas as pd

# dataset_name = 'legal_plain_english'
# df_full = pd.read_csv(f"/Users/oritht/Projects/language-model-utilization/data/{dataset_name}/test.csv")

# dataset_name = 'tldr'
# df_full = pd.read_csv(f"/Users/oritht/Projects/language-model-utilization/data/{dataset_name}_under_650_tokens/test.csv")

dataset_name = 'multiwoz'
df_full = pd.read_csv(f"/Users/oritht/Projects/language-model-utilization/data/{dataset_name}_unlabeled/test.csv")

df_train_texts = pd.read_csv(f"{dataset_name}/train.csv")['text'].tolist()
texts_full = df_full['text'].tolist()
filtered_rows = []
for i, t in enumerate(texts_full):
    if t in df_train_texts:
        filtered_rows.append(i)

print(f'dataset {dataset_name}')
print(f'num train {len(df_train_texts)}')
print(f'num samples full {len(df_full)}')
print(f'num samples filtered {len(filtered_rows)}')
df_full.drop(filtered_rows, inplace=True)
print(f'num samples after filtering {len(df_full)}')

df_full.to_csv(f"{dataset_name}/test_full.csv", index=False)


