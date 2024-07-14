import pandas as pd

df_full = pd.read_csv("/Users/oritht/Projects/language-model-utilization/data/legal_plain_english/test.csv")
df_train_texts = pd.read_csv("legal_plain_english/train.csv")['text'].tolist()
df_test_texts = pd.read_csv("legal_plain_english/eval.csv")['text'].tolist()

texts_full = df_full['text'].tolist()
filtered_rows = []
for i, t in enumerate(texts_full):
    if t in df_train_texts or t in df_test_texts:
        filtered_rows.append(i)

print('legal_plain_english')
print(f'num train {len(df_train_texts)}, num test {len(df_test_texts)}')
print(f'num samples full {len(df_full)}')
print(f'num samples filtered {len(filtered_rows)}')
df_full.drop(filtered_rows, inplace=True)
print(f'num samples after filtering {len(df_full)}')
df_full.to_csv("legal_plain_english/test_full.csv", index=False)


df_full = pd.read_csv("/Users/oritht/Projects/language-model-utilization/data/tldr_under_650_tokens/test.csv")
df_train_texts = pd.read_csv("tldr/train.csv")['text'].tolist()
df_test_texts = pd.read_csv("tldr/test.csv")['text'].tolist()

texts_full = df_full['text'].tolist()
filtered_rows = []
for i, t in enumerate(texts_full):
    if t in df_train_texts or t in df_test_texts:
        filtered_rows.append(i)

print('tldr')
print(f'num train {len(df_train_texts)}, num test {len(df_test_texts)}')
print(f'num samples full {len(df_full)}')
print(f'num samples filtered {len(filtered_rows)}')
df_full.drop(filtered_rows, inplace=True)
print(f'num samples after filtering {len(df_full)}')
df_full.to_csv("tldr/test_full.csv", index=False)