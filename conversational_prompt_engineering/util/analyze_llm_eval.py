import os

import pandas as pd
import re
from collections import Counter
from scipy.stats import chisquare
import numpy as np


summary_prompt_types = ['baseline', 'zero_shot', 'few_shot']
DO_NOT_EVALUATE_FEW_SHOT = True

actual_summary_prompt_types = summary_prompt_types
if DO_NOT_EVALUATE_FEW_SHOT:
    actual_summary_prompt_types = ['baseline', 'zero_shot']
zero_counter = Counter({pt: 0 for pt in actual_summary_prompt_types})


def get_normalized_counts(type_name, counts, with_print=True):
    c_norm = {k: v / sum(counts.values()) for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)}
    if with_print:
        print(type_name, counts, "Normalized:", c_norm, sum(counts.values()))
    return c_norm


def llm_evaluation_stats(df):
    print('\nAbsolute scores (mean, var):')
    for col in df.columns:
        if DO_NOT_EVALUATE_FEW_SHOT and "few_shot" in col:
            continue
        if 'llm_judge_abs_result' in col:
            print(col, f'{df[col].mean():.2f}', f'{df[col].var():.2f}', len(df[col]), df[col].tolist())

    print('\nRelative scores:')
    total_counts = {}
    for col in df.columns:
        if DO_NOT_EVALUATE_FEW_SHOT and "few_shot" in col:
            continue
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
        c.subtract(zero_counter)
        print(pt, "Chi-square:", chisquare(list(c.values())).pvalue, sum(c.values()))
        overall_counts += c
        c_norm = get_normalized_counts(pt, c)
    c_norm = get_normalized_counts("\nOverall best prompt", overall_counts)
    overall_counts.subtract(zero_counter)
    chisq = chisquare(list(overall_counts.values())).pvalue
    num = sum(overall_counts.values())
    print("Overall", "Chi-square:", chisq, num)
    return (chisq, num)


def get_manual_selection(row):
    #print("Best:", row["ranked_prompt_('dim1', 'Best')"], "Worst:", row["ranked_prompt_('dim1', 'Worst')"])
    manual_best = row["ranked_prompt_('dim1', 'Best')"]
    manual_worst = row["ranked_prompt_('dim1', 'Worst')"]
    for sp in summary_prompt_types:
        if sp != manual_best and sp != manual_worst:
            manual_middle = sp
            break
    #print("Manual Best:", manual_best)
    #print("Manual Middle:", manual_middle)
    #print("Manual Worst:", manual_worst)
    if DO_NOT_EVALUATE_FEW_SHOT:
        if "few_shot" in manual_best:
            manual_best = manual_middle
        if "few_shot" in manual_worst:
            manual_worst = manual_middle
    #print("--> Manual Best:", manual_best)
    #print("--> Manual Middle:", manual_middle)
    #print("--> Manual Worst:", manual_worst)
    return manual_best, manual_worst


def compute_agreement(df):
    print("Manual bset:\t", df["Manual_ranked_prompt_best"].tolist())
    print("LLM best:   \t", df["Best_llm_judge_rel"].tolist())
    print("LLM best score:\t", df["Best_llm_judge_rel_score"].tolist())
    agreement = [1 if l == m else 0 for m, l in
                 zip(df["Manual_ranked_prompt_best"], df["Best_llm_judge_rel"])]
    df["Agreement"] = agreement
    #agreement = [a for a, s in zip(agreement, df["Best_llm_judge_rel_score"]) if s > 0.5]
    #agreement_avg = sum(agreement) / len(agreement)
    agreement_avg = np.dot(df["Agreement"], df["Best_llm_judge_rel_score"])/len(df)
    num_decisions = sum([1 if s > 0.5 else 0 for s in df["Best_llm_judge_rel_score"]])
    res = {"Weighted Agreement": agreement_avg, "Num": len(agreement), "Num decisions": num_decisions}
    print(res)
    return res


def save_evaluation(df, eval_chat_file):
    if DO_NOT_EVALUATE_FEW_SHOT:
        eval_type = ""
    else:
        eval_type = "_with_few_shot"
    out_csv = eval_chat_file.replace('.csv', f'_analysis{eval_type}.csv')
    print('Analysis output file:', out_csv)
    df.to_csv(out_csv)


def analyze_llm_evaluation(df):
    llm_best_prompt = []
    llm_best_prompt_score = []
    llm_chisquare = []
    for row in df.to_dict(orient='records'):
        llm_selected_prompt = []
        for k in row.keys():
            if "_llm_judge_rel_result" in k:
                if DO_NOT_EVALUATE_FEW_SHOT and "few_shot" in k:
                    continue
                llm_selected_prompt.append(row[k])
        counts = Counter(llm_selected_prompt)
        counts.subtract(zero_counter)
        norm_counts = get_normalized_counts('Overall', counts, with_print=False)
        max_key = max(norm_counts, key=lambda key: norm_counts[key])
        # print(max_key, len(norm_counts), norm_counts[max_key])
        llm_best_prompt.append(max_key)
        llm_best_prompt_score.append(norm_counts[max_key])
        llm_chisquare.append(chisquare(list(counts.values())).pvalue)

    df['Best_llm_judge_rel'] = llm_best_prompt
    df['Best_llm_judge_rel_score'] = llm_best_prompt_score
    df['Llm_chisquare'] = llm_chisquare
    return df


def analyze_manual_evaluation(df):
    df_dict = df.to_dict(orient='records')
    for row in df_dict:
        manual_best, manual_worst = get_manual_selection(row)
        print("Best Worst:", manual_best, manual_worst)
        row["Manual_ranked_prompt_best"] = manual_best
        row["Manual_ranked_prompt_worst"] = manual_worst
    df_res = pd.DataFrame.from_dict(df_dict)
    col = "Manual_ranked_prompt_best"
    counts_best = Counter(df_res[col])
    counts_best.subtract(zero_counter)
    get_normalized_counts(col, counts_best)
    chisq_best = chisquare(list(counts_best.values())).pvalue
    num_best = sum(counts_best.values())
    print(col, "Chi-square:", chisq_best, num_best)
    col = "Manual_ranked_prompt_worst"
    counts_worst = Counter(df_res[col])
    counts_worst.subtract(zero_counter)
    get_normalized_counts(col, counts_worst)
    chisq_worst = chisquare(list(counts_worst.values())).pvalue
    num_worst = sum(counts_worst.values())
    print(col, "Chi-square:", chisq_worst, num_worst)
    return df_res, chisq_best, chisq_worst, num_best, num_worst


def evaluate_offline(test_split):
    offline_eval_results = f'llm_judge/{target_model}/{test_split}.offline.llm_judge_evaluation.csv'
    eval_llm_file = os.path.join(chat_output_path, offline_eval_results)
    if not os.path.isfile(eval_llm_file):
        print(f'Skip {offline_eval_results}, file does not exist')
        return
    # Offline eval
    print(f"\n======= Offline Evaluation {eval_llm_file}")
    df_llm_offline = pd.read_csv(eval_llm_file)
    print("Num samples", len(df_llm_offline))
    chisq, n = llm_evaluation_stats(df_llm_offline)
    print(f"\n==================================")
    eval_res = {"pvalue": chisq, "num": n}
    return eval_res


def evaluate_chat():
    llm_eval_results = f'llm_judge/{target_model}/eval_results.chat.llm_judge_evaluation.csv'
    eval_chat_llm_file = os.path.join(chat_output_path, llm_eval_results)
    if not os.path.isfile(eval_chat_llm_file):
        print(f'Skip {llm_eval_results}, file does not exist in {chat_output_path}')
        return
    # Chat eval
    print(f"\n====== Chat Evaluation {eval_chat_llm_file}")
    df_chat = pd.read_csv(eval_chat_llm_file).dropna()
    print("Num samples", len(df_chat))
    llm_evaluation_stats(df_chat)

    print(f"\n====== Manual Best Worst counts")
    df_chat, chisq_best, chisq_worst, num_best, num_worst = analyze_manual_evaluation(df_chat)
    print(f"\n====== LLM Best counts")
    df_chat = analyze_llm_evaluation(df_chat)
    print(f"\n====== Manual and LLM Best agreement")
    agreement = compute_agreement(df_chat)
    eval_res = {"agreement":agreement, "best_pvalue": chisq_best, "worst_pvalue": chisq_worst, "num_best": num_best, "num_worst": num_worst}
    save_evaluation(df_chat, eval_chat_llm_file)
    return eval_res


if __name__ == "__main__":
    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out"

    chats_list = [
        "oritht/14-07-2024 12:36:46",
        "liat/21-07-2024 12:16:37",
        "shai/21-07-2024 12:36:52",
    ]

    chats_list = [
        "shai/wiki_animals",
    ]

    target_model = 'llama-3'
    offline_test_splits = ["eval", "test", "test_full"]

    print("Don't evaluate few-shot summary:", DO_NOT_EVALUATE_FEW_SHOT)

    offline_res = {}
    manual_res = {}
    for chat_dir in chats_list:
        chat_output_path = os.path.join(chats_output_dir, chat_dir)
        print(f"Evaluating {chat_dir}")
        manual_res.update({chat_dir:{}})
        eval_result = evaluate_chat()
        manual_res[chat_dir].update(eval_result)
        offline_res.update({chat_dir: {}})
        for split in offline_test_splits:
            eval_result = evaluate_offline(split)
            if eval_result is None:
                continue
            offline_res[chat_dir].update({split:eval_result})

    print("\n\nSUMMARY:")
    print("Manual Chat:", manual_res)
    print("Offline Test:", offline_res)









