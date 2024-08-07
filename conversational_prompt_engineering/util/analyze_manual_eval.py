import datetime
import json
import os

import pandas as pd
import re
from collections import Counter
from scipy.stats import chisquare, ttest_1samp
import numpy as np
from conversational_prompt_engineering.backend.llm_as_a_judge import get_all_pairs

import matplotlib.pyplot as plt


summary_prompt_types = ['baseline', 'zero_shot', 'few_shot']

PARTIAL_EVALUATION = False

actual_summary_prompt_types = summary_prompt_types
eval_ver = "_ALL"
if PARTIAL_EVALUATION:
    eval_ver = "_BS_ZS"
    actual_summary_prompt_types = ['baseline', 'zero_shot']

    #eval_ver = "_BS_FS"
    #actual_summary_prompt_types = ['baseline', 'few_shot']

    #eval_ver = "_ZS_FS"
    #actual_summary_prompt_types = ['zero_shot', 'few_shot']

    # eval_ver = "_BSFT_FT"
    # actual_summary_prompt_types = ['baseline_few_shot', 'few_shot']

    eval_type = f"_partial{len(actual_summary_prompt_types)}{eval_ver}"
else:
    eval_type = f"_full{len(actual_summary_prompt_types)}{eval_ver}"

zero_counter = Counter({pt: 0 for pt in actual_summary_prompt_types})

ttest_mappings = {"baseline-zero_shot": {"zero_shot": 0.5, "baseline": -0.5, "-1": 0},
                  "baseline-few_shot": {"few_shot": 0.5, "baseline": -0.5, "-1": 0},
                  #"zero_shot-few_shot": {"few_shot": 0.5, "zero_shot": -0.5, "-1": 0},  # ORIG_MAP
                  "zero_shot-few_shot": {"zero_shot": 0.5, "few_shot": -0.5, "-1": 0},  # NEW_MAP
                  "baseline_few_shot-few_shot": {"few_shot": 0.5, "baseline_few_shot": -0.5, "-1": 0}}
pvalue_alpha = 0.05


def get_normalized_counts(type_name, counts, with_print=True):
    c_norm = {k: v / sum(counts.values()) for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)}
    if with_print:
        print(type_name, counts, "Normalized:", c_norm, sum(counts.values()))
    return c_norm


def get_manual_selection(row):
    if sum([1 if "('dim1'" in k else 0 for k in row.keys()]):
        manual_best = row["ranked_prompt_('dim1', 'Best')"]
        manual_worst = row["ranked_prompt_('dim1', 'Worst')"]
    else:
        manual_best = row["ranked_prompt_('', 'Best')"]
        manual_worst = row["ranked_prompt_('', 'Worst')"]
    for sp in summary_prompt_types:
        if sp != manual_best and sp != manual_worst:
            manual_middle = sp
            break
    if manual_best not in actual_summary_prompt_types:
        manual_best = manual_middle
    if manual_worst not in actual_summary_prompt_types:
        manual_worst = manual_middle
    return manual_best, manual_worst, manual_middle


def save_evaluation(df, eval_chat_file):
    print('Analysis output csv file:', eval_chat_file)
    df.to_csv(eval_chat_file, index=False)


def save_evaluation_results_json(eval_res, eval_json_file):
    print('Analysis output json file:', eval_json_file)
    with open(eval_json_file, 'w') as f:
        json.dump(eval_res, f)


def run_ttest_one_sided(ttest_data, mapping, alpha=pvalue_alpha):
    val_map = dict((v, k) for k, v in mapping.items())
    ttest_res = ttest_1samp(ttest_data, popmean=0.0, alternative='greater')
    pvalue = getattr(ttest_res, 'pvalue')
    tstat = getattr(ttest_res, 'statistic')
    significant = False
    selected = ""
    if tstat > 0 and pvalue < alpha:
        significant = True
        selected = val_map[np.sign(tstat)*0.5]
    print("RUN TTEST ONE SIDED", pvalue, "STAT", tstat, significant, selected, mapping)
    return {"ttest_pvalue": pvalue, "ttest_t": str(tstat), "num_ttest": len(ttest_data), "significant": significant,
            "greater": selected, "ttest_data": ttest_data, "ttest_map": mapping}


def compute_ttest_for_pair(df, col_name):
    assert len(actual_summary_prompt_types) == 2
    p1 = actual_summary_prompt_types[0]
    p2 = actual_summary_prompt_types[1]
    ttest_map = ttest_mappings[f"{p1}-{p2}"]
    ttest_data = [ttest_map[p] for p in df[col_name]]
    ttest_res = run_ttest_one_sided(ttest_data, ttest_map)
    ttest_res.update({"col_name": col_name, "prompt_col": df[col_name].tolist(), "prompts_type": (p1, p2)})
    print(ttest_res)
    return(ttest_res)


def compute_pvalue(df, col_name):
    counts_col = Counter(df[col_name])
    counts_col.subtract(zero_counter)
    counts_col_norm = get_normalized_counts(col_name, counts_col)
    pvalue_col = chisquare(list(counts_col.values())).pvalue
    num_col = sum(counts_col.values())
    print(col_name, "Chi-sq P-value:", pvalue_col, "Num:", num_col)
    return pvalue_col, num_col, counts_col, counts_col_norm


def get_manual_analysis(df):
    pvalue_best, num_best, counts_best, counts_best_norm = compute_pvalue(df, "Manual_ranked_prompt_best")
    pvalue_middle, num_middle, counts_middle, counts_middle_norm = compute_pvalue(df, "Manual_ranked_prompt_middle")
    pvalue_worst, num_worst, counts_worst, counts_worst_norm = compute_pvalue(df, "Manual_ranked_prompt_worst")
    analysis_res = {"manual_eval": {"best_pvalue": pvalue_best, "middle_pvalue": pvalue_middle, "worst_pvalue": pvalue_worst,
                                    "num_best": num_best, "num_middle": num_middle, "num_worst": num_worst,
                                    "counts_best": counts_best, "counts_middle": counts_middle, "counts_worst": counts_worst,
                                    "norm_counts_best": counts_best_norm, "norm_counts_middle": counts_middle_norm,
                                    "norm_counts_worst": counts_worst_norm}}
    if PARTIAL_EVALUATION:
        ttest_res = compute_ttest_for_pair(df, "Manual_ranked_prompt_best")
        analysis_res["manual_eval"].update({"ttest_pair": ttest_res})
    return analysis_res


def analyze_manual_evaluation(df):
    df_dict = df.to_dict(orient='records')
    for row in df_dict:
        manual_best, manual_worst, manual_middle = get_manual_selection(row)
        # print("Best Worst:", manual_best, manual_worst)
        row["Manual_ranked_prompt_best"] = manual_best
        row["Manual_ranked_prompt_worst"] = manual_worst
        row["Manual_ranked_prompt_middle"] = manual_middle
    df_res = pd.DataFrame.from_dict(df_dict)
    analysis_res = get_manual_analysis(df_res)
    return df_res, analysis_res


def evaluate_chats(df_chats):
    print_overall_info(df_chats)
    df_chats, res_chats = analyze_manual_evaluation(df_chats)

    print("Save results")
    out_results = os.path.join(overall_analysis_dir, f"manual_eval_all_chats{eval_type}.csv")
    eval_chats_file = os.path.join(chat_output_path, out_results)
    save_evaluation(df_chats, eval_chats_file)
    out_json = os.path.join(overall_analysis_dir, f"manual_eval_all_chats{eval_type}.json")
    eval_chats_file = os.path.join(chat_output_path, out_json)
    save_evaluation_results_json(res_chats, eval_chats_file)

    print("Plot results")
    plot_chat_histograms_manual_eval(df_chats)
    if not PARTIAL_EVALUATION:
        plot_chat_histograms_manual_eval_transpose(df_chats)


def print_overall_info(df):
    print("Manual evaluation results on all chats")
    print("Num samples", len(df))
    num_chats = len(set(df['chat_dir']))
    print(f'Overall counts of manual evaluation on {num_chats} chats')
    print('Num samples per chat:')
    idx = 1
    for name, group in df.groupby('chat_dir'):
        print(idx, name, "Num samples", len(group))
        idx += 1
    print('Eval type', eval_type)
    print("====")


def plot_chat_histograms_manual_eval(df):
    print(f"PLOT Manual Evaluation Results on {len(df)} samples")
    manual_best = Counter(df["Manual_ranked_prompt_best"])
    manual_middle = Counter(df["Manual_ranked_prompt_middle"])
    manual_worst = Counter(df["Manual_ranked_prompt_worst"])
    labels_map = {"baseline": "Baseline", "zero_shot": "CPE ZS", "few_shot": "CPE FS"}
    fig, ax = plt.subplots(layout='constrained')
    fig.set_size_inches(6.4, 2.4)
    x = np.arange(3)
    mul = 0
    width = 0.25
    for pt in actual_summary_prompt_types:
        c_best = manual_best[pt] / sum(manual_best.values())
        c_middle = manual_middle[pt] / sum(manual_middle.values())
        c_worst = manual_worst[pt] / sum(manual_worst.values())
        bars = [c_best, c_middle, c_worst]
        print("BARS", pt, "BEST", c_best, "MID", c_middle, "WORST", c_worst, "Num samples", len(df), sum(manual_best.values()))
        plt.bar(x+mul*width, bars, width, label=labels_map[pt])
        mul += 1
    ax.set_xticks(x + width/2, ["Best", "Middle", "Worst"])
    ax.legend(loc='upper left', ncols=len(actual_summary_prompt_types))
    #ax.set_title('Chat: Histogram of Manual Selection')
    out_eval_name = "manual_chats_hist_manual_eval_small.pdf"
    out_fig_file = os.path.join(overall_analysis_dir, out_eval_name)
    plt.savefig(out_fig_file, format='pdf')
    plt.show()


def plot_chat_histograms_manual_eval_transpose(df):
    print(f"PLOT TRANSPOSE Manual Evaluation Results on {len(df)} samples")
    manual_best = Counter(df["Manual_ranked_prompt_best"])
    manual_middle = Counter(df["Manual_ranked_prompt_middle"])
    manual_worst = Counter(df["Manual_ranked_prompt_worst"])
    labels_map = {"baseline": "Baseline", "zero_shot": "Zero-Shot", "few_shot": "Few-Shot"}
    fig, ax = plt.subplots(layout='constrained')
    x = np.arange(3)
    mul = 0
    width = 0.25
    for grade in ["best", "middle", "worst"]:
        manual_selection = Counter(df[f"Manual_ranked_prompt_{grade}"])
        manual_selection.subtract(zero_counter)
        manual_selection = get_normalized_counts(grade, manual_selection)
        bars = [manual_selection[pt] for pt in actual_summary_prompt_types]
        #print("Num samples", len(df), sum(manual_best.values()))
        plt.bar(x+mul*width, bars, width, label=grade)
        mul += 1
    ax.set_xticks(x + width/2, [labels_map[pt] for pt in actual_summary_prompt_types])
    ax.legend(loc='upper left', ncols=3)
    #ax.set_title('Chat: Histogram of Manual Selection')
    out_eval_name = "manual_chats_hist_manual_eval_tranpose.pdf"
    out_fig_file = os.path.join(overall_analysis_dir, out_eval_name)
    plt.savefig(out_fig_file, format='pdf')
    plt.show()


if __name__ == "__main__":

    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_4_8_2024"
    chats_list = [
        "ella.rabinovich1_tldr/31-07-2024 09_38_05",
        "lilache_debate_speeches/31-07-2024 07_55_01",
        "ronicon_wiki_movies/31-07-2024 15:44:31",
        "ronicon_wiki_animals/31-07-2024 14:45:15",
        "Roi.Cohen_wiki_animals/31-07-2024 10:22:19",
        "yoavka_wiki_animals/01-08-2024 05_54_11",
        "koren.lazar_wiki_animals/01-08-2024 13:43:30",
        "Gabija.Mikulyte_tldr/02-08-2024 16:28:31",
        "sam.a.smith_wiki_movies/02-08-2024 16:38:22",
        "philippos.arkis.hadjimarkou_tldr/02-08-2024 16:29:49",
        "ozery_wiki_animals/04-08-2024 13:39:04",
        "assaf.toledo_wiki_animals/05-08-2024 11:00:54",
    ]
    # "matano_wiki_movies/04-08-2024 07_50_37",    # analysis_summary_04-08-2024_15-00-30: continued chat session, few-shot examples are not uploaded, so ZS and FS prompts are identical

    time_stamp = "_" + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    print("Evaluation analysis time stamps:", time_stamp)
    print("Partial evaluation:", PARTIAL_EVALUATION, "evaluated_prompt_types", actual_summary_prompt_types)
    print("Evaluation chats directory:", chats_output_dir)
    overall_analysis_dir = os.path.join(chats_output_dir, f"analysis_summary{time_stamp}")

    df_manual_res = pd.DataFrame()
    for chat_dir in chats_list:
        chat_output_path = os.path.join(chats_output_dir, chat_dir)
        chat_eval_csv_file = os.path.join(chat_output_path, "eval/eval_results.csv")
        df_eval_chat = pd.read_csv(chat_eval_csv_file)
        df_eval_chat['chat_dir'] = chat_dir
        print(f"Evaluating {chat_dir} num samples {len(df_eval_chat)}")
        df_manual_res = pd.concat([df_manual_res, df_eval_chat], ignore_index=True, sort=False)

    os.makedirs(overall_analysis_dir, exist_ok=True)
    print("\n\nSUMMARY:", overall_analysis_dir, "Num chats", len(chats_list), "Total num smaples", len(df_manual_res))
    evaluate_chats(df_manual_res)




