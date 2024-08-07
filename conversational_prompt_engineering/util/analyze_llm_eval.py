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

PARTIAL_EVALUATION = False #True
EVAL_CHAT = True
EVAL_OFFLINE = False #True

actual_summary_prompt_types = summary_prompt_types
eval_ver = "_ALL"
if PARTIAL_EVALUATION:
    eval_ver = "_BS_ZS"
    actual_summary_prompt_types = ['baseline', 'zero_shot']
    EVAL_CHAT = True

    #eval_ver = "_BS_FS"
    #actual_summary_prompt_types = ['baseline', 'few_shot']
    #EVAL_CHAT = True

    #eval_ver = "_ZS_FS"
    #actual_summary_prompt_types = ['zero_shot', 'few_shot']
    #EVAL_CHAT = True

    # eval_ver = "_BSFT_FT"
    # actual_summary_prompt_types = ['baseline_few_shot', 'few_shot']
    # EVAL_CHAT = False

    eval_type = f"_partial{len(actual_summary_prompt_types)}{eval_ver}"
else:
    eval_type = f"_full{len(actual_summary_prompt_types)}{eval_ver}"

zero_counter = Counter({pt: 0 for pt in actual_summary_prompt_types})

ttest_mappings = {"baseline-zero_shot": {"zero_shot": 0.5, "baseline": -0.5, "-1": 0},
                  "baseline-few_shot": {"few_shot": 0.5, "baseline": -0.5, "-1": 0},
                  "zero_shot-few_shot": {"few_shot": 0.5, "zero_shot": -0.5, "-1": 0},
                  "baseline_few_shot-few_shot": {"few_shot": 0.5, "baseline_few_shot": -0.5, "-1": 0}}
pvalue_alpha = 0.05


def get_normalized_counts(type_name, counts, with_print=True):
    c_norm = {k: v / sum(counts.values()) for k, v in sorted(counts.items(), key=lambda x: x[1], reverse=True)}
    if with_print:
        print(type_name, counts, "Normalized:", c_norm, sum(counts.values()))
    return c_norm


def get_prompt_types(column_name):
    if 'llm_judge_rel_result' in column_name:
        return re.findall(r'\<(.*?)\>', column_name)
    if 'llm_judge_abs_result' in column_name:
        prompt_type = column_name.replace("_llm_judge_abs_result", "")
        return [prompt_type]
    return []


def use_llm_result(column_name):
    prompt_types = get_prompt_types(column_name)
    if len(prompt_types) > 0:
        return all([True if p in actual_summary_prompt_types else False for p in prompt_types])
    return False


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


def run_ttest_two_sided(ttest_data, mapping, alpha=pvalue_alpha):
    val_map = dict((v, k) for k, v in mapping.items())
    ttest_res = ttest_1samp(ttest_data, popmean=0.0)
    pvalue = getattr(ttest_res, 'pvalue')
    tstat = getattr(ttest_res, 'statistic')
    significant = False
    selected = ""
    if (tstat > 0 or tstat < 0) and (pvalue/2 < alpha):
        significant = True
        selected = val_map[np.sign(tstat)*0.5]
    print("RUN TTEST TWO SIDED", pvalue, "STAT", tstat, significant, selected, mapping)


def llm_evaluation_stats(df):
    print(len(df))
    stats_res = {"llm_abs": {}, "llm_rel": {}}
    print('\nAbsolute scores (mean, var):')
    for col in df.columns:
        if not use_llm_result(col):
            continue
        if 'llm_judge_abs_result' in col:
            pt = get_prompt_types(col)[0]
            print(col, pt, f'{df[col].mean():.2f}', f'{df[col].var():.2f}', len(df[col]), df[col].tolist())
            stats_res["llm_abs"].update({pt: {"avg": df[col].mean(), "var": df[col].var(),
                            "num": len(df[col])}})#, "counts": Counter(df[col])}})

    print('\nRelative scores:')
    total_counts = {}
    for col in df.columns:
        if not use_llm_result(col):
            continue
        if 'llm_judge_rel_result' in col:
            counter = Counter(df[col])
            print(col, dict(counter), 'num_samples:', sum(counter.values()))
            prompt_types = "-".join(sorted(get_prompt_types(col)))
            if prompt_types not in total_counts:
                total_counts[prompt_types] = Counter()
            total_counts[prompt_types] += counter

    print('\nTotal all pairs relative normalized counts')
    overall_counts = Counter()
    for pt, c in total_counts.items():
        c_norm = get_normalized_counts(pt, c)
        c.subtract(zero_counter)
        print(pt, "Chisq P-value:", chisquare(list(c.values())).pvalue, sum(c.values()))
        overall_counts += c
    c_norm = get_normalized_counts("\nOverall all pairs best prompt", overall_counts)
    overall_counts.subtract(zero_counter)
    pvalue = chisquare(list(overall_counts.values())).pvalue
    num = sum(overall_counts.values())
    print("Overall all pairs Chisq", "P-value:", pvalue, "Num:", num)
    stats_res["llm_rel"].update({"Overall all pairs": {"chisq_pvalue": pvalue, "num": num, "counts": Counter(overall_counts)}})
    stats_res["llm_rel"].update({"Ttest pairs": {}})
    for (p1, p2) in get_all_pairs(actual_summary_prompt_types, add_reversed_pairs=False):
        prompt1_col = f"<{p1}>-<{p2}>_llm_judge_rel_result"
        prompt2_col = f"<{p2}>-<{p1}>_llm_judge_rel_result"
        if prompt1_col in df.columns and prompt2_col in df.columns:
            ttest_map = ttest_mappings[f"{p1}-{p2}"]
            ttest_data = [ttest_map[p1]+ttest_map[p2] for p1, p2 in zip(df[prompt1_col], df[prompt2_col])]
            ttest_res = run_ttest_one_sided(ttest_data, ttest_map)
            ttest_res.update({"prompt1_col": df[prompt1_col].tolist(), "prompt2_col": df[prompt2_col].tolist()})
            print(ttest_res)
            stats_res["llm_rel"]["Ttest pairs"].update({f"<{p1}>-<{p2}>": ttest_res})
    return stats_res


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


def compute_agreement(df):
    print("Manual bset:\t", df["Manual_ranked_prompt_best"].tolist())
    print("LLM best:   \t", df["Best_llm_judge_rel"].tolist())
    print("LLM best score:\t", df["Best_llm_judge_rel_score"].tolist())
    agreement = [1 if manual in llm else 0 for manual, llm in
                 zip(df["Manual_ranked_prompt_best"], df["Best_llm_judge_rel"])]
    df["Agreement"] = agreement
    #agreement = [a for a, s in zip(agreement, df["Best_llm_judge_rel_score"]) if s > 0.5]
    #agreement_avg = sum(agreement) / len(agreement)
    agreement_avg = np.dot(df["Agreement"], df["Best_llm_judge_rel_score"])/len(df)
    num_decisions = sum([1 if s > 0.5 else 0 for s in df["Best_llm_judge_rel_score"]])
    res = {"weighted_agreement": agreement_avg, "num": len(agreement), "num_llm_decisions": num_decisions}
    print(res)
    return res


def save_evaluation(df, eval_chat_file):
    out_csv = eval_chat_file.replace('.csv', f'_analysis{eval_type}.csv')
    print('Analysis output csv file:', out_csv)
    df.to_csv(out_csv, index=False)


def save_evaluation_results_json(eval_res, eval_out_path):
    out_json = os.path.join(eval_out_path, f"evaluation_analysis{eval_type}.json")
    print('Analysis output json file:', out_json)
    with open(out_json, 'w') as f:
        json.dump(eval_res, f)


def analyze_llm_evaluation(df):
    df_dict = df.to_dict(orient='records')
    if 'llm_evaluated_instruction' in df_dict[0]:
        evaluated_instruction = df_dict[0]['llm_evaluated_instruction']
    else:
        evaluated_instruction = "N/A"
    total_counts = Counter()
    total_counts_all_pairs = Counter()
    llm_best_prompt = []
    llm_best_prompt_score = []
    llm_pvalue_rel = []
    llm_num_rel = []
    for row in df_dict:
        llm_selected_prompt = []
        for k in row.keys():
            if not use_llm_result(k):
                continue
            if "_llm_judge_rel_result" in k:
                llm_selected_prompt.append(row[k])
        counts = Counter(llm_selected_prompt)
        counts.subtract(zero_counter)
        max_count = max(counts.values())
        max_keys = [k for k in counts.keys() if counts[k] == max_count]
        norm_counts = get_normalized_counts('Overall', counts, with_print=False)
        llm_best_prompt.append(max_keys)
        llm_best_prompt_score.append(1.0/len(max_keys))
        llm_pvalue_rel.append(chisquare(list(counts.values())).pvalue)
        llm_num_rel.append(sum(counts.values()))
        total_counts += norm_counts
        total_counts_all_pairs += counts

    df['Best_llm_judge_rel'] = llm_best_prompt
    df['Best_llm_judge_rel_score'] = llm_best_prompt_score
    df['Llm_pvalue_rel'] = llm_pvalue_rel
    df['Llm_num_rel'] = llm_num_rel
    norm_total_counts = get_normalized_counts('Overall', total_counts, with_print=False)
    analysis_res = {"llm_evaluated_instruction": evaluated_instruction, "norm_total_counts": norm_total_counts,
                    "total_counts": total_counts, "total_num": sum(total_counts.values()),
                    "total_counts_all_pairs": total_counts_all_pairs,
                    "total_num_all_pairs": sum(total_counts_all_pairs.values())}
    return df, analysis_res


def compute_pvalue(df, col_name):
    counts_col = Counter(df[col_name])
    counts_col.subtract(zero_counter)
    get_normalized_counts(col_name, counts_col)
    pvalue_col = chisquare(list(counts_col.values())).pvalue
    num_col = sum(counts_col.values())
    print(col_name, "Chi-sq P-value:", pvalue_col, "Num:", num_col)
    return pvalue_col, num_col, counts_col


def get_manual_pvalue(df):
    pvalue_best, num_best, counts_best = compute_pvalue(df, "Manual_ranked_prompt_best")
    pvalue_worst, num_worst, counts_worst = compute_pvalue(df, "Manual_ranked_prompt_worst")
    analysis_res = {"manual_eval": {"best_pvalue": pvalue_best, "worst_pvalue": pvalue_worst, "num_best": num_best,
                                    "num_worst": num_worst,
                                    "counts_best": counts_best, "counts_worst": counts_worst}}
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
    analysis_res = get_manual_pvalue(df_res)
    return df_res, analysis_res


def evaluate_offline(test_split):
    llm_eval_file = f'{test_split}.offline.llm_judge_evaluation.csv'
    offline_eval_results = os.path.join(llm_judge_dir, llm_eval_file)
    #offline_eval_results = f'llm_judge/{target_model}/{test_split}.offline.llm_judge_evaluation.csv'
    eval_llm_file = os.path.join(chat_output_path, offline_eval_results)
    if not os.path.isfile(eval_llm_file):
        print(f'Skip {offline_eval_results}, file does not exist')
        return None, None
    # Offline eval
    print(f"\n======= Offline Evaluation {eval_llm_file}")
    df_llm_offline = pd.read_csv(eval_llm_file, index_col=False)
    print("Num samples", len(df_llm_offline))
    eval_llm_stats = llm_evaluation_stats(df_llm_offline)
    print(f"\n==================================")
    df_llm_offline, eval_llm_res = analyze_llm_evaluation(df_llm_offline)

    eval_res = {"llm_eval": {}}
    eval_res["llm_eval"].update(eval_llm_res)
    eval_res["llm_eval"].update(eval_llm_stats)
    df_llm_offline["chat_dir"] = chat_dir
    df_llm_offline["chat_path"] = chat_output_path
    df_llm_offline["data_split"] = split

    out_results = os.path.join(llm_judge_analysis_dir, llm_eval_file)
    eval_llm_file = os.path.join(chat_output_path, out_results)
    save_evaluation(df_llm_offline, eval_llm_file)
    return eval_res, df_llm_offline


def eval_chat_df(df_chat):
    print("Num samples", len(df_chat))
    eval_llm_stats = llm_evaluation_stats(df_chat)

    print(f"\n====== Manual Best Worst counts")
    df_chat, eval_res = analyze_manual_evaluation(df_chat)
    print(f"\n====== LLM Best counts")
    df_chat, eval_llm_res = analyze_llm_evaluation(df_chat)
    print(f"\n====== Manual and LLM Best agreement")
    agreement = compute_agreement(df_chat)
    eval_res.update({"manual_llm_agreement": agreement, "llm_eval": {}})
    eval_res["llm_eval"].update(eval_llm_res)
    eval_res["llm_eval"].update(eval_llm_stats)
    return eval_res, df_chat


def evaluate_chat():
    llm_eval_file = 'eval_results.chat.llm_judge_evaluation.csv'
    llm_eval_results = os.path.join(llm_judge_dir, llm_eval_file)
    # llm_eval_results = f'llm_judge/{target_model}/eval_results.chat.llm_judge_evaluation.csv'
    eval_chat_llm_file = os.path.join(chat_output_path, llm_eval_results)
    if not os.path.isfile(eval_chat_llm_file):
        print(f'Skip {llm_eval_results}, file does not exist in {chat_output_path}')
        return
    # Chat eval
    print(f"\n====== Chat Evaluation {eval_chat_llm_file}")
    df_chat = pd.read_csv(eval_chat_llm_file, index_col=False).dropna()
    eval_res, df_chat = eval_chat_df(df_chat)
    df_chat["chat_dir"] = chat_dir
    df_chat["chat_path"] = chat_output_path

    out_results = os.path.join(llm_judge_analysis_dir, llm_eval_file)
    eval_chat_llm_file = os.path.join(chat_output_path, out_results)
    save_evaluation(df_chat, eval_chat_llm_file)
    return eval_res, df_chat


def print_overall_manual_res(df):
    num_chats = len(set(df['chat_dir']))
    print(f'Overall counts of manual evaluation on {num_chats} chats')
    print('Num samples per chat:')
    idx = 1
    for name, group in df.groupby('chat_dir'):
        print(idx, name, "Num samples", len(group))
        idx += 1
    print('Eval type', eval_type)
    print("====")
    get_normalized_counts("Manual Best", Counter(df["Manual_ranked_prompt_best"]))
    get_normalized_counts("Manual Middle", Counter(df["Manual_ranked_prompt_middle"]))
    get_normalized_counts("Manual Worst", Counter(df["Manual_ranked_prompt_worst"]))
    print("====")
    get_manual_pvalue(df)
    print("Overall agreement")
    compute_agreement(df)


def print_overall_llm_res(df, eval_name):
    num_chats = len(set(df['chat_dir']))
    print(f'Overall on all chats: counts of llm evaluation on {num_chats} chats, num samples {len(df)}')
    print('Eval name', eval_name)
    print('Eval type', eval_type)
    for p in actual_summary_prompt_types:
        abs_score = df[f"{p}_llm_judge_abs_result"]
        print(f"Overall on all chats: avg. absolute score {p}: mean:", np.mean(abs_score), "std:", np.std(abs_score), "num:", len(abs_score))
    ### CORRECT THIS
    ##all_counts = Counter([p for pl in df['Best_llm_judge_rel'] for p in pl])
    ##print(all_counts)
    ##get_normalized_counts("All chats", all_counts)


def print_overall_ttest_res(summary_res):
    print('Overall T-test results:')
    for chat_dir in chats_list:
        if EVAL_CHAT:
            test_type = "manual_chat"
            for pair in summary_res[f"{test_type}_evaluation"][chat_dir]["llm_eval"]["llm_rel"]["Ttest pairs"].keys():
                if ">-<" in pair:
                    print(chat_dir, "llm_eval", test_type, pair,
                          summary_res[f"{test_type}_evaluation"][chat_dir]["llm_eval"]["llm_rel"]["Ttest pairs"][pair])

        if EVAL_OFFLINE:
            test_type = "offline_test"
            for split in offline_test_splits:
                if split in summary_res[f"{test_type}_evaluation"][chat_dir].keys():
                    for pair in summary_res[f"{test_type}_evaluation"][chat_dir][split]["llm_eval"]["llm_rel"]["Ttest pairs"].keys():
                        if ">-<" in pair:
                            print(chat_dir, "llm_eval", test_type, split, pair,
                                  summary_res[f"{test_type}_evaluation"][chat_dir][split]["llm_eval"]["llm_rel"]["Ttest pairs"][pair])


def plot_chat_histograms_orig(df):
    manual_best = Counter(df["Manual_ranked_prompt_best"])
    manual_worst = Counter(df["Manual_ranked_prompt_worst"])
    llm_abs_score = {p: np.mean(df[f"{p}_llm_judge_abs_result"]) for p in actual_summary_prompt_types}
    fig, ax = plt.subplots(layout='constrained')
    x = np.arange(2)
    mul = 0
    width = 0.25
    for pt in actual_summary_prompt_types:
        c_best = manual_best[pt] / sum(manual_best.values())
        c_worst = manual_worst[pt] / sum(manual_worst.values())
        c_llm = llm_abs_score[pt] / 5.0
        bars = [c_best, c_llm] #[c_best, c_worst, c_llm]
        print(pt, c_best, c_worst, c_llm)
        plt.bar(x+mul*width, bars, width, label=pt)
        mul += 1
    ax.set_xticks(x + width/2, ["best", "llm_abs"])  # ["best", "worst", "llm_abs"]
    ax.legend(loc='upper left', ncols=len(actual_summary_prompt_types))
    ax.set_title('Chat: 1) Histogram of Manual Selection 2) Avg. LLM Absolute Score')
    out_eval_name = "manual_chats_hist.pdf"
    out_fig_file = os.path.join(overall_analysis_dir, out_eval_name)
    plt.savefig(out_fig_file, format='pdf')
    plt.show()


def plot_chat_llm_rel_select_best_histogram(df, eval_name, manual_best=None):
    if manual_best is None:
        if "Manual_ranked_prompt_best" in df.columns:
            manual_best = Counter(df["Manual_ranked_prompt_best"])
        else:
            manual_best = zero_counter
    llm_best = df['Best_llm_judge_rel']
    llm_best_score = df['Best_llm_judge_rel_score']
    print(llm_best.tolist())
    print(llm_best_score.tolist())
    fig, ax = plt.subplots(layout='constrained')
    x = np.arange(2)
    mul = 0
    width = 0.25
    for pt in actual_summary_prompt_types:
        c_best = manual_best[pt]
        if sum(manual_best.values()):
            c_best /= sum(manual_best.values())
        best_samples = [1 if pt in b else 0 for b in llm_best]
        c_llm = np.dot(best_samples, llm_best_score)/len(best_samples)
        bars = [c_best, c_llm]
        print(pt, c_best, c_llm, best_samples)
        plt.bar(x + mul * width, bars, width, label=pt)
        mul += 1
    ax.set_xticks(x + width / 2, ["manual_best", "llm_rel_best"])
    ax.legend(loc='upper left', ncols=len(actual_summary_prompt_types))
    ax.set_title('Chat: 1) Histogram of Manual Selection 2) Histogram of LLM Selection by Relative Evaluation')
    out_eval_name = f"{eval_name}_manual_chats_llm_rel_select_best_hist.pdf"
    out_fig_file = os.path.join(overall_analysis_dir, out_eval_name)
    plt.savefig(out_fig_file, format='pdf')
    plt.show()


def plot_chat_llm_rel_best_all_pairs_histogram(llm_stats, res_manual, eval_name):
    manual_best = res_manual['manual_eval']['counts_best']
    llm_rel_best = llm_stats['llm_rel']['Overall all pairs']['counts']
    print("HIST_JSON", manual_best, llm_rel_best)
    manual_best = get_normalized_counts("Overall", manual_best, with_print=False)
    llm_rel_best = get_normalized_counts("Overall", llm_rel_best, with_print=False)
    fig, ax = plt.subplots(layout='constrained')
    x = np.arange(2)
    mul = 0
    width = 0.25
    for pt in actual_summary_prompt_types:
        bars = [manual_best[pt], llm_rel_best[pt]]
        plt.bar(x + mul * width, bars, width, label=pt)
        mul += 1
    ax.set_xticks(x + width / 2, ["manual_best", "llm_rel_best"])
    ax.legend(loc='upper left', ncols=len(actual_summary_prompt_types))
    ax.set_title(f'{eval_name}: 1) Histogram of Manual Selection 2) Histogram of LLM Selection by Relative Evaluation')
    out_eval_name = f"{eval_name}_manual_chats_llm_rel_best_all_pairs_hist.pdf"
    out_fig_file = os.path.join(overall_analysis_dir, out_eval_name)
    plt.savefig(out_fig_file, format='pdf')
    plt.show()


def plot_chat_histograms_manual_eval(df):
    print(f"PLOT Manual Evaluation Results on {len(df)} samples")
    manual_best = Counter(df["Manual_ranked_prompt_best"])
    manual_middle = Counter(df["Manual_ranked_prompt_middle"])
    manual_worst = Counter(df["Manual_ranked_prompt_worst"])

    fig, ax = plt.subplots(layout='constrained')
    x = np.arange(3)
    mul = 0
    width = 0.25
    for pt in actual_summary_prompt_types:
        c_best = manual_best[pt] / sum(manual_best.values())
        c_middle = manual_middle[pt] / sum(manual_middle.values())
        c_worst = manual_worst[pt] / sum(manual_worst.values())
        bars = [c_best, c_middle, c_worst]
        print("BARS", pt, "BEST", c_best, "MID", c_middle, "WORST", c_worst, "Num samples", len(df), sum(manual_best.values()))
        plt.bar(x+mul*width, bars, width, label=pt)
        mul += 1
    ax.set_xticks(x + width/2, ["Best", "Middle", "Worst"])
    ax.legend(loc='upper left', ncols=len(actual_summary_prompt_types))
    #ax.set_title('Chat: Histogram of Manual Selection')
    out_eval_name = "manual_chats_hist_manual_eval.pdf"
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

def plot_offline_histograms_abs_orig(df, eval_name):
    datasets_counts = {}
    for name, group in df.groupby('dataset_name'):
        datasets_counts[name] = group
    fig, ax = plt.subplots(layout='constrained')
    datasets_names = list(datasets_counts.keys())
    if len(datasets_names) > 1:
        x = np.arange(len(datasets_names)+1)
    else:
        x = np.arange(len(datasets_names))
    mul = 0
    width = 0.25
    for pt in actual_summary_prompt_types:
        scores = {d: gdf[f"{pt}_llm_judge_abs_result"].tolist() for d, gdf in datasets_counts.items()}
        bars = [np.mean(scores[d]) for d in datasets_names]
        num_per_dataset = [len(scores[d]) for d in datasets_names]
        if len(datasets_names) > 1:
            overall_mean_score = [np.dot(bars, num_per_dataset) / sum(num_per_dataset)]
            overall_name = ['All']
        else:
            overall_mean_score = []
            overall_name = []
        print("HIST", pt, "Num:", num_per_dataset, "Total Num:", sum(num_per_dataset))
        plt.bar(x + mul * width, bars + overall_mean_score, width, label=pt)
        mul += 1
    ax.set_xticks(x + width/2, datasets_names + overall_name)
    ax.legend(loc='upper left', ncols=len(actual_summary_prompt_types))
    ax.set_title(eval_name + ": Avg. LLM Absolute Score")
    out_eval_name = "llm_offline." + eval_name.lower().replace(" ", "_") + ".pdf"
    out_fig_file = os.path.join(overall_analysis_dir, out_eval_name)
    plt.savefig(out_fig_file, format='pdf')
    plt.show()


def plot_chat_histograms_manual_eval_v2(chats_stats):
    print("HIST CHAT NEW", chats_stats)
    manual_best = chats_stats["manual_eval"]["counts_best"]
    #manual_middle = Counter(df["Manual_ranked_prompt_middle"])
    manual_worst = chats_stats["manual_eval"]["counts_worst"]
    manual_best = get_normalized_counts("Overall", manual_best, with_print=False)
    #manual_middle = get_normalized_counts("Overall", manual_middle, with_print=False)
    manual_worst = get_normalized_counts("Overall", manual_worst, with_print=False)

    fig, ax = plt.subplots(layout='constrained')
    x = np.arange(2)
    mul = 0
    width = 0.25
    for pt in actual_summary_prompt_types:
        c_best = manual_best[pt]
        #c_middle = manual_middle[pt]
        c_worst = manual_worst[pt]
        bars = [c_best, c_worst]
        print(pt, c_best, c_worst)
        plt.bar(x+mul*width, bars, width, label=pt)
        mul += 1
    ax.set_xticks(x + width/2, ["best", "worst"])
    ax.legend(loc='upper left', ncols=len(actual_summary_prompt_types))
    ax.set_title('Chat: Histogram of Manual Selection (Best, Worst)')
    out_eval_name = "manual_chats_hist_manual_eval_v2.pdf"
    out_fig_file = os.path.join(overall_analysis_dir, out_eval_name)
    plt.savefig(out_fig_file, format='pdf')
    plt.show()


def plot_offline_histograms(llm_stats, eval_name):
    llm_abs = llm_stats['llm_abs']
    llm_rel_best = llm_stats['llm_rel']['Overall all pairs']['counts']
    print("HIST ABS NEW JSON", llm_abs, llm_rel_best)
    llm_rel_best = get_normalized_counts("Overall", llm_rel_best, with_print=False)
    fig, ax = plt.subplots(layout='constrained')
    bars = [llm_abs[pt]['avg'] for pt in actual_summary_prompt_types]
    errors = [llm_abs[pt]['var'] for pt in actual_summary_prompt_types]
    print("HIST ABS NEW BARS", bars)
    width = 0.1
    x = np.arange(len(actual_summary_prompt_types))
    plt.bar(x, bars)
    ax.set_xticks(x, actual_summary_prompt_types)
    #ax.legend(loc='upper left', ncols=len(actual_summary_prompt_types))
    ax.set_title(f'{eval_name}: Histogram of LLM Avg. Absolute Score')
    out_eval_name = f"{eval_name}_offline_llm_abs.pdf"
    out_fig_file = os.path.join(overall_analysis_dir, out_eval_name)
    plt.savefig(out_fig_file, format='pdf')
    plt.show()


if __name__ == "__main__":
    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out"

    chats_list = [
        "oritht/14-07-2024 12:36:46",
        "liat/21-07-2024 12:16:37",
        "shai/21-07-2024 12:36:52",
    ]

    chats_list = [
        "liat/21-07-2024 12:16:37",
        "shai/wiki_animals",
    ]

    chats_list = [
        "shai/wiki_animals",
        "Evaluation_24_7_2024/Shai_20ng_space/24-07-2024 12:33:50",
        "Evaluation_24_7_2024/Artem_cfpb/24-07-2024 10:25:30",
        "Evaluation_24_7_2024/Artem_financial_news/24-07-2024 11:09:44",
        "Evaluation_24_7_2024/Artem_reddit/24-07-2024 09:45:58",
        #"Evaluation_24_7_2024/CIO/24-07-2024 14:12:09",
        "Evaluation_24_7_2024/Artem_speeches/24-07-2024 13:09:34",
        "Evaluation_24_7_2024/Artem_wiki_movies/24-07-2024 15:57:47",
        "Evaluation_24_7_2024/Liat_speeches/24-07-2024 16:47:16",
        "Evaluation_24_7_2024/Liat_wiki_movies/24-07-2024 17:54:36",
        "Evaluation_24_7_2024/Orith_wiki_movies/25-07-2024 11:52:11",
    ]

    ## Evaluation for paper: CIO
    #chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_CIO"
    #chats_list = [
    #    "gmelino_microsoft/24-07-2024 14:17:00"
    #]

    ## Evaluation for paper: ISRL
    #chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_ISRL"
    #chats_list = [
    #    "eladv_wiki_movies/25-07-2024 13:22:07",
    #    "Roi.Cohen_wiki_animals/25-07-2024 12:38:25"
    #]

    #chats_list = [
    #    "Evaluation_24_7_2024/Shai_20ng_space_copy/24-07-2024 12:33:50",
    #]

    # Chats for analysis
    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_24_7_2024"
    chats_list = [
        "Shai_20ng_space/24-07-2024 12:33:50",
        "Liat_speeches/24-07-2024 16:47:16",
        "Liat_wiki_movies/24-07-2024 17:54:36",
    ]

    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_old"
    chats_list = [
        "liat/21-07-2024 12:16:37",
    ]

    #chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_30_7_2024"
    #chats_list = [
    #    "Ariel.gera1/30-07-2024 14:44:27",
    #    "shachar.don-yehiya/30-07-2024 10:25:36",
    #]


    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_31_7_2024"
    # 1.8.24 Revert the LLM relative prompt to the original one from prometheus_eval
    chats_output_dir = "/Users/oritht/Projects/conversational-prompt-engineering/conversational_prompt_engineering/_out/Evaluation_1_8_2024"
    chats_list = [
        "ella.rabinovich1_tldr/31-07-2024 09_38_05",
        "lilache_debate_speeches/31-07-2024 07_55_01",
        "ronicon_wiki_movies/31-07-2024 15:44:31",
        "ronicon_wiki_animals/31-07-2024 14:45:15",
        # "matano_wiki_movies/31-07-2024 14:03:59", # dummy labels
        "Roi.Cohen_wiki_animals/31-07-2024 10:22:19",
        "yoavka_wiki_animals/01-08-2024 05_54_11",
        "koren.lazar_wiki_animals/01-08-2024 13:43:30",
    ]

    #chats_list = [
    #    "noams/30-07-2024 10:49:32",
    #]

    time_stamp = "_" + datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    #time_stamp = "_debugXXXXX"
    print("Evaluation analysis time stamps:", time_stamp)

    llm_pivot_prompt = 'zero_shot'
    target_model = 'llama-3'
    #llm_judge_dir = f"llm_judge/pivot_{llm_pivot_prompt}/{target_model}"
    # 1.8.24 change rubric with summarization
    llm_judge_dir = f"llm_judge_new_rubric/pivot_{llm_pivot_prompt}/{target_model}"

    llm_judge_analysis_dir = f"{llm_judge_dir}/analysis{time_stamp}"
    overall_analysis_dir = os.path.join(chats_output_dir, f"analysis_summary{time_stamp}")

    offline_test_splits = ["eval", "test", "test_full"]

    print("LLM judge pivot:", llm_pivot_prompt)
    print("Parial evaluation:", PARTIAL_EVALUATION, "evaluated_prompt_types", actual_summary_prompt_types)

    offline_res = {}
    manual_res = {}
    df_manual_res = pd.DataFrame()
    df_llm_res = pd.DataFrame()
    for chat_dir in chats_list:
        chat_output_path = os.path.join(chats_output_dir, chat_dir)
        print(f"Evaluating {chat_dir}")
        chat_analysis_dir = os.path.join(chat_output_path, llm_judge_analysis_dir)
        os.makedirs(chat_analysis_dir, exist_ok=True)
        manual_res.update({chat_dir: {}})
        chat_res = {"chat": chat_dir, "target_model": target_model, "evaluated_prompt_types": actual_summary_prompt_types}
        if EVAL_CHAT:
            eval_result, df_chat_eval = evaluate_chat()
            df_manual_res = pd.concat([df_manual_res, df_chat_eval], ignore_index=True, sort=False)
            manual_res[chat_dir].update(eval_result)
            chat_res.update({"manual": manual_res[chat_dir]})
        if EVAL_OFFLINE:
            offline_res.update({chat_dir: {}})
            for split in offline_test_splits:
                eval_result, df_llm_eval = evaluate_offline(split)
                if eval_result is None:
                    continue
                df_llm_res = pd.concat([df_llm_res, df_llm_eval], ignore_index=True, sort=False)
                offline_res[chat_dir].update({split:eval_result})
            chat_res.update({"offline": offline_res[chat_dir]})
        save_evaluation_results_json(chat_res, chat_analysis_dir)

    os.makedirs(overall_analysis_dir, exist_ok=True)
    print("\n\nSUMMARY:", overall_analysis_dir)
    summary_res = {"manual_chat_evaluation": manual_res, "offline_test_evaluation": offline_res,
                   "target_model": target_model, "evaluated_prompt_types": actual_summary_prompt_types}
    save_evaluation_results_json(summary_res, overall_analysis_dir)
    print_overall_ttest_res(summary_res)

    if EVAL_CHAT:
        eval_name = "Offline Chat Evaluation"
        print("Manual evaluation results")
        out_csv = os.path.join(overall_analysis_dir, f"manual_evaluation_analysis.csv")
        print('Analysis output out csv file:', out_csv)
        save_evaluation(df_manual_res, out_csv)

        print("=======================")
        print("==== OVERALL RESULTS ==")
        print("=======================")
        print("Manual evaluation results on chat")
        print_overall_manual_res(df_manual_res)
        #df_chats, res_chats = analyze_manual_evaluation(df_manual_res)
        #stats_llm = llm_evaluation_stats(df_manual_res)
        #_, res_llm = analyze_llm_evaluation(df_manual_res)
        #plot_chat_histograms(df_manual_res)
        plot_chat_histograms_manual_eval(df_manual_res)
        plot_chat_histograms_manual_eval_transpose(df_manual_res)
        #plot_chat_histograms_manual_eval_v2(res_chats)

        #plot_chat_llm_rel_select_best_histogram(df_manual_res, "Chat")
        ##plot_chat_llm_rel_best_all_pairs_histogram(stats_llm, res_chats, "Chat")

        #print("=======================")
        #print("LLM evaluation results on chat")
        #print_overall_llm_res(df_manual_res, eval_name)
        #print("=======================")
        #plot_offline_histograms_abs_orig(df_manual_res, eval_name)
        ##plot_offline_histograms(stats_llm, "Chat")

    if EVAL_OFFLINE:
        eval_name = "Offline Test Evaluation"
        print("LLM evaluation results on offline tests")
        out_csv = os.path.join(overall_analysis_dir, f"llm_evaluation_analysis.csv")
        print('Analysis output out csv file:', out_csv)
        save_evaluation(df_llm_res, out_csv)

        print("=======================")
        print("==== OVERALL RESULTS ==")
        print("=======================")
        print("=======================")
        print("LLM evaluation results on chat")
        print_overall_llm_res(df_llm_res, eval_name)
        print("=======================")

        plot_offline_histograms_abs_orig(df_llm_res, eval_name)
        #plot_offline_histograms(stats_llm, "Test")

        #stats_llm = llm_evaluation_stats(df_llm_res)
        #_, res_llm = analyze_llm_evaluation(df_llm_res)
        plot_chat_llm_rel_select_best_histogram(df_llm_res, "Offline")
        #plot_chat_llm_rel_best_all_pairs_histogram(stats_llm, res_chats, "Offline")

    print("Partial evaluation:", PARTIAL_EVALUATION, "evaluated_prompt_types", actual_summary_prompt_types, "time_stamp", time_stamp)


