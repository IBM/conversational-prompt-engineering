import json
import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from conversational_prompt_engineering.util.bam import BamGenerate

NUM_EXAMPLES_TO_LABEL = 5


parser = argparse.ArgumentParser()
parser.add_argument('--prompts_path', help='path for prompts file')
parser.add_argument('--data_path', help='path for test data')
parser.add_argument('--out_dir', help='path for saving evaluation files')


def compare_prompts_within_conversation(prompts_path, data_path, out_dir):
    with open(prompts_path, "r") as f:
        prompts = json.load(f)
    if len(prompts) > 3:
        prompts = [prompts[0]] + [prompts[-1]]  # keeping the first and last prompts
    test_df = pd.read_csv(data_path)
    if len(test_df) > NUM_EXAMPLES_TO_LABEL:
        test_df = test_df.sample(n=NUM_EXAMPLES_TO_LABEL, random_state=0)
    texts = test_df['text'].tolist()

    with open("conversational_prompt_engineering/backend/bam_params.json", "r") as f:
        bam_params = json.load(f)
    bam_params['api_key'] = os.environ['BAM_APIKEY']
    bam_client = BamGenerate(bam_params)

    generated_data_ordered = []
    generated_data_mixed = []
    for t in texts:
        row_data_ordered = {"prompts_file": prompts_path, "text": t}
        row_data_mixed = {"prompts_file": prompts_path, "text": t}
        prompts_responses = []
        for i, prompt in enumerate(tqdm(prompts)):
            prompt_str = prompt['prompt']
            conversation = prompt_str + "\n\nText: {text}\n\nSummary: "
            conversation_t = conversation.format(text=t)
            resp = bam_client.send_messages(conversation_t)[0]
            prompts_responses.append(resp)
        for i in range(len(prompts)):
            row_data_ordered[str(i)+"_prompt"] = prompts[i]['prompt']
            row_data_ordered[str(i)] = prompts_responses[i]
        mixed_indices = np.argsort(prompts_responses)
        for i in mixed_indices:
            row_data_mixed[str(i) + "_prompt"] = prompts[i]['prompt']
            row_data_mixed[str(i)] = prompts_responses[i]
        generated_data_ordered.append(row_data_ordered)
        generated_data_mixed.append(row_data_mixed)

    os.makedirs(out_dir, exist_ok=True)
    df = pd.DataFrame(generated_data_ordered)
    df.to_csv(os.path.join(out_dir, "evaluate_ordered.csv"), index=False)

    df = pd.DataFrame(generated_data_mixed)
    df.to_csv(os.path.join(out_dir, "evaluate_mixed.csv"), index=False)
    df = df[[c for c in df.columns if not c.endswith("prompt")]]
    df.to_csv(os.path.join(out_dir, "evaluate_mixed_hidden.csv"), index=False)


if __name__ == "__main__":
    args = parser.parse_args()
    compare_prompts_within_conversation(prompts_path=args.prompts_path,
                                        data_path=args.data_path,
                                        out_dir=args.out_dir)