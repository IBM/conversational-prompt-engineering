import json
import logging
import os
import random

import pandas as pd
from tqdm import tqdm
import argparse
from conversational_prompt_engineering.util.bam import BamGenerate

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

NUM_EXAMPLES_TO_LABEL = 5


parser = argparse.ArgumentParser()
parser.add_argument('--prompts_path', help='path for prompts file')
parser.add_argument('--data_path', help='path for test data')
parser.add_argument('--out_dir', help='path for saving evaluation files')


class Evaluation:

    def __init__(self, bam_client):
        self.bam_client = bam_client

    def get_prompts_to_evaluate(self, prompts):
        if len(prompts) > 2:
            prompts = [prompts[0]] + [prompts[-1]]  # keeping the first and last prompts
        return prompts

    def compare_prompts_within_conversation(self, prompts_path, data_path, out_dir):
        with open(prompts_path, "r") as f:
            prompts = json.load(f)
        prompts = self.get_prompts_to_evaluate(prompts)
        test_df = pd.read_csv(data_path)
        if len(test_df) > NUM_EXAMPLES_TO_LABEL:
            test_df = test_df.sample(n=NUM_EXAMPLES_TO_LABEL, random_state=0)
        texts = test_df['text'].tolist()

        generated_data_mixed, generated_data_ordered = self.summarize(prompts, [f"{i}" for i in range(len(prompts))], texts)

        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(generated_data_ordered)
        df.to_csv(os.path.join(out_dir, "evaluate_ordered.csv"), index=False)

        df = pd.DataFrame(generated_data_mixed)
        df.to_csv(os.path.join(out_dir, "evaluate_mixed.csv"), index=False)
        df = df[[c for c in df.columns if not c.endswith("prompt")]]
        df.to_csv(os.path.join(out_dir, "evaluate_mixed_hidden.csv"), index=False)

        logging.info(f"evaluation files saved to {out_dir}")

    def summarize(self, prompts, prompt_types, texts):
        generated_ordered = []
        for i, t in enumerate(texts):
            row_data_ordered = {"text": t, "index": i}
            prompts_responses = []
            for _,prompt in enumerate(tqdm(prompts)):
                prompt_str = prompt.format(text=t)
                resp = self.bam_client.send_messages(prompt_str)[0]
                prompts_responses.append(resp[0].strip().replace('\n', '\\n').replace('\\n', '  \\n'))
            mixed_indices = list(range(len(prompts)))
            random.shuffle(mixed_indices)
            mixed_mapping = {}
            for i in range(len(prompts)):
                row_data_ordered[f"{prompt_types[i]}_prompt"] = prompts[i]
                row_data_ordered[f"{prompt_types[i]}_output"] = prompts_responses[i]
                mixed_mapping[mixed_indices[i]] = prompt_types[i]
            row_data_ordered["mixed_indices_mapping_to_prompt_type"] = mixed_mapping
            generated_ordered.append(row_data_ordered)
        random.shuffle((generated_ordered))
        return generated_ordered