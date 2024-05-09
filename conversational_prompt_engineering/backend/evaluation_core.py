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

    def __init__(self, bam_api_key):
        with open("backend/bam_params.json", "r") as f:
            bam_params = json.load(f)
        bam_params['api_key'] = bam_api_key
        self.bam_client = BamGenerate(bam_params)

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

        generated_data_mixed, generated_data_ordered = self.summarize(prompts, texts)

        os.makedirs(out_dir, exist_ok=True)

        df = pd.DataFrame(generated_data_ordered)
        df.to_csv(os.path.join(out_dir, "evaluate_ordered.csv"), index=False)

        df = pd.DataFrame(generated_data_mixed)
        df.to_csv(os.path.join(out_dir, "evaluate_mixed.csv"), index=False)
        df = df[[c for c in df.columns if not c.endswith("prompt")]]
        df.to_csv(os.path.join(out_dir, "evaluate_mixed_hidden.csv"), index=False)

        logging.info(f"evaluation files saved to {out_dir}")

    def summarize(self, prompts, texts):
        generated_data_ordered = []
        generated_data_mixed = []
        for t in texts:
            row_data_ordered = {"text": t}
            row_data_mixed = {"text": t}
            prompts_responses = []
            for i, prompt in enumerate(tqdm(prompts)):
                prompt_str = prompt['prompt_ready_to_use']
                prompt_str_t = prompt_str.format(text=t)
                resp = self.bam_client.send_messages(prompt_str_t)[0]
                prompts_responses.append(resp)
            for i in range(len(prompts)):
                row_data_ordered[str(i) + "_prompt"] = prompts[i]['prompt']
                row_data_ordered[str(i)] = prompts_responses[i]
            mixed_indices = list(range(len(prompts)))
            random.shuffle(mixed_indices)
            for i in range(len(mixed_indices)):
                row_data_mixed[str(i) + "_prompt"] = prompts[mixed_indices[i]]['prompt']
                row_data_mixed[str(i)] = prompts_responses[mixed_indices[i]]
            generated_data_ordered.append(row_data_ordered)
            generated_data_mixed.append(row_data_mixed)
        return generated_data_mixed, generated_data_ordered


if __name__ == "__main__":
    args = parser.parse_args()
    evaluation = Evaluation(os.getenv('BAM_APIKEY'))
    evaluation.compare_prompts_within_conversation(prompts_path=args.prompts_path,
                                                   data_path=args.data_path,
                                                   out_dir=args.out_dir)