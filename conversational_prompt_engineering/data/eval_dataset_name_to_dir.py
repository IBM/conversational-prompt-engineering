# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

from conversational_prompt_engineering.data.main_dataset_name_to_dir import dataset_name_to_dir as all_datasets

relevant_datasets = ["Reddit posts", "Space Newsgroup", "Debate Speeches", "Wikipedia Animal pages", "Wikipedia Movie pages"]
dataset_name_to_dir = {x:all_datasets[x] for x in relevant_datasets}

