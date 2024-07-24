from conversational_prompt_engineering.data.main_dataset_name_to_dir import dataset_name_to_dir as all_datasets

relevant_datasets = ["Reddit posts", "Space Newsgroup", "Debate Speeches", "Wikipedia Animal pages", "Wikipedia Movie pages"]
dataset_name_to_dir = {x:all_datasets[x] for x in relevant_datasets}

