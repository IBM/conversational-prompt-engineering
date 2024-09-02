# (c) Copyright contributors to the conversational-prompt-engineering project

# LICENSE: Apache License 2.0 (Apache-2.0)
# http://www.apache.org/licenses/LICENSE-2.0

dataset_name_to_dir = {  "Microsoft Support" : {"train":
                                            "./data/TLS/Microsoft Support/train.csv",
                                            "eval": "./data/TLS/Microsoft Support/test.csv",
                                            "eval_llm": "./data/TLS/Microsoft Support/test_full.csv",
                                            "desc": "Tickets related to Microsoft Support."},
                         "Storwize" : {"train":
                                            "./data/TLS/Storwize/train.csv",
                                            "eval": "./data/TLS/Storwize/test.csv",
                                            "eval_llm": "./data/TLS/Storwize/test_full.csv",
                                            "desc": "Tickets related to Storwize."},
                          "Linux Support" : {"train":
                                            "./data/TLS/Linux Support/train.csv",
                                            "eval": "./data/TLS/Linux Support/test.csv",
                                            "eval_llm": "./data/TLS/Linux Support/test_full.csv",
                                            "desc": "Tickets related to Linux Support."}
                       }