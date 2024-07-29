dataset_name_to_dir = {"Movie Reviews" : {"train":
                                            "./data/public/movie reviews/train.csv",
                                          "eval": "./data/public/movie reviews/eval.csv",
                                          "desc": "This dataset consists of movie reviews \npublished at https://www.rogerebert.com/."},
                       "Hotels and Restaurants": {"train": "./data/public/multiwoz/train.csv",
                                         "eval": "./data/public/multiwoz/test.csv",
                                         "eval_llm": "./data/public/multiwoz/test_full.csv",
                                         "desc": "This dataset consists of multi-turn dialogues \nabout hotel or restaurant reservation. "},
                       "Privacy Policies and Software Licenses": {"train":
                                             "./data/public/legal_plain_english/train.csv",
                                         "eval": "./data/public/legal_plain_english/eval.csv",
                                         "eval_llm": "./data/public/legal_plain_english/test_full.csv",
                                        "desc": "This dataset consists of passages from legal documents \ndiscussing privacy policies or software licenses."},
                       "IBM blog": {"train": "./data/public/ibm blog/train.csv",
                                                "eval": "./data/public/ibm blog/test.csv",
                                                 "desc": "This dataset contains blog entries from IBM blog."},
                       "climate blog": {"train": "./data/public/climate blog/train.csv",
                                    "eval": "./data/public/climate blog/test.csv",
                                    "desc": "This dataset consists of articles about climate related issues \npublished in https://www.climaterealityproject.org."},

                       "Reddit posts": {"train": "./data/public/tldr/train.csv",
                                        "eval": "./data/public/tldr/test.csv",
                                        "eval_llm": "./data/public/tldr/test_full.csv",
                                        "desc": "This dataset consists of posts from Reddit (TL;DR dataset)"},

                       "Restaurant reviews": {"train": "./data/public/lentricote_trip_advisor/train.csv",
                                        "eval": "./data/public/lentricote_trip_advisor/test.csv",
                                        "desc": "This dataset consists of reviewes of the restaurant\n \"L'entrecote\" in London, posted on the trip-advisor website"},

                       "Space Newsgroup": {"train": "./data/public/20_newsgroup/train.csv",
                            "eval": "./data/public/20_newsgroup/test.csv",
                            "eval_llm": "./data/public/20_newsgroup/test_full.csv",
                            "desc": "This is part of the 20 Newsgroups dataset, a collection of \nnewsgroup documents. The topic of the documents is Space."},

                       "Complaints on Credit Reporting": {"train": "./data/public/cfpb/train.csv",
                            "eval": "./data/public/cfpb/test.csv",
                            "eval_llm": "./data/public/cfpb/test_full.csv",
                            "desc": "This is part of the CFPB dataset that consists of complaints about\nconsumer financial products and services. The topic of the complaints\nis Credit Reporting."},

                       "Financial News on Acquisition": {"train": "./data/public/reuters/train.csv",
                            "eval": "./data/public/reuters/test.csv",
                            "eval_llm": "./data/public/reuters/test_full.csv",
                            "desc": "This is part of the Reuters dataset, which is a collection of\ndocuments with news articles. The topic of the articles\nis Acquisition."},
                       "Debate Speeches": {"train": "./data/public/debate_speeches/train.csv",
                                                          "eval": "./data/public/debate_speeches/test.csv",
                                                          "eval_llm": "./data/public/debate_speeches/test_full.csv",
                                                          "desc": "This is part of the Debate Speeches dataset.\nThe dataset contains manually-corrected transcripts \nof speeches recorded by expert debaters."},

                       "Wikipedia Animal pages": {"train": "./data/public/wiki_animals/train.csv",
                            "eval": "./data/public/wiki_animals/test.csv",
                            "eval_llm": "./data/public/wiki_animals/test_full.csv",
                            "desc": "This dataset consists of animal pages extracted from \nWikipedia"},

                       "Wikipedia Movie pages": {"train": "./data/public/wiki_movies/train.csv",
                            "eval": "./data/public/wiki_movies/test.csv",
                            "eval_llm": "./data/public/wiki_movies/test_full.csv",
                            "desc": "This dataset consists of movies pages extracted from \nWikipedia"},

                       }