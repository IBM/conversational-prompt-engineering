dataset_name_to_dir = {"Movie Reviews" : {"train":
                                              "./data/public/movie reviews/train.csv",
                                          "eval": "./data/public/movie reviews/eval.csv",
                                          "desc": "This dataset consists of movie reviews \npublished at https://www.rogerebert.com/."},
                       "Hotels and Restaurants": {"train":
                                             "./data/public/multiwoz/train.csv",
                                         "eval": "./data/public/multiwoz/test.csv",
                                         "desc": "This dataset consists of multi-turn dialogues \nabout hotel or restaurant reservation. "},
                        "Privacy Policies and Software Licenses": {"train":
                                             "./data/public/legal_plain_english/train.csv",
                                         "eval": "./data/public/legal_plain_english/eval.csv",
                                        "desc": "This dataset consists of passages from legal documents \ndiscussing privacy policies or software licenses."},
                       "IBM blog": {"train": "./data/public/ibm blog/train.csv",
                                                "eval": "./data/public/ibm blog/test.csv",
                                                 "desc": "This dataset contains blog entries from IBM blog."},
                       "climate blog": {"train": "./data/public/climate blog/train.csv",
                                    "eval": "./data/public/climate blog/test.csv",
                                    "desc": "This dataset consists of articles about climate related issues \npublished in https://www.climaterealityproject.org."},

                       "Reddit posts": {"train": "./data/public/tldr/train.csv",
                                        "eval": "./data/public/tldr/test.csv",
                                        "desc": "This dataset consists of posts from Reddit (TL;DR dataset)"},

                       "Restaurant reviews": {"train": "./data/public/lentricote_trip_advisor/train.csv",
                                        "eval": "./data/public/lentricote_trip_advisor/test.csv",
                                        "desc": "This dataset consists of reviewes of the restaurant\n \"L'entrecote\" in London, posted on the trip-advisor website"},

                       "20 Newsgroups - Space": {"train": "./data/public/20_newsgroup/train.csv",
                            "eval": "./data/public/20_newsgroup/test.csv",
                            "eval_llm": "./data/public/20_newsgroup/test_full.csv",
                            "desc": "This dataset is a collection newsgroup documents.\nThe topic of the documents is 'space'"},

                       "Consumer Financial Protection Bureau - Credit Reporting": {"train": "./data/public/cfpb/train.csv",
                            "eval": "./data/public/cfpb/test.csv",
                            "eval_llm": "./data/public/cfpb/test_full.csv",
                            "desc": "This dataset consists of complaints about\nconsumer financial products and services.\nThe topic of the complaints is 'credit reporting'."},

                       "Reuters Financial News - Acquisition": {"train": "./data/public/reuters/train.csv",
                            "eval": "./data/public/reuters/test.csv",
                            "eval_llm": "./data/public/reuters/test_full.csv",
                            "desc": "This dataset is a collection of documents with news articles.\nThe topic of the articles is 'acquisition'."},

                       }