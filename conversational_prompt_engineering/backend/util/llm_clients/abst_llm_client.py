import abc
import sys
import logging
import os
from enum import Enum
import dotenv

dotenv.load_dotenv()


class HumanRole(Enum):
    User = "user"
    Admin = "admin"



class AbstLLMClient:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.sent_words_count = 0
        self.received_words_count = 0

    def _get_env_var(self, var_name):
        val = os.environ.get(var_name)
        if not val:
            raise ValueError(f"{var_name} is not defined in your environment variables.")
        return val

    @abc.abstractmethod
    def prompt_llm(self, conversation, max_new_tokens=None):
        """
        return the list of replies.
        """
        raise NotImplementedError()

    def do_send_message(self, conversation, max_new_tokens):
        sys.tracebacklimit = 1000
        for i in [0,1]:
            try:
                res = self.prompt_llm(conversation, max_new_tokens)
                texts = [x.strip() for x in res]
                return texts
            except Exception as e:
                if i == 0:
                    logging.debug("ERROR Got API response exception", e)
                else:
                    logging.error("ERROR Got API response exception", e)
        sys.tracebacklimit = 0
        raise Exception("There is an error connecting to the LLM service. Either check your API key or try again in a few minutes.")

    def send_messages(self, conversation, max_new_tokens=None):
        def log_message(text, member_to_update):
            if isinstance(text, list):
                cnt = sum(len(x) for x in text)
            else:
                cnt = len(text.split())
            setattr(self, member_to_update, getattr(self, member_to_update) + cnt)
            logging.info(f"{member_to_update} = {getattr(self, member_to_update)} (added {cnt} in this turn)")
            return cnt
        sent_words = log_message(conversation, "sent_words_count")
        res = self.do_send_message(conversation, max_new_tokens)
        received_words = log_message(res[0], "received_words_count")
        stats_dict = {"sent words": sent_words, "received words": received_words}
        return res, stats_dict



