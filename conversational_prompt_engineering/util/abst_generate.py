import abc

import logging

class AbstGenerate:
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.sent_words_count = 0
        self.received_words_count = 0

    @abc.abstractmethod
    def do_send_messages(self, conversation, max_new_tokens=None):
        raise NotImplementedError()


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
        res = self.do_send_messages(conversation, max_new_tokens)
        received_words = log_message(res[0], "received_words_count")
        stats_dict = {"sent words": sent_words, "received words": received_words}
        return res, stats_dict



