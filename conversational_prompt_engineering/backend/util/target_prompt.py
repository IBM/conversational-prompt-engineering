import abc

class AbstPrompt():
    def __init__(self, prompt):
        self.prompt = prompt

    @abc.abstractmethod
    def update_text(self, text):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_display_format(self):
        raise NotImplementedError()

class StringPrompt(AbstPrompt):
    #Generated prompt contains the string {text}
    def __init__(self, prompt):
        super(StringPrompt, self).__init__(prompt)

    def update_text(self, text):
        return self.prompt.format(text=text)

    def get_display_format(self):
        return self.prompt


class ConversationPrompt(AbstPrompt):
    # The prompt contains a list of multi turn messages. One of them contains the string {text}
    def __init__(self, prompt):
        super(ConversationPrompt, self).__init__(prompt)

    def update_text(self, text):
        return [x for x in self.prompt[:-1]] + [{"role": self.prompt[-1]["role"], "content": self.prompt[-1]["content"].format(text=text)}]

    def get_display_format(self):
        return str(self.prompt)