import json
import os

from genai.schema import ChatRole

from conversational_prompt_engineering.util.bam import BamGenerate


class ConversationState:
    INTRODUCTION = 'introduction'
    CONFIRM_CHARACTERISTICS = 'confirm_characteristics'
    CONFIRM_PROMPT = 'confirm_prompt'
    CONFIRM_SUMMARY = 'confirm_summary'


def extract_delimited_text(txt, delim):
    if delim not in txt:
        return None
    begin = txt.index(delim) + len(delim)
    end = begin + txt[begin:].index(delim)
    return txt[begin:end]


class DoubleChatManager:
    def __init__(self, bam_api_key) -> None:
        with open("backend/bam_params.json", "r") as f:
            params = json.load(f)
        params['api_key'] = bam_api_key
        self.bam_client = BamGenerate(params)

        self.user_chat = None
        self.hidden_chat = None
        self.text_examples = None
        self.approved_prompts = None
        self.state = None

    def _add_msg(self, chat, role, msg):
        chat.append({'role': role, 'content': msg})

    def _add_usr_msg(self, msg):
        self._add_msg(self.user_chat, ChatRole.USER, msg)
        self._add_msg(self.hidden_chat, ChatRole.USER, msg)

    def _add_system_msg(self, msg):
        self._add_msg(self.hidden_chat, ChatRole.SYSTEM, msg)

    def _add_assistant_msg(self, messages, to_chat):
        chats = {
            'user': [self.user_chat],
            'hidden': [self.hidden_chat],
            'both': [self.user_chat, self.hidden_chat],
        }[to_chat]
        if isinstance(messages, str):
            messages = [messages]

        for msg in messages:
            for chat in chats:
                self._add_msg(chat, ChatRole.ASSISTANT, msg)

    def _get_assistant_response(self, chat=None):
        chat = chat or self.hidden_chat
        conversation = ''.join([f'\n<|{m["role"]}|>\n{m["content"]}\n' for m in chat])
        conversation += f'<|{ChatRole.ASSISTANT}|>'

        generated_texts = self.bam_client.send_messages(conversation)
        agent_response = ''
        for txt in generated_texts:
            if any([f'<|{r}|>' in txt for r in [ChatRole.SYSTEM, ChatRole.USER]]):
                agent_response += txt[: txt.index('<|')]
                break
            agent_response += txt
        return agent_response.strip()

    def _init_chats(self):
        self.user_chat = []
        self.hidden_chat = []
        self.text_examples = []
        self.approved_prompts = []

        self._add_system_msg("""You are an IBM prompt building assistant that helps the user build an instruction for a text summarization task. 
You will be interacting with two actors: system and user. The direct interaction will be only with system. 
The system will guide you through the stages necessary to build the prompt.
Please answer only the word 'understood' if you understand these instructions.  
""")
        resp = self._get_assistant_response()
        assert resp.lower().startswith('understood')

        self._add_system_msg("""Introduce yourself to the user, 
then ask if they can provide any insights about the texts to summarize and/or the summaries to produce, 
or maybe they have an example of the text, or even an example of an acceptable summary. 
This information can be helpful to figure out the texts domain, and required properties of the summary""")
        resp = self._get_assistant_response()
        self._add_assistant_msg(resp, 'both')
        self.state = ConversationState.INTRODUCTION

    def _extract_text_example(self):
        self._add_system_msg(
            'Did you obtain a text example from the user? If you did, write "yes" and the text enclosed in triple quotes (```). If no, just write "no"')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        if resp.lower().startswith('yes'):
            self.text_examples.append(extract_delimited_text(resp, "```"))
            return True
        return False

    def _confirm_characteristics(self):
        self._add_system_msg(
            "What is your current understanding of the input texts and the expected properties of the summaries?")
        resp = self._get_assistant_response()
        # keep only the first paragraph, the model can go on
        if '\n' in resp:
            resp = resp[: resp.index('\n')]
        self._add_assistant_msg(resp, 'hidden')
        self._add_system_msg('Please validate your suggestion with the user, and update it if necessary.')
        resp = self._get_assistant_response()
        self._add_assistant_msg(resp, 'both')
        self.state = ConversationState.CONFIRM_CHARACTERISTICS

    def _confirm_prompt(self, is_new):
        self._add_system_msg(
            'Build the summarization prompt based on your current understanding. Enclose the prompt text in triple quotes (```).')
        resp = self._get_assistant_response()
        prompt = extract_delimited_text(resp, '```')
        if is_new:
            self.approved_prompts.append(prompt)
        else:
            self.approved_prompts[-1] = prompt

        self._add_assistant_msg(prompt, 'hidden')
        self._add_system_msg('Please validate your suggestion with the user, and update it if necessary.')
        resp = self._get_assistant_response()
        self._add_assistant_msg(resp, 'both')
        self.state = ConversationState.CONFIRM_PROMPT

    def _suggestion_accepted(self):
        self._add_system_msg(
            'Has the user accepted your suggestion or corrected it? Answer either "accepted" or "corrected"')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question

        is_accepted = 'accepted' in resp.lower()
        is_corrected = 'corrected' in resp.lower()
        assert is_accepted != is_corrected
        return is_accepted

    def _evaluate_prompt(self):
        # TODO: handle the cases when there's no example yet, or multiple examples
        eval_chat = []
        self._add_msg(eval_chat, ChatRole.SYSTEM, self.approved_prompts[-1])
        example = self.text_examples[0]
        self._add_msg(eval_chat, ChatRole.USER, example)
        summary = self._get_assistant_response(eval_chat)

        self._add_system_msg(
            f'When the latest prompt was use for summarization of the example ```{example}``` it produced the result ```{summary}```')
        self._add_system_msg(f'Present these results to the user and ask if they want any changes.')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-2]  # remove the last messages

        self._add_assistant_msg(resp, 'both')
        self.state = ConversationState.CONFIRM_SUMMARY

    def process_user_input(self, user_message):
        if self.state is None:
            self._init_chats()
        elif user_message:
            self._add_usr_msg(user_message)

            if self.state in [ConversationState.INTRODUCTION]:
                self._extract_text_example()
                self._confirm_characteristics()

            elif self.state == ConversationState.CONFIRM_CHARACTERISTICS:
                if self._suggestion_accepted():
                    self._confirm_prompt(is_new=True)
                else:
                    self._confirm_characteristics()

            elif self.state == ConversationState.CONFIRM_PROMPT:
                if self._suggestion_accepted():
                    self._evaluate_prompt()
                else:
                    self._confirm_prompt(is_new=False)

            elif self.state == ConversationState.CONFIRM_SUMMARY:
                if self._suggestion_accepted():
                    self._add_assistant_msg('BYE', self.user_chat)
                    self.state = None
                else:
                    self._confirm_prompt(is_new=True)

        return self.user_chat
