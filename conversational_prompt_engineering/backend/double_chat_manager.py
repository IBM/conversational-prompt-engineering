import json
import os

from genai.schema import ChatRole

from conversational_prompt_engineering.util.bam import BamGenerate

SYSTEM_INSTRUCTIONS = """You are an IBM prompt building assistant that helps the user build an instruction for a text summarization task. 
You will be interacting with two actors: system and user. The direct interaction will be only with system. 
The system will guide you through the stages necessary to build the prompt.
Please answer only the word 'understood' if you understand these instructions.  
"""

INTRODUCE = """Introduce yourself to the user, 
then ask if they can provide any insights about the texts to summarize and/or the summaries to produce, 
or maybe they have an example of the text, or even an example of an acceptable summary. 
This information can be helpful to figure out the texts domain, and required properties of the summary"""

ASK_EXAMPLE_1 = """In order to build a good prompt, you need to figure out the required properties of a summary. 
If the user has a typical example of a text to summarize, you could deduce these properties yourself.
Ask the user for an example."""


class ConversationState:
    INTRODUCTION = 'introduction'
    VALIDATE_CONCLUSIONS = 'validate_conclusions'


class DoubleChatManager:
    def __init__(self) -> None:
        with open("backend/params.json", "r") as f:
            params = json.load(f)
        params['api_key'] = os.getenv("BAM_APIKEY")
        self.bam_client = BamGenerate(params)

        self.user_chat = []
        self.hidden_chat = []
        self.text_examples = []
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
        self._add_system_msg(SYSTEM_INSTRUCTIONS)
        resp = self._get_assistant_response()
        if resp.lower().startswith('understood'):
            self._add_system_msg(INTRODUCE)
            resp = self._get_assistant_response()
            self._add_assistant_msg(resp, 'both')
            self.state = ConversationState.INTRODUCTION

    def _extract_text_example(self):
        self._add_system_msg("Did you obtain a text example from the user? If you did, write 'yes' and the text enclosed in triple quotes. If no, just write 'no'")
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        if resp.lower().startswith('yes'):
            delim = "```"
            begin = resp.index(delim) + len(delim)
            end = begin + resp[begin:].index(delim)
            self.text_examples.append(resp[begin:end])
            return True
        return False

    def _next_move(self):
        if self.state in [ConversationState.INTRODUCTION]:
            self._extract_text_example()
            self._add_system_msg("What can you say so far about the texts that should be summarized and suggested characteristics of the summaries?")
            resp = self._get_assistant_response()
            self._add_assistant_msg(resp, 'hidden')
            self._add_system_msg('Please validate your conclusions with the user, and update them if necessary.')
            resp = self._get_assistant_response()
            self._add_assistant_msg(resp, 'both')
            self.state = ConversationState.VALIDATE_CONCLUSIONS

        elif self.state == ConversationState.VALIDATE_CONCLUSIONS:
            # ask if we are good and
            print('over here')

    def process_user_input(self, user_message):
        if self.state is None:
            self._init_chats()
        elif user_message:
            self._add_usr_msg(user_message)
            self._next_move()

        return self.user_chat
