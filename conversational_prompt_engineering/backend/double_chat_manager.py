import json
import os
import logging

from genai.schema import ChatRole

from conversational_prompt_engineering.util.bam import BamGenerate

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class ConversationState:
    INTRODUCTION = 'introduction'
    CONFIRM_CHARACTERISTICS = 'confirm_characteristics'
    CONFIRM_PROMPT = 'confirm_prompt'
    EVALUATE_PROMPT = 'evaluate_prompt'
    CONFIRM_SUMMARY = 'confirm_summary'


def extract_delimited_text(txt, delim):
    if delim not in txt:
        return txt #None
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
        self.summaries = {}
        self.approved_prompts = None
        self.state = None

    def _load_admin_params(self):
        with open("backend/admin_params.json", "r") as f:
            params = json.load(f)
        return params

    def _add_msg(self, chat, role, msg):
        chat.append({'role': role, 'content': msg})

    def _add_usr_msg(self, msg):
        logging.info(f"got input from user: {msg}")
        self._add_msg(self.user_chat, ChatRole.USER, msg)
        self._add_msg(self.hidden_chat, ChatRole.USER, msg)

    def _add_system_msg(self, msg):
        logging.info(f"adding system msg: {msg}")
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
        logging.info(f"got response from model: {agent_response}")
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
        self._add_assistant_msg(resp, 'hidden')
        assert resp.lower().startswith('understood')
        self._add_assistant_msg(resp, 'hidden')
        logging.info("initializing chat")

        self.approved_prompts.append(
            'Summarize the following text in 2-3 sentences, highlighting the main ideas and key points.')
        self._add_system_msg(f"""Thanks. Now, introduce yourself to the user. 
        The initial prompt you suggest the user for summarization is: {self.approved_prompts[-1]} 
        Please validate your suggestion with the user, and update it if necessary.""")

        resp = self._get_assistant_response()
        self._add_assistant_msg(resp, 'both')


    def _got_introduction_responses(self):
        self._add_system_msg(
            'Did User already respond to all of the questions you should be asking? Answer yes or no ONLY.')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        if resp.lower().startswith('yes'):
            return True
        return False

    def _continue_ask_introduction_questions(self):
        self._add_system_msg(
            'Ok, please ask the user the next question.')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]
        self._add_assistant_msg(resp, 'both')

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
            'Build the summarization prompt based on your current understanding (only the instruction). Enclose the prompt in triple quotes (```).')
        resp = self._get_assistant_response()
        prompt = extract_delimited_text(resp, '```')
        prompt = prompt.strip("\"")
        if is_new:
            self.approved_prompts.append(prompt)
        else:
            self.approved_prompts[-1] = prompt

        self._add_assistant_msg(prompt, 'hidden')
        self._add_system_msg('Please validate your suggestion with the user, and update it if necessary.')
        resp = self._get_assistant_response()
        self._add_assistant_msg(resp, 'both')

    def _suggestion_accepted(self):
        self._add_system_msg(
            'Has the user accepted your suggestion or corrected it? Answer either "accepted" or "corrected"')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question

        is_accepted = 'accepted' in resp.lower()
        is_corrected = 'corrected' in resp.lower()
        assert is_accepted != is_corrected
        return is_accepted

    def _ask_for_text(self):
        self._add_system_msg(
            f'Please ask user to share a text example.')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]
        self._add_assistant_msg(resp, 'both')
        self.state = ConversationState.EVALUATE_PROMPT
        return resp

    def _evaluate_prompt(self):
        # TODO: handle the cases when there's no example yet, or multiple examples
        eval_chat = []
        self._add_msg(eval_chat, ChatRole.SYSTEM, self.approved_prompts[-1])
        example = self.text_examples[0]
        self._add_msg(eval_chat, ChatRole.USER, example)
        summary = self._get_assistant_response(eval_chat)
        if self.approved_prompts[-1] not in self.summaries:
            self.summaries[self.approved_prompts[-1]] = {}
        self.summaries[self.approved_prompts[-1]][example] = summary
        self._add_system_msg(
            f'When the latest prompt was used for summarization of the example ```{example}``` it produced the result ```{summary}```')
        self._add_system_msg(f'Present these results to the user and ask if they want any changes.')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-2]  # remove the last messages

        self._add_assistant_msg(resp, 'both')

    def _share_prompt(self):
        prompt = self.approved_prompts[-1]
        if self.approved_prompts[-1] in self.summaries:
            prompt += "\n\n"
            texts_and_summaries = self.summaries[self.approved_prompts[-1]]
            prompt += "\n\n".join(["Text: " + t + "\n\nSummary: " + s for t, s in texts_and_summaries.items()])
        self._add_assistant_msg("Here is the final prompt: \n\n" + prompt, 'both')
        self.state = None

    def _no_texts(self):
        return len(self.text_examples) == 0

    def process_user_input(self, user_message):
        if self.state is None:
            self._init_chats()
            self.state = ConversationState.CONFIRM_PROMPT
        elif user_message:
            logging.info(f"in {self.state}")
            self._add_usr_msg(user_message)

            # if self.state in [ConversationState.INTRODUCTION]:
            #     if self._got_introduction_responses():
            #         logging.info("in introduction, "
            #                      "user shared info or text example so moving to confirm_characteristics")
            #         self._confirm_prompt(is_new=True)  # self._confirm_characteristics()
            #     else:
            #         self._continue_ask_introduction_questions()

            # if self.state == ConversationState.CONFIRM_CHARACTERISTICS:
            #     if self._suggestion_accepted():
            #         logging.info("in confirm_characteristics, user approved them "
            #                      "so moving to confirm_prompt")
            #         self._confirm_prompt(is_new=True)
            #     else:
            #         logging.info(
            #             "in confirm_characteristics, user did not approve them so staying")
            #         self._confirm_characteristics()

            if self.state == ConversationState.CONFIRM_PROMPT:
                if self._suggestion_accepted():
                    logging.info("user approved it")
                    if self._no_texts():
                        logging.info(f"asking for text to summarize")
                        self._ask_for_text()
                        self.state = ConversationState.EVALUATE_PROMPT
                    else:
                        logging.info(f"text exists, summarizing it")
                        self._evaluate_prompt()
                        self.state = ConversationState.CONFIRM_SUMMARY
                else:
                    logging.info("user did not approve")
                    self._confirm_prompt(is_new=False)

            elif self.state == ConversationState.EVALUATE_PROMPT:
                if self._no_texts():
                    if self._extract_text_example():
                        logging.info("extracted text from user")
                        self._evaluate_prompt()
                        self.state = ConversationState.CONFIRM_SUMMARY
                    else:
                        logging.info("user did not share an example")
                        self._ask_for_text()

            elif self.state == ConversationState.CONFIRM_SUMMARY:
                if self._suggestion_accepted():
                    logging.info("user approved the summary so ending the chat")
                    self._share_prompt()
                else:
                    logging.info("user did not approve so making changes to prompt")
                    self._confirm_prompt(is_new=True)
                    self.state = ConversationState.CONFIRM_PROMPT

        return self.user_chat[-1]['content']


