import json
import os
import logging
import time

import pandas as pd
from genai.schema import ChatRole

from conversational_prompt_engineering.util.bam import BamGenerate

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class ConversationState:
    INTRODUCTION = 'introduction'
    CONFIRM_CHARACTERISTICS = 'confirm_characteristics'
    CONFIRM_PROMPT = 'confirm_prompt'
    PROCESS_TEXTS = 'process_texts'
    PROCESS_RESPONSES = 'process_responses'
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
        self.validated_example_idx = None
        self.approved_prompts = None
        self.state = None
        self.user_has_more_texts = True
        self.timing_report = []

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

    def _get_assistant_response(self, chat=None, max_new_tokens=None):
        chat = chat or self.hidden_chat
        conversation = ''.join([f'\n<|{m["role"]}|>\n{m["content"]}\n' for m in chat])
        conversation += f'<|{ChatRole.ASSISTANT}|>'
        start_time = time.time()
        generated_texts = self.bam_client.send_messages(conversation, max_new_tokens=max_new_tokens)
        elapsed_time = time.time()-start_time
        timing_dict = {"state": self.state, "context_length": len(conversation), "output_length": sum([len(gt) for gt in generated_texts]), "time": elapsed_time}
        logging.info(timing_dict)
        self.timing_report.append(timing_dict)
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
        self.validated_example_idx = 0
        self.approved_prompts = []

        self._add_system_msg("""You are an IBM prompt building assistant that helps the user build an instruction for a text summarization task. 
You will be interacting with two actors: system and user. The direct interaction will be only with system. 
The system will guide you through the stages necessary to build the prompt.
Please answer only the word 'understood' if you understand these instructions.
""")
        resp = self._get_assistant_response(max_new_tokens=10)
        self._add_assistant_msg(resp, 'hidden')
        assert resp.lower().startswith('understood')
        logging.info("initializing chat")

        self.add_prompt( 'Summarize the following text in 2-3 sentences, highlighting the main ideas and key points.')
        self._add_system_msg(f"""Thanks. Now, introduce yourself to the user. 
The initial prompt you suggest the user for summarization is: {self.approved_prompts[-1]['prompt']} 
Please validate your suggestion with the user, and update it if necessary.""")

        resp = self._get_assistant_response(max_new_tokens=200)
        self._add_assistant_msg(resp, 'both')

    def add_prompt(self, prompt, is_new=True):
        if is_new:
            self.approved_prompts.append({'prompt': prompt,  'stage': self.state})
        else:
            self.approved_prompts[-1]['prompt'] = prompt

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
            'Did you obtain a text example from the user in the last message? If you did, write "yes" and the text enclosed in triple quotes (```). If no, just write "no"')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        if resp.lower().startswith('yes'):
            example = extract_delimited_text(resp, "```")
            if example not in self.text_examples:
                self.text_examples.append(example)
            return True
        return False

    def _confirm_characteristics(self):
        self._add_system_msg(
            "What is your current understanding of the input texts and the expected properties of the summaries?")
        resp = self._get_assistant_response(max_new_tokens=200)
        # keep only the first paragraph, the model can go on
        if '\n' in resp:
            resp = resp[: resp.index('\n')]
        self._add_assistant_msg(resp, 'hidden')
        self._add_system_msg('Please validate your suggestion with the user, and update it if necessary.')
        resp = self._get_assistant_response(max_new_tokens=200)
        self._add_assistant_msg(resp, 'both')
        self.state = ConversationState.CONFIRM_CHARACTERISTICS

    def _confirm_prompt(self, is_new):
        self._add_system_msg(
            'Build the summarization prompt based on your current understanding (only the instruction). Enclose the prompt in triple quotes (```).')
        resp = self._get_assistant_response(max_new_tokens=200)
        prompt = extract_delimited_text(resp, '```')
        prompt = prompt.strip("\"")

        self.add_prompt(prompt, is_new=is_new or len(self.approved_prompts) == 1)

        self._add_assistant_msg(prompt, 'hidden')
        self._add_system_msg('Please validate your suggestion with the user, and update it if necessary.')
        resp = self._get_assistant_response(max_new_tokens=200)
        self._add_assistant_msg(resp, 'both')

    def _suggestion_accepted(self):
        self._add_system_msg(
            'Has the user accepted your suggestion or corrected it? Answer either "accepted" or "corrected"')
        resp = self._get_assistant_response(max_new_tokens=10)
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question

        is_accepted = 'accepted' in resp.lower()
        is_corrected = 'corrected' in resp.lower()
        assert is_accepted != is_corrected
        return is_accepted

    def _ask_for_text(self):
        self._add_system_msg("""Ask the user to provide up to three typical examples of the texts he or she wish to summarize. 
This will help you get familiar with the domain and the flavor of the user's documents. Mention to the user that they need to share three examples one at a time, but at each stage they can indicate that they do not have anymore examples to share.
Do not share your insights until you have collected all examples.""")
        resp = self._get_assistant_response(max_new_tokens=200)
        # self.hidden_chat = self.hidden_chat[:-1]
        self._add_assistant_msg(resp, 'both')
        return resp

    def _do_nothing(self):
        resp = self._get_assistant_response(max_new_tokens=100)
        self._add_assistant_msg(resp, 'both')
        return resp

    def _has_more_texts(self):
        if self.user_has_more_texts:
            self._add_system_msg(
                'Has the user indicated they finished sharing texts, or not? Answer either "finished" or "not"')
            resp = self._get_assistant_response(max_new_tokens=10)
            self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
            self.user_has_more_texts = "not" in resp.lower()
        return self.user_has_more_texts

    def _ask_text_questions(self):
        self._add_system_msg(
            """Now, based on these examples, ask the user up to 3 relevant questions about his summary preferences. 
Please do not ask questions that refer to a specific example. 
Ask the user to answer all the questions at the same turn.""")
        resp = self._get_assistant_response()
        # self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        self._add_assistant_msg(resp, 'both')

    def _evaluate_prompt(self):
        # TODO: handle the cases when there's no example yet, or multiple examples
        eval_chat = []
        self._add_msg(eval_chat, ChatRole.SYSTEM, self.approved_prompts[-1]['prompt'])
        example = self.text_examples[self.validated_example_idx]
        self._add_msg(eval_chat, ChatRole.USER, example)
        summary = self._get_assistant_response(eval_chat)
        if self.approved_prompts[-1]['prompt'] not in self.summaries:
            self.summaries[self.approved_prompts[-1]['prompt']] = {}
        self.summaries[self.approved_prompts[-1]['prompt']][example] = summary
        self._add_system_msg(
            f'When the latest prompt was used for summarization of the example ```{example}``` it produced the result ```{summary}```')
        self._add_system_msg(f'Present these results to the user, mention they are based on the text of example no. {(self.validated_example_idx + 1)} shared by the user, and ask if they want any changes.')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-2]  # remove the last messages

        self._add_assistant_msg(resp, 'both')

    def _share_prompt_and_save(self):
        prompt = self.approved_prompts[-1]['prompt']
        temp_chat = []
        self._add_msg(temp_chat, ChatRole.USER,
                      'Suggest a name for the following summarization prompt. '
                      'The name should be short and descriptive, it will be used as a title in the prompt library. '
                      f'Enclose the suggested name in triple quotes (```). The prompt is "{prompt}"')
        resp = self._get_assistant_response(temp_chat)
        name = extract_delimited_text(resp, "```").strip().replace('"', '').replace(" ", "_")

        if self.approved_prompts[-1]['prompt'] in self.summaries:
            prompt += "\n\n"
            texts_and_summaries = self.summaries[self.approved_prompts[-1]['prompt']]
            prompt += "\n\n".join(["Text: " + t + "\n\nSummary: " + s for t, s in texts_and_summaries.items()])

        self.add_prompt(prompt, is_new=True)

        final_msg = "Here is the final prompt: \n\n" + prompt
        saved_name, bam_url = self.bam_client.save_prompt(name, prompt)
        final_msg += f'\n\nThis prompt has been saved to your prompt Library under the name "{saved_name}". ' \
                     f'You can try it in the [BAM Prompt Lab]({bam_url})'
        self._add_assistant_msg(final_msg, 'user')
        self.state = None

        # saving prompts
        out_dir = f"_out/{name}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "prompts.json"), "w") as f:
            json.dump(self.approved_prompts, f)
        logging.info(f"prompts saved for evaluation to {os.path.join(out_dir, 'prompts.json')}")

    def _no_texts(self):
        return len(self.text_examples) == 0

    def print_timing_report(self):
        df = pd.DataFrame(self.timing_report)
        logging.info(df)
        logging.info(f"Average processing time: {df['time'].mean()}")
        self.timing_report = sorted(self.timing_report, key=lambda row: row['time'])
        logging.info(f"Highest processing time: {self.timing_report[-1]}")
        logging.info(f"Lowest processing time: {self.timing_report[0]}")

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
                    if self.user_has_more_texts:
                        logging.info(f"asking for text to summarize")
                        self._ask_for_text()
                        self.state = ConversationState.PROCESS_TEXTS
                    else:
                        logging.info(f"user gave {len(self.text_examples)} text examples. ({self.validated_example_idx})")
                        self._evaluate_prompt()
                        self.state = ConversationState.CONFIRM_SUMMARY
                else:
                    logging.info("user did not approve")
                    self._confirm_prompt(is_new=False)

            elif self.state == ConversationState.PROCESS_TEXTS:
                example_extracted = self._extract_text_example()
                if example_extracted:
                    logging.info("extracted text from user")
                if self._has_more_texts():
                    self._do_nothing()
                else:
                    self._ask_text_questions()
                    self.state = ConversationState.PROCESS_RESPONSES

            elif self.state == ConversationState.PROCESS_RESPONSES:
                self._confirm_prompt(is_new=True)
                self.state = ConversationState.CONFIRM_PROMPT

            elif self.state == ConversationState.EVALUATE_PROMPT:
                logging.info("extracted text from user")
                self._evaluate_prompt()
                self.state = ConversationState.CONFIRM_SUMMARY

            elif self.state == ConversationState.CONFIRM_SUMMARY:
                if self._suggestion_accepted():
                    logging.info(f"user approved the summary of one example. ({self.validated_example_idx})")
                    self.validated_example_idx += 1
                    if self.validated_example_idx == len(self.text_examples):
                        logging.info("user approved all the summaries so ending the chat and sharing the final prompt")
                        self._share_prompt_and_save()
                        self.print_timing_report()
                    else:
                        self._evaluate_prompt()
                else:
                    logging.info("user did not approve so making changes to prompt")
                    self._confirm_prompt(is_new=True)
                    self.state = ConversationState.CONFIRM_PROMPT

        return self.user_chat[-1]['content']


