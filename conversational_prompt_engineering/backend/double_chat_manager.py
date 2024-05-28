import json
import os
import logging
import random
import time

import pandas as pd
from genai.schema import ChatRole

from conversational_prompt_engineering.util.bam import BamGenerate

BASELINE_PROMPT = 'Summarize the following text in 2-3 sentences, highlighting the main ideas and key points.'

NUM_USER_EXAMPLES = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class ConversationState:
    INITIALIZING = 'initializing'
    INTRODUCTION = 'introduction'
    CONFIRM_CHARACTERISTICS = 'confirm_characteristics'
    CONFIRM_PROMPT = 'confirm_prompt'
    PROCESS_TEXTS = 'process_texts'
    PROCESS_RESPONSES = 'process_responses'
    EVALUATE_PROMPT = 'evaluate_prompt'
    CONFIRM_SUMMARY = 'confirm_summary'
    DONE = 'done'


LLAMA_END_OF_MESSAGE = "<|eot_id|>"

LLAMA_START_OF_INPUT = '<|begin_of_text|>'


def build_few_shot_prompt(prompt, texts_and_summaries, model_id):
    if 'llama' in model_id:
        return build_few_shot_prompt_llama(prompt, texts_and_summaries)
    elif 'mixtral' in model_id:
        return build_few_shot_prompt_mixtral(prompt, texts_and_summaries)
    else:
        raise Exception(f"model {model_id} not supported")


def build_few_shot_prompt_mixtral(prompt, texts_and_summaries):
    prompt += "\n\n"
    if len(texts_and_summaries) > 0:
        if len(texts_and_summaries) > 1:  # we already have at least two approved summary examples
            prompt += "Here are some typical text examples and their corresponding summaries."
        else:
            prompt += "Here is an example of a typical text and its summary."
        for item in texts_and_summaries:
            text = item['text']
            summary = item['summary']
            prompt += f"\n\nText: {text}\n\nSummary: {summary}"
        prompt += "\n\nNow, please summarize the following text.\n\n"
    prompt += "Text: {text}\n\nSummary: "
    return prompt


def build_few_shot_prompt_llama(prompt, texts_and_summaries):
    summary_prefix = "Here is a summary of the text:"
    prompt = LLAMA_START_OF_INPUT + _get_llama_header(ChatRole.USER) + "\n\n" + prompt + "\n\n"
    if len(texts_and_summaries) > 0:
        if len(texts_and_summaries) > 1:  # we already have at least two approved summary examples
            prompt += "Here are some typical text examples and their corresponding summaries."
        else:
            prompt += "Here is an example of a typical text and its summary."
        for i, item in enumerate(texts_and_summaries):
            if i > 0:
                prompt += _get_llama_header(ChatRole.USER)
            text = item['text']
            summary = item['summary']
            prompt += f"\n\nText: {text}\n\n{LLAMA_END_OF_MESSAGE}" \
                      f"{_get_llama_header(ChatRole.ASSISTANT)}{summary_prefix}{summary}{LLAMA_END_OF_MESSAGE}"
        prompt += _get_llama_header(ChatRole.USER) + "\n\nNow, please summarize the following text.\n\n"
    prompt += "Text: {text}\n\n" + LLAMA_END_OF_MESSAGE + _get_llama_header(ChatRole.ASSISTANT) + summary_prefix
    return prompt


def _get_llama_header(role):
    return "<|start_header_id|>" + role + "<|end_header_id|>"


def extract_delimited_text(txt, delims):
    if type(delims) is str:
        delims = [delims]
    for delim in delims:
        if delim in txt:
            begin = txt.index(delim) + len(delim)
            end = begin + txt[begin:].index(delim)
            return txt[begin:end]
    return txt  # delims not found in text


class DoubleChatManager:
    def __init__(self, bam_api_key, model) -> None:
        with open("backend/bam_params.json", "r") as f:
            params = json.load(f)
        logging.info(f"selected {model}")
        bam_params = params['models'][model]
        bam_params['api_key'] = bam_api_key
        bam_params['api_endpoint'] = params['api_endpoint']
        self.bam_client = BamGenerate(bam_params)

        self.user_chat = []
        self.hidden_chat = []
        self.text_examples = []

        self.approved_prompts = []
        self.approved_summaries = []
        self.validated_example_idx = 0
        self.state = None

        self.user_has_more_texts = True
        self.enable_upload_file = True
        self.timing_report = []

    def _load_admin_params(self):
        with open("backend/admin_params.json", "r") as f:
            params = json.load(f)
        return params

    def _add_msg(self, chat, role, msg):
        chat.append({'role': role, 'content': msg})

    def add_user_message(self, msg):
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

    def _format_chat(self, chat):
        if 'mixtral' in self.bam_client.parameters['model_id']:
            return ''.join([f'\n<|{m["role"]}|>\n{m["content"]}\n' for m in chat]) + f'<|{ChatRole.ASSISTANT}|>'
        elif 'llama' in self.bam_client.parameters['model_id']:
            msg_str = LLAMA_START_OF_INPUT
            for m in chat:
                msg_str += _get_llama_header(m['role']) + "\n\n" + m['content'] + LLAMA_END_OF_MESSAGE
            msg_str += _get_llama_header(ChatRole.ASSISTANT)
            return msg_str
        else:
            raise Exception(f"model {self.bam_client.parameters['model_id']} not supported")

    def _get_assistant_response(self, chat=None, max_new_tokens=None):
        chat = chat or self.hidden_chat
        conversation = self._format_chat(chat)
        start_time = time.time()
        generated_texts = self.bam_client.send_messages(conversation, max_new_tokens=max_new_tokens)
        elapsed_time = time.time() - start_time
        timing_dict = {"state": self.state, "context_length": len(conversation),
                       "output_length": sum([len(gt) for gt in generated_texts]), "time": elapsed_time}
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
        self._add_system_msg(
            "You are an IBM prompt building assistant that helps the user build an instruction for a text summarization task. "
            "You will be interacting with two actors: system and user. The direct interaction will be only with system. "
            "The system will guide you through the stages necessary to build the prompt. "
            "Please answer only the word 'understood' if you understand these instructions. "
        )
        resp = self._get_assistant_response(max_new_tokens=10)
        self._add_assistant_msg(resp, 'hidden')
        assert resp.lower().startswith('understood')
        logging.info("initializing chat")

        self._add_system_msg(
            "Thanks. "
            "Now, introduce yourself to the user, and present the following flow (do not act on this flow, just present it to the user): "
            "1. You'll agree on an initial prompt based on some unlabeled data."
            "2. You'll then refine the prompt based on the user's feedback on model outputs."
            "3. You'll share the final few-shot prompt."
            "\nMention to the user that after a prompt has been built, the user can evaluate it by clicking on Evaluate on the side-bar. "
            "\nThen, suggest the user to select a dataset from our catalog, or to upload a csv file, where the first column contains the text inputs. "
            "\nIf the user doesn't provide any evaluation data they can mention that in their response, and you'll proceed without it."
        )
        resp = self._get_assistant_response(max_new_tokens=200)
        self._add_assistant_msg(resp, 'both')

    def _add_prompt(self, prompt, is_new=True):
        prompt = prompt.strip("\n")

        if is_new:
            self.approved_prompts.append({'prompt': prompt, 'stage': self.state})
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

    def _need_clarification_from_the_user(self):
        self._add_system_msg(
            "Do you need any clarification on the user\'s respond? "
            "or maybe you are missing some details in understanding the user preferences? "
            "or maybe the user is not happy with the prompt you suggested but they do not say why? "
            "Answer yes or no ONLY."
        )
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        if resp.lower().startswith('yes'):
            return True
        return False

    def _ask_clarification_question(self):
        self._add_system_msg(
            'Ok, please ask a clarification question if needed.')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]
        self._add_assistant_msg(resp, 'both')

    def _extract_text_example(self):
        self._add_system_msg(
            'Did you obtain a text example from the user in the last message? answer "yes" or "no"')
        resp = self._get_assistant_response()
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        if resp.lower().startswith('yes'):
            example = self.user_chat[-1]['content']
            if "".join(example.split()) not in ["".join(ex.split()) for ex in self.text_examples]:
                self.text_examples.append(example)
                logging.info(f"Extracted text examples ({len(self.text_examples)}): {self.text_examples}")
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
        if 'llama' in self.bam_client.parameters['model_id']:
            self._add_system_msg(
                'Build the summarization prompt based on your current understanding (only the instruction). '
                'Do not add any additional information besides the prompt.'
            )
            resp = self._get_assistant_response(max_new_tokens=200)
            prompt = resp
        else:  # mixtral
            self._add_system_msg(
                'Build the summarization prompt based on your current understanding (only the instruction). '
                'Enclose the prompt in triple quotes (```).'
            )
            resp = self._get_assistant_response(max_new_tokens=200)
            prompt = extract_delimited_text(resp, ['```', '"""'])
            prompt = prompt.strip("\"")

        self._add_prompt(prompt, is_new=is_new)
        logging.info(f"added prompt: {prompt} | prompt is {'new' if is_new else 'corrected'}")
        self._add_assistant_msg(prompt, 'hidden')
        self._add_system_msg('Please validate your suggested prompt with the user, and update it if necessary.')
        resp = self._get_assistant_response(max_new_tokens=200)
        self._add_assistant_msg(resp, 'both')

    def _user_asked_for_correction(self):
        self._add_system_msg(
            'Has the user asked for a correction or a modification of the suggested prompt in the last message? answer "yes" or "no"')
        resp = self._get_assistant_response(max_new_tokens=50)
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        if resp.lower().startswith('yes'):
            return True
        return False

    def _prompt_suggestion_accepted(self):
        self._add_system_msg(
            'Has the user accepted the prompt you suggested? Answer "yes" or "no"')
        resp = self._get_assistant_response(max_new_tokens=50)
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        if resp.lower().startswith('yes'):
            return True
        return False

    def _summary_suggestion_corrected(self):
        self._add_system_msg(
            'Has the user asked for a correction or a modification of the suggested summary? Answer "yes" or "no"')
        resp = self._get_assistant_response(max_new_tokens=50)
        self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        if resp.lower().startswith('yes'):
            return True
        return False

    def _ask_for_text(self):
        self._add_system_msg(
            "Ask the user to provide up to three typical examples of the texts he or she wish to summarize. "
            "This will help you get familiar with the domain and the flavor of the user's documents. "
            "Mention to the user that they need to share three examples one at a time, "
            "but at each stage they can indicate that they do not have anymore examples to share. "
            "Please ask them to share only the clean text of the examples without any prefixes or suffixes. "
            "Do not share your insights until you have collected all examples."
        )
        resp = self._get_assistant_response(max_new_tokens=200)
        self._add_assistant_msg(resp, 'both')
        return resp

    def _do_nothing(self):
        resp = self._get_assistant_response(max_new_tokens=200)
        self._add_assistant_msg(resp, 'both')
        return resp

    def _ask_for_next_text(self):
        self._add_system_msg(
            "Ask the user for the next text example. "
            "Please remind them to share only the clean text of the examples without any prefixes or suffixes. "
        )
        resp = self._get_assistant_response(max_new_tokens=200)
        self._add_assistant_msg(resp, 'both')
        return resp

    def _has_more_texts(self):
        if self.user_has_more_texts:
            self._add_system_msg(
                'Has the user indicated they finished sharing texts (e.g. that they have no more examples to share), or not? Answer either "finished" or "not finished"')
            resp = self._get_assistant_response(max_new_tokens=20)
            self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
            self.user_has_more_texts = ("not finished" in resp.lower()) or not (resp.lower().startswith("finished") or
                "no more" in resp.lower() or "have finished" in resp.lower() or "they finished sharing" in resp.lower())
            logging.info(f"user_has_more_texts is set to {self.user_has_more_texts}")
        return self.user_has_more_texts

    def _ask_text_questions(self):
        self._add_system_msg(
            "Now, if the user shared some examples, ask the user up to 5 relevant questions about his summary preferences. "
            "Please do not ask questions that refer to a specific example. "
            "Please clarify to the user that he doesn׳t need to answer all the questions, only those that he feels are relevant for his summary preferences. "
            "Ask the user to provide  all the answers at the same message. "
            "If the user did not provide any examples, ask only general questions about the prompt "
            "without mentioning that the user shared examples."
        )
        resp = self._get_assistant_response()
        # self.hidden_chat = self.hidden_chat[:-1]  # remove the last question
        self._add_assistant_msg(resp, 'both')

    def _evaluate_prompt(self):
        summary_correction = len(self.approved_summaries) > self.validated_example_idx
        prompt = self.approved_prompts[-1]['prompt']

        prompt_str = build_few_shot_prompt(prompt,
                                           self.approved_summaries[:self.validated_example_idx],
                                           self.bam_client.parameters['model_id'])
        example = self.text_examples[self.validated_example_idx]
        prompt_str = prompt_str.format(text=example)
        summary = self.bam_client.send_messages(prompt_str)[0]

        if summary_correction:
            self.approved_summaries[self.validated_example_idx]['summary'] = summary
        else:
            self.approved_summaries.append({'text': example, 'summary': summary})

        self._add_system_msg(f'When the latest prompt was used for summarization of the example ```{example}``` '
                             f'it produced the result ```{summary}```')
        self._add_system_msg(
            f'Present these results to the user, mention they are based on the text of example no. '
            f'{(self.validated_example_idx + 1)} shared by the user, and ask if they want any changes.'
        )
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
        name = extract_delimited_text(resp, "```").strip().replace('"', '').replace(" ", "_")[:50]

        prompt_str = build_few_shot_prompt(prompt, self.approved_summaries[:self.validated_example_idx],
                                           self.bam_client.parameters['model_id'])
        final_msg = "Here is the final prompt: \n\n" + prompt_str
        saved_name, bam_url = self.bam_client.save_prompt(name, prompt_str)
        final_msg += f'\n\nThis prompt has been saved to your prompt Library under the name "{saved_name}". ' \
                     f'You can try it in the [BAM Prompt Lab]({bam_url}) or in the Evaluate tab. This prompt works best for model {self.bam_client.parameters["model_id"]}.'
        self._add_assistant_msg(final_msg, 'user')
        self.state = ConversationState.DONE

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

    def generate_agent_message(self):
        if (len(self.user_chat) > 0 and self.user_chat[-1]['role'] == ChatRole.ASSISTANT) or \
                (self.state == ConversationState.INITIALIZING) :
            return None

        logging.info(f"in {self.state}")
        if self.state is None:
            self.state = ConversationState.INITIALIZING
            self._init_chats()
            self.state = ConversationState.INTRODUCTION

        elif self.state == ConversationState.INTRODUCTION:
            if len(self.text_examples) == 0:
                self.enable_upload_file = False
                logging.info(f"asking for text to summarize")
                self._ask_for_text()
                next_state = ConversationState.PROCESS_TEXTS
            else:
                instruction_txt = 'Look at the following text examples, and suggest a summarization prompt for them. ' \
                                  'Do not include the examples into the prompt. ' \
                                  'Enclose the suggested prompt in triple quotes (```).\n'
                self._add_system_msg(instruction_txt + '\n'.join(self.text_examples))
                resp = self._get_assistant_response(max_new_tokens=200)
                initial_prompt = extract_delimited_text(resp, '```')
                next_state = ConversationState.CONFIRM_PROMPT

            self.state = next_state

        elif self.state == ConversationState.CONFIRM_PROMPT:
            if self._user_asked_for_correction():
                logging.info("user asked for correction")
                if self._need_clarification_from_the_user():
                    logging.info("clarification question")
                    self._ask_clarification_question()
                else:
                    self._confirm_prompt(is_new=False)
            else:
                if self._prompt_suggestion_accepted():
                    logging.info("user accepted the suggested prompt")
                    if self.user_has_more_texts:
                        logging.info(f"asking for text to summarize")
                        self._ask_for_text()
                        self.state = ConversationState.PROCESS_TEXTS
                    else:
                        logging.info(f"user gave {len(self.text_examples)} text examples. ({self.validated_example_idx})")
                        self._evaluate_prompt()
                        self.state = ConversationState.CONFIRM_SUMMARY
                else:
                    logging.info("user did not accept the prompt")
                    self._confirm_prompt(is_new=False)

        elif self.state == ConversationState.PROCESS_TEXTS:
            example_extracted = self._extract_text_example()
            if example_extracted:
                logging.info("extracted text from user")
            if self._has_more_texts() and len(self.text_examples) < NUM_USER_EXAMPLES:
                logging.info("ask the user for another example")
                self._ask_for_next_text()
            else:
                self.user_has_more_texts = False
                logging.info("ask questions on the examples provided")
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
            if not self._summary_suggestion_corrected():
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

        elif self.state == ConversationState.DONE:
            self._add_msg(self.user_chat, ChatRole.ASSISTANT, 'Please press the "Reset" button to restart the session')

        return self.user_chat[-1]

    def process_examples(self, df):
        self.enable_upload_file = False
        self.user_has_more_texts = False

        text_col = df.columns[0]  # can ask the model which column is most likely the text
        texts = df[text_col].tolist()
        self.text_examples = texts[:3]
        if len(texts) > 10:
            texts = texts[3:]
            random.shuffle(texts)

        temp_chat = []
        system_message = 'We are working on a tailored prompt for text summarization. '\
                         'Following are few examples of texts to be summarized. '\
                         'Describe common characteristics of those examples which may be relevant for the summarization.'
        self._add_msg(temp_chat, ChatRole.SYSTEM, system_message)

        max_len_tokens = self.bam_client.parameters['max_total_tokens'] - 1000  # random value to account for the role headers
        token_counts = self.bam_client.count_tokens(texts)
        total_len = 0
        num_examples = 0
        for txt, token_count in zip(texts, token_counts):
            if total_len + token_count > max_len_tokens:
                break
            num_examples += 1
            total_len += token_count
            self._add_msg(temp_chat, ChatRole.SYSTEM, txt)

        characteristics = self._get_assistant_response(temp_chat)
        self._add_system_msg(f'The user has provided {num_examples} examples')
        self._add_msg(self.hidden_chat, ChatRole.ASSISTANT, characteristics)
        self._ask_text_questions()

        self.state = ConversationState.PROCESS_RESPONSES

    def get_prompts(self):
        return self.approved_prompts

