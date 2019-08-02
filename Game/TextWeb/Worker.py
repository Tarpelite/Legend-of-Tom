from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange
import re


import torch
import torch.nn.functional as F
from torch.utils.data import SequentialSampler, DataLoader
import numpy as np
import editdistance

from pytorch_transformers import GPT2Config, OpenAIGPTConfig

from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer


class GPTWorker(object):
    '''
        create GPTworker to handle the request
    '''
    def __init__(self):

        # step 1: init parameters
        self.model_type = "gpt2"
        self.model_name_or_path = "gpt2"
        self.top_k = 4
        self.length = 50
        self.prompt = ""
        self.text = ""
        self.top_p = 0.9
        self.temperature = 1.0
        self.no_cuda = False
        self.seed = 42
        self.device = torch.device("cuda" if torch.cuda.is_available() and not self.no_cuda else "cpu")
        self.n_gpu = torch.cuda.device_count()

        # step 2: init the model and tokenizer
        self.set_seed()
        self.model_type = self.model_type.lower()
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name_or_path)
        self.model = GPT2LMHeadModel.from_pretrained(self.model_name_or_path)
        self.model.to(self.device)
        self.model.eval()

        if self.length < 0 and self.model.config.max_position_embeddings > 0:
            self.length = self.model.config.max_position_embeddings
        elif 0 < self.model.config.max_position_embeddings < self.length:
            self.length = self.model.max_position_embeddings
        elif self.length < 0:
            self.length = int(10000)


    def set_seed(self):
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(self.seed)
    
    
    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def sample_sequence(self, model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        with torch.no_grad():
            for _ in trange(length):

                inputs = {'input_ids': generated}
                if is_xlnet: 
                    # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                    # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                    input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                    perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                    perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                    target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                    target_mapping[0, 0, -1] = 1.0  # predict last token
                    inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

                outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
        return generated

    def predict_one(self, prompt_text):
        if prompt_text:
            prompt = prompt_text
        else:
            prompt = self.prompt
        
        context_tokens = self.tokenizer.encode(prompt)
        text = self.product_text(context_tokens)
        return text
    
    def predict_choices(self, prompt_text, ans_num):
        if prompt_text:
            prompt = prompt_text
        else:
            prompt = self.prompt
        context_tokens = self.tokenizer.encode(prompt)
        choices = []
        for i in range(ans_num):
            text = self.product_text(context_tokens)
            choices.append(text)
        return choices
    
    def product_text(self, context_tokens):

        out = self.sample_sequence(
                model=self.model,
                context=context_tokens,
                length=self.length,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                device=self.device,
                is_xlnet=bool(self.model_type == "xlnet"),
            )

        out = out[0, len(context_tokens):].tolist()
        text = self.tokenizer.decode(out, clean_up_tokenization_spaces=True)
        # print('before split:', text)
        text = text.strip()
        text = text.strip('\n')   
        text_splits = re.split(r'(\.|\?|\n|\t|\<\|endoftext\|\>)', text)
        text = text_splits[0]
        # print("splits", text_splits)    
        i = 0
        text = text_splits[0]
        while i < len(text_splits) and len(text) <= 2:
            i = i + 1
            text = text_splits[i]    
        text = re.sub(u"([^\u4e00-\u9fa5\u0033-\u0047\u0041-\u005a\u0061-\u007a\u0020-\u007E])","",text)
        return text

    def check(self, text, choices):
        min_score = 100
        for chosen in choices:
            score = editdistance.eval(text, chosen)*1.00 / min(len(text), len(chosen))
            min_score = min(min_score, score)
        print("minscore:{}".format(min_score))
        if min_score < 0.7:
            return False
        return True
    
    def time_based_predict(self, prev, time):
        '''
            make predictions based on time
        '''
        ans_num = 4
        end_phrase = " He came to the end of his life."
        if time >= 2000:
            return prev + "\n" + "Then in " + str(time) + ", " + end_phrase
        prompt = prev + "\n" + "In " + str(time) + ", "
        
        context_tokens = self.tokenizer.encode(prompt)
        choices = []
        for i in range(ans_num):
            text = self.product_text(context_tokens)
            text = "\n" + "In " + str(time) +  ", " + text
            lim = 4
            while not self.check(text, choices) and lim > 0:
                text = self.product_text(context_tokens)
                text = "\n" + "In " + str(time) +  ", " + text
                lim -= 1
            lim = 4
            choices.append(text)
        return choices



if __name__ == "__main__":
    worker = GPTWorker()
    prompt = """Tom was born in 1931, he was an Irish poet and playwright. After writing in different forms throughout the 1880s, he became one of London's most popular playwrights in the early 1890s. He is best remembered for his epigrams and plays, his novel The Picture of Dorian Gray, and the circumstances of his criminal conviction for 'gross indecency', imprisonment, and early death at age 46. 
Tom's parents were successful Anglo-Irish intellectuals in Dublin. Their son became fluent in French and German early in life. At university, Tom read Greats; he proved himself to be an outstanding classicist, first at Trinity College Dublin, then at Oxford. He became known for his involvement in the rising philosophy of aestheticism, led by two of his tutors, Walter Pater and John Ruskin. After university, Tom moved to London into fashionable cultural and social circles. 
    """
    time = 1950
    cnt = 0
    span = 2
    print(prompt)
    while time < 2010:
        if time == 2000:
            endding = worker.time_based_predict(prompt, time)
            prompt += endding
            time += 10
            print(prompt)
        elif cnt % span == 0:
            choices = worker.time_based_predict(prompt, time)
            for step, choice in enumerate(choices):
                print("{}: {}".format(step, choice))
            num = input()
            num = int(num.strip())
            text = choices[num]
            prompt += text
            time += 10
            print(prompt)
        else:
            text = worker.predict_one(prompt)
            prompt += ' ' + text
            print(prompt)
        cnt += 1
        





        
            
