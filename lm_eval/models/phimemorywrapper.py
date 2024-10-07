# lm_eval --model phi3memory --model_args model_path=/datadrive2/llm-memory/2024_8_27_simple/full_model.pth --tasks openbookqa --apply_chat_template


import random

from tqdm import tqdm

from lm_eval.api.model import TemplateLM, LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    clear_torch_cache,
    configure_pad_token,
    get_dtype,
    pad_and_concat,
    stop_sequences_criteria,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoModel, pipeline
from datasets import load_from_disk
import json
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torch.optim as optim
import os
from abc import ABC, abstractmethod
import argparse
from accelerate.logging import get_logger
import logging
import random
import numpy as np
import logging
import os
from logging.handlers import RotatingFileHandler
import re
from dataclasses import dataclass
from datasets import Dataset
import pandas as pd
from typing import Dict, List, Literal, Optional, Tuple, Union

# local modules
phimemory_dir = "/datadrive/llm-memory/src"
import sys
sys.path.append(phimemory_dir)
from phimemory import MemoryAugmentedPhi3ForQA
from semantic import getFineGrainedSem, parse_sentence


@register_model("phi3memory")
class phi3_llmmemory(TemplateLM):
    def __init__(
        self,
        model_path,
        base_model,
        batch_size: Optional[Union[int, str]] = 1,
    ) -> None:
        super().__init__()

        self.model = MemoryAugmentedPhi3ForQA(base_model, 10, 
                                    device="cuda:0", memory_type="simple")
        self.model.load_state_dict(torch.load(model_path))
        self.device = self.model.device
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.batch_size = batch_size
        print("set batch size to", self.batch_size)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def prefix_token_id(self):
        return self.tokenizer.bos_token_id
    
    @property
    def max_length(self):
        return 2048
    
    @property
    def tokenizer_name(self):
        return self.tokenizer.name_or_path.replace("/", "__")
    
    def tok_encode(self, text, add_special_tokens=False):
        return self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
    
    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False):
        res = []
   
        def _collate(x):
            toks = x[1] + x[2]
            return -len(toks), tuple(toks)
        
        re_ord = Collator(requests, sort_fn=_collate)
        chunks = re_ord.get_batched(
            n=int(self.batch_size) if self.batch_size != "auto" else 0, batch_fn=None
        )

        pbar = tqdm(
            total=len(requests),
            disable=disable_tqdm,
            desc="Running loglikelihood requests",
        )
        for chunk in chunks:
            inps = []
            cont_toks_list = []
            inplens = []

            conts = []
            inp_attns = []
            padding_len_inp = None
            padding_len_cont = None
            for _, context_enc, continuation_enc in chunk:
                # sanity check
                assert len(context_enc) > 0
                assert len(continuation_enc) > 0
                assert len(continuation_enc) <= self.max_length

                # how this all works (illustrated on a causal decoder-only setup):
                #          CTX      CONT
                # inp    0 1 2 3|4 5 6 7 8 9   <- last token is deleted by inp[:, :-1]
                # model  \               \
                # logits   1 2 3|4 5 6 7 8 9   <- the ctx half gets tossed out by the
                # cont_toks      4 5 6 7 8 9      [:, -len(continuation_enc):, :self.vocab_size] slice

                inp = torch.tensor(
                    (context_enc + continuation_enc)[-(self.max_length + 1) :][:-1],
                    dtype=torch.long,
                    device=self.device,
                )
                (inplen,) = inp.shape
                # create a torch tensor with all one same length as context_enc
                context_attn = torch.tensor(context_enc[-self.max_length :], dtype=torch.long, device=self.device)
                inp_attns.append(torch.ones_like(context_attn))
                
                padding_len_inp = (
                    max(padding_len_inp, inplen)
                    if padding_len_inp is not None
                    else inplen
                )

                inps.append(inp) 
                cont_toks_list.append(continuation_enc)
                inplens.append(inplen)
            
            batched_inps = pad_and_concat(
                padding_len_inp, inps, padding_side="right"
            )  # [batch, padding_len_inp]
        
            batched_encoder_mask = pad_and_concat(
                padding_len_inp, inp_attns
            )  # [batch, padding_len_inp]
            
            with torch.no_grad():
                multi_logits, _ = self.model.generate(input_ids=batched_inps, attention_mask=batched_encoder_mask, memory_state=None, update_memory=False)
            
            multi_logits = multi_logits.logits

            multi_logits = F.log_softmax(multi_logits, dim=-1)

            for (request_str, ctx_tokens, _), logits, inplen, cont_toks in zip(
                chunk, multi_logits, inplens, cont_toks_list
            ):
                # Slice to original seq length
                contlen = len(cont_toks)
                # take only logits in the continuation
                # (discard context toks if decoder-only ; discard right-padding)
                # also discards + checks for "virtual tokens" in the causal LM's input window
                # from prompt/prefix tuning tokens, if applicable
                ctx_len = (
                    inplen + (logits.shape[0] - padding_len_inp)
                )
                logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
                logits = logits.unsqueeze(0)  # [1, seq, vocab]

                # Check if per-token argmax is exactly equal to continuation
                greedy_tokens = logits.argmax(dim=-1)

                # check for one-token continuation cache hits.
                # noop in case group_by != "contexts" or no cache hit and returns the
                # original args. Otherwise, expands the logits batch dimension and yields each
                # batch along with matching continuation tokens and prompt strings.
                # logits -> [1, seq, vocab]
                for request_str, cont_toks, logits in re_ord.get_cache(
                    req_str=request_str,
                    cxt_toks=ctx_tokens,
                    cont_toks=cont_toks,
                    logits=logits,
                ):
                    cont_toks = torch.tensor(
                        cont_toks, dtype=torch.long, device=self.device
                    ).unsqueeze(0)  # [1, seq]
                    max_equal = (greedy_tokens == cont_toks).all()

                    # Obtain log-probs at the corresponding continuation token indices
                    # last_token_slice = logits[:, -1, :].squeeze(0).tolist()
                    logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(
                        -1
                    )  # [1, seq]

                    # Answer: (log prob, is-exact-match)
                    answer = (float(logits.sum()), bool(max_equal))

                    res.append(answer)
                    pbar.update(1)

        pbar.close()

        return re_ord.get_original(res)
    
    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        assert (
            contlen and inplen
        ), "Must pass input len and cont. len to select scored logits for causal LM"
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]
        return logits
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("This model does not support rolling log-likelihoods")

    def generate_until(self, requests):
        raise NotImplementedError("This model does not support generate_until")

    def chat_template(self, chat_template: Union[bool, str] = False) -> str:
        """
        Get the appropriate chat template for the model based on the `chat_template` argument.

        This method returns the chat template string to build the prompt from a chat history.
        The chat template is saved in the evaluation results for reproducibility.
        Boolean arguments should be used with models that have only one chat template,
        while string arguments are used with models that have multiple chat templates.
        For the reference implementation, see HFLM class in `lm_eval.models.huggingface`.

        Args:
            chat_template (Union[bool, str]): Specifies whether to apply a chat template:
                - If False: Do not apply any chat template.
                - If True: Apply the default chat template.
                - If str: Apply the specified chat template by name.

        Returns:
            str: The selected chat template in Jinja format.
        """
        return self.tokenizer.chat_template

    def apply_chat_template(self, chat_history: List[Dict[str, str]]) -> str:
        """
        Process a chat history to create a string that can be tokenized and input into the model.

        Args:
            chat_history (List[Dict[str, str]]): A list of dictionaries representing the chat history,
                where each dictionary has "role" and "content" keys.

        Returns:
            str: A string representing the chat history that can be tokenized and fed into the model.
        """
        return self.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)