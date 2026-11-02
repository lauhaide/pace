import time
from abc import ABC
from collections import Counter

from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import random
random.seed(42)
#sht update
import ray
import torch
from datasets import load_dataset
import torch.nn as nn
from tqdm import tqdm
from datasets import Dataset
import copy
import requests
import time
from openrlhf.models.actor import Actor
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean, unpacking_samples, list_chunks, unpacking_samples_list
from openrlhf.utils.logging_utils import init_logger
from openrlhf.utils.remote_rm_utils import remote_rm_fn, remote_rm_fn_ray
import re
from collections import defaultdict
logger = init_logger(__name__)

# new
import json
import jsonlines
import requests
from bs4 import BeautifulSoup
import wikipediaapi
from urllib.parse import unquote
from urllib.request import urlopen
from urllib.parse import urlparse
import wikipedia
from requests.exceptions import Timeout
from tqdm import tqdm #这个没有
import time
import concurrent #这个没有
from concurrent.futures import ThreadPoolExecutor
import pdfplumber #这个没有
from io import BytesIO
import re
import string
from typing import Optional, Tuple
#from nltk.tokenize import sent_tokenize #没有，但也没用上
#import nltk #没有，但也没用上
from typing import List

import multiprocessing #没有
from openai import OpenAI
import sys
import os
from datasets import load_dataset
import http.client
from time import sleep
from collections import defaultdict
import random
import math
import spacy

from scipy.optimize import linear_sum_assignment
import numpy as np


def to(tensor: Union[torch.Tensor, list[torch.Tensor]], device):
    if isinstance(tensor, list):
        return [to(t, device) for t in tensor]
    return tensor.to(device) if isinstance(tensor, torch.Tensor) else tensor


def pin_memory(tensor: Union[torch.Tensor, list[torch.Tensor]]):
    if isinstance(tensor, list):
        return [pin_memory(t) for t in tensor]
    return tensor.pin_memory() if isinstance(tensor, torch.Tensor) else tensor


# Update 1229 For GRPO
def conditional_cat(attr1, attr2):
    if attr1 is not None and attr2 is not None:
        if isinstance(attr1, torch.Tensor):
            op = lambda x, y: torch.cat((x, y), dim=0)
        else:
            op = lambda x, y: x + y
        return op(attr1, attr2)
    return None

def bool_mapping(s):
    """布尔值映射为yes/no"""
    if s == "True": return "yes"
    elif s == "False": return "no"
    else: return s

def exact_match_score(prediction, ground_truth):
    return normalize_answer(bool_mapping(prediction)) == normalize_answer(
        bool_mapping(ground_truth)
    )

def cover_exact_match_score_1(prediction, ground_truth):

    pre_list = normalize_answer(bool_mapping(prediction)).split(" ")
    ground_list = normalize_answer(bool_mapping(ground_truth)).split(" ")

    # 不考虑顺序和连续
    return all(ground in pre_list for ground in ground_list)

def f1_score(prediction, ground_truth):
    """计算F1分数
    Args:
        prediction: 模型预测的答案
        ground_truth: 标准答案
    Returns:
        (f1, precision, recall): F1分数及其组成部分
    """
    # Standardized processing of predictions and actual answers
    normalized_prediction = normalize_answer(bool_mapping(prediction))
    normalized_ground_truth = normalize_answer(bool_mapping(ground_truth))

    ZERO_METRIC = 0

    # 特殊情况处理：yes/no/noanswer答案
    if (normalized_prediction in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC
    if (normalized_ground_truth in ["yes", "no", "noanswer"] and 
        normalized_prediction != normalized_ground_truth):
        return ZERO_METRIC

    # 分词
    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    
    # 计算共同词的数量
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return ZERO_METRIC
        
    # 计算precision和recall
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    # 计算F1
    f1 = (2 * precision * recall) / (precision + recall)
    return f1 #, precision, recall


def normalize_text(text):
    text = re.sub("[,.:\"'\[\]\-=\+\\|!@#$%^&*();<>?/！￥…（）—\{\}：”“《》？]", " ", text.lower())
    text = re.sub("import\s[a-zA-Z\.]+(\sas\s[a-zA-Z\.]+)\n", " ", text)
    text = re.sub("\s+", " ", text)
    return text.strip()

def strip_sequence(text, pad_token, eos_token):
    pad_token_escaped = re.escape(pad_token)
    eos_token_escaped = re.escape(eos_token)

    pattern = f"^({eos_token_escaped}|{pad_token_escaped})+"
    text = re.sub(pattern, "", text)

    pattern = f"({eos_token_escaped}|{pad_token_escaped})+$"
    text = re.sub(pattern, "", text)
    return text


def extract_answer_math(s):
    return s.split("<answer>")[-1].split("</answer>")[0].strip()

def extract_all_answers(s):
    return [ans.strip() for ans in re.findall(r"<answer>(.*?)</answer>", s, re.DOTALL)]
def extract_all_answers_with_full_spans(s):
    results = []
    for match in re.finditer(r"<answer>(.*?)</answer>", s, re.DOTALL):
        answer_text = match.group(1).strip()
        full_start = match.start()  # 从 <answer> 开始
        full_end = match.end()      # 到 </answer> 结束
        results.append((answer_text, full_start, full_end))
    return results

# [其余函数保持不变...]
def normalize_answer(s):
    """标准化答案文本的处理函数"""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["'", "'", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    return white_space_fix(remove_articles(remove_punc(lower(replace_underscore(s)))))

# End of Update

def retrieval_precision_(golden_answers, doc_list):
    for doc in doc_list:
        for golden_answer in golden_answers:
            if normalize_answer(golden_answer) in normalize_answer(doc):
                return True
    return False

@dataclass
class Experience:
    """
    This represents a single experience (used in rollout algorithms such as PPO).

    Description of the shape for each field:
    sequences: (B, S) - token Sequence (containing prompt + response)
    action_log_probs: (B, A) - 当前策略下的 action 的对数概率
    base_action_log_probs: (B, A) - 旧策略（参考模型）下的 log probs
    values: (B, A) - 价值函数预测值（critic 输出）
    returns: (B, A) - 折扣累计奖励
    advantages: (B, A) - 优势函数值
    attention_mask: (B, S) - 自注意力 mask
    action_mask: (B, A) - 表示哪些 token 属于 action 部分
    retrieve_mask: (B, A) - 表示哪些 token 是非检索部分（用于 KL 计算）
    kl: (B, A) - KL divergence value
    info: Additional information dictionary (containing reward, num_actions, etc.)

    """
    """Experience is a batch of data.
    These data should have the the sequence length and number of actions.
    Left padding for sequences is applied.

    Shapes of each tensor:
    sequences: (B, S)
    action_log_probs: (B, A)
    base_action_log_probs: (B, A)
    values: (B, A)
    returns: (B, A)
    advantages: (B, A)
    attention_mask: (B, S)
    action_mask: (B, A)
    retrieve_mask: (B, A)
    kl: (B, A)

    "A" is the number of actions.
    """

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    base_action_log_probs: torch.Tensor
    values: torch.Tensor
    returns: Optional[torch.Tensor]
    advantages: Optional[torch.Tensor]
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    retrieve_mask: Optional[torch.BoolTensor]
    retrieve_score_segments: Optional[torch.Tensor]
    retrieve_score: Optional[torch.Tensor]
    lets_score: Optional[torch.Tensor]
    group_diversity: Optional[torch.Tensor]
    state_acoverage: Optional[torch.Tensor]
    state_diversity: Optional[torch.Tensor]
    info: Optional[dict]
    kl: Optional[torch.Tensor] = None


    @torch.no_grad()
    def to_device(self, device: torch.device):
        self.sequences = to(self.sequences, device)
        self.action_log_probs = to(self.action_log_probs, device)
        self.base_action_log_probs = to(self.base_action_log_probs, device)
        self.returns = to(self.returns, device)
        self.advantages = to(self.advantages, device)
        self.values = to(self.values, device)
        self.attention_mask = to(self.attention_mask, device)
        self.action_mask = to(self.action_mask, device)
        self.retrieve_mask = to(self.retrieve_mask, device)
        self.retrieve_score_segments = to(self.retrieve_score_segments, device)
        self.retrieve_score = to(self.retrieve_score, device)
        self.lets_score = to(self.lets_score, device)
        self.group_diversity = to(self.group_diversity, device)
        self.state_acoverage = to(self.state_acoverage, device)
        self.state_diversity = to(self.state_diversity, device)        
        self.kl = to(self.kl, device)
        self.info = {key: to(value, device) for key, value in self.info.items()}
        return self

    def pin_memory(self):
        self.sequences = pin_memory(self.sequences)
        self.action_log_probs = pin_memory(self.action_log_probs)
        self.base_action_log_probs = pin_memory(self.base_action_log_probs)
        self.returns = pin_memory(self.returns)
        self.advantages = pin_memory(self.advantages)
        self.values = pin_memory(self.values)
        self.attention_mask = pin_memory(self.attention_mask)
        self.action_mask = pin_memory(self.action_mask)
        self.retrieve_mask = pin_memory(self.retrieve_mask)
        self.retrieve_score_segments = pin_memory(self.retrieve_score_segments)
        self.retrieve_score = pin_memory(self.retrieve_score)
        self.lets_score = pin_memory(self.lets_score)
        self.group_diversity = pin_memory(self.group_diversity)
        self.state_acoverage = pin_memory(self.state_acoverage)
        self.state_diversity = pin_memory(self.state_diversity)
        self.kl = pin_memory(self.kl)
        self.info = {key: pin_memory(value) for key, value in self.info.items()}
        return self

    # CZP Update 1229 For GRPO
    def __add__(self, other):
        """支持 Experience 之间的合并，用于拼接 batch"""
        if not isinstance(other, Experience):
            return NotImplemented

        info = {}
        for k in self.info.keys():
            info[k] = conditional_cat(self.info[k], other.info[k])

        return Experience(
            sequences=conditional_cat(self.sequences, other.sequences),
            action_log_probs=conditional_cat(self.action_log_probs, other.action_log_probs),
            values=conditional_cat(self.values, other.values),
            returns=conditional_cat(self.returns, other.returns),
            advantages=conditional_cat(self.advantages, other.advantages),
            attention_mask=conditional_cat(self.attention_mask, other.attention_mask),
            action_mask=conditional_cat(self.action_mask, other.action_mask),
            retrieve_mask=conditional_cat(self.retrieve_mask, other.retrieve_mask),
            retrieve_score_segments=conditional_cat(self.retrieve_score_segments, other.retrieve_score_segments),
            retrieve_score=conditional_cat(self.retrieve_score, other.retrieve_score),
            lets_score=conditional_cat(self.lets_score, other.lets_score),
            group_diversity=conditional_cat(self.group_diversity, other.group_diversity),
            state_acoverage=conditional_cat(self.state_acoverage, other.state_acoverage),
            state_diversity=conditional_cat(self.state_diversity, other.state_diversity),
            # retrieve_mask=None,
            info=info,
            kl=conditional_cat(self.kl, other.kl),
        )

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    # End of Update


@dataclass
class Samples:
    """Samples is a batch of data.
    There can be 2 formats to store the samples, batched or packed.
    The batched format means padding is applied to the sequences, while the packed format
    will concatenate the prompt and response without padding.

    Shapes of each tensor, when 2 shapes are shown, the first one is for batched format
        and the second one is for packed format:
    sequences: (B, S) or (1, total_length), the tokens of both prompt and response.
    attention_mask: (B, S) or (1, total_length), the attention mask for sequences.
    action_mask: (B, A) or None, the action (response) mask to show which part of the
        sequence is the response. When the samples are packed, this is None.
    num_actions: int or (B,), the number of actions (tokens) in the response.
        When the samples are not packed, we will use action_mask, so this is an int to
        show the size of action_mask. Otherwise, this is a tensor to show the number of
        actions for each sample.
    packed_seq_lens: None or (B,), the length of each sample in the packed samples.
    response_length: (B,), the number of tokens in the response.
    total_length: (B,), the total number of tokens in the sequences.
    """
    """
    表示从模型生成的一个 batch 的样本。

    有两种格式：
    - batched 格式：所有 sample 使用 padding 对齐
    - packed 格式：将 prompt + response 拼接成一条长序列，不做 padding

    字段含义：
    sequences: 主体 token 序列，形状为 (B, S) 或 (1, total_length)
    attention_mask: 对应 attention 的 mask
    action_mask: 标记哪些 token 属于 response（动作）部分
    retrieve_mask: 标记哪些 token 是非检索区域（用于 KL）
    num_actions: 一个整数或张量，表示 response 的 token 数
    packed_seq_lens: 每条 sample 的长度，仅在 packed 格式下有效
    response_length: 每个 sample 的 response 长度
    total_length: 每个 sample 的总 token 长度
    retrieve_num: 每条 sample 中的检索次数
    pure_response_length: 每条 sample 中非检索区域 response token 长度
    """

    sequences: torch.Tensor
    attention_mask: Optional[torch.LongTensor]
    action_mask: Optional[torch.BoolTensor]
    retrieve_mask: Optional[torch.BoolTensor]
    retrieve_score_segments: Optional[torch.Tensor]
    retrieve_score: Optional[torch.Tensor]
    lets_score: Optional[torch.Tensor]
    num_actions: Union[int, torch.Tensor]
    packed_seq_lens: Optional[torch.Tensor]
    response_length: torch.Tensor
    total_length: torch.Tensor
    retrieve_num: Union[int, torch.Tensor]
    group_diversity: Union[int, torch.Tensor]
    state_acoverage: Union[int, torch.Tensor]
    state_diversity: Union[int, torch.Tensor]
    shallow_search: Union[int, torch.Tensor]
    pure_response_length: torch.Tensor
    exit_flag: torch.Tensor
    answer_num: torch.Tensor

class NaiveExperienceMaker(ABC):
    """
    Naive experience maker.
    """

    def __init__(
        self,
        actor: Actor,  # 当前 actor 模型（用于采样和 log_probs）
        critic: nn.Module,  # 当前 critic 模型（用于 value 计算）
        reward_model: nn.Module,  # The reward model can be a local model or a remote model.
        initial_model: Actor,  # 参考模型（旧策略），用于 KL 估计
        tokenizer,  # 用于编码/解码 prompt 和生成
        prompt_max_len: int,  # prompt 最大长度
        kl_controller,  # KL 控制器（动态调整 KL 系数）
        strategy=None,  # 包含训练参数（args）和 rank 信息的辅助类
        remote_rm_url: str = None,  # If using a remote reward model, provide its URL.
        remote_sufficient_url: str = None,  # 如果使用远程 summarizer，提供其 URL
        reward_fn=None,  # Optional custom reward aggregation function (for scenarios with multiple reward models)
    ) -> None:
        super().__init__()
        self.actor = actor
        self.critic = critic
        self.reward_model = reward_model
        self.remote_rm_url = remote_rm_url
        self.remote_sufficient_url = remote_sufficient_url
        self.initial_model = initial_model #initial model就是reference model
        self.tokenizer = tokenizer
        self.prompt_max_len = prompt_max_len
        self.kl_ctl = kl_controller
        self.strategy = strategy
        self.reward_fn = reward_fn
        self.perf_stats = None
        self.advantage_estimator = strategy.args.advantage_estimator
        # CZP Update 1229 For GRPO
        self.args = strategy.args
        print(self.args)
        # End of Update

    # tokenizer
    def tokenize_fn(self, texts, max_length, padding=True, device=None):
        """
        对输入的 prompts 进行 tokenizer 编码。
        - 当 padding=True 时，返回一个 batch 字典（张量格式）
        - 当 padding=False 时，返回一个 token id 的列表（适用于 packed 格式）
        """

        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], global_step: int, wandb: None, total_training_steps: int, **generate_kwargs) -> List[Experience]:
        """
        Make a list of experience with the micro_rollout_batch_size.

        This method will first calculate the response sequences and rewards for the given prompts.
        Then, if we need certain processing for the rewards or do certain filtering, we can process the rollout as a whole.
        After that, we will calculate the advantages and returns for each experience.
        """
        """
        Core function: Constructs multiple Experiences from prompts.

        The process is as follows:
        1. Use the actor to generate the response, and you will get Samples (containing sequences, action_masks, etc.).
        2. Iterate through Samples and transform each one into Experience (logprobs, value, reward, KL).
        3. Perform reward post-processing (e.g., baseline minus mean) on all Experiences.
        4. Calculate returns and advantages (GAE or other methods).
        """

        args = self.strategy.args
        if self.strategy.args.vllm_enable_sleep:
            from openrlhf.trainer.ray.vllm_engine import batch_vllm_engine_call

            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            
            torch.distributed.barrier()
            torch.cuda.synchronize()

        # param_stats = ray.get(self.vllm_engines[0].get_param_stats.remote())
        # for stat in param_stats:
        #     print(stat)                

        samples_list = self.generate_samples(all_prompts, **generate_kwargs) # When generating the vllm, the action_mask will be assigned a value here. Change it here #TODO!! #N outputs have already been rolled out here.
        print(f'Generated samples: {len(samples_list)}')
        #time.sleep(20)
        torch.distributed.barrier()# Synchronize all processes
        torch.cuda.synchronize()  
        
        if self.strategy.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")
            torch.distributed.barrier()# Synchronize all processes
            torch.cuda.synchronize()  
        
        experiences = []
        for i in tqdm(
            range(0, len(samples_list), args.n_samples_per_prompt * 10),
            desc="make_experience",
            disable=not self.strategy.is_rank_0(),
        ):
            samples = samples_list[i : i + args.n_samples_per_prompt * 10]
            m = self.make_experience(samples, global_step, total_training_steps = total_training_steps)
            if(type(m) == list):
                experiences.extend(m)
            else:
                experiences.append(m)

        variant_to_filter = not args.naive_grpo or (args.naive_grpo and args.reward_shape)

        #Do the filter implemented by Jie He et al.
        if args.enable_accuracy_filter and global_step > args.freezing_filter_steps and not args.reshape_filter:
            experiences = self.filter(experiences, global_step, wandb = wandb)
            if len(experiences) == 0:
                return []

        experiences, rewards, group_retrives_stds = self.process_experiences(experiences, global_step, total_training_steps) 

        if args.enable_accuracy_filter and global_step > args.freezing_filter_steps and args.reward_shape and args.reshape_filter:
            experiences, rewards = self.filter_after_shaping(experiences, rewards, group_retrives_stds, global_step, wandb = wandb)
        
        if args.reward_shape:
            # remains to flatten the rewards list
            rewards = [item for sublist in rewards for item in sublist]
        
        assert len(experiences) == len(rewards)

        for e, (experience, reward) in enumerate(zip(experiences, rewards)):
            if args.reward_shape:
                experience.info["reward"] = reward[-1].unsqueeze(0)

            experience = experience.to_device("cuda")
            reward = reward.to(device="cuda")
            num_actions = experience.info["num_actions"]  # Number of actions (per sample)
            reward = compute_reward(
                reward,
                self.kl_ctl.value,
                experience.kl,
                action_mask=experience.action_mask,
                retrieve_mask=experience.retrieve_mask,
                retrieve_score_segments=experience.retrieve_score_segments,
                num_actions=num_actions,
                reward_clip_range=args.reward_clip_range,
                reward_shape=args.reward_shape,
            )
            if self.advantage_estimator == "gae":
                experience.advantages, experience.returns = self.get_advantages_and_returns(
                    experience.values,
                    reward,
                    experience.action_mask,
                    generate_kwargs["gamma"],
                    generate_kwargs["lambd"],
                )
            elif self.advantage_estimator in ["reinforce_baseline", "rloo", "group_norm"]:
                if args.reward_shape:
                    # Not calling cumulative_returns as this is being compute directly w/ the rewards and only for the group_norm case.
                    # the following for is only to apply .squeeze(0) but is not needed now, and buggy when the tensor is 1D with only 1 element, it returns empty
                    #returns = []
                    #for ret in reward:
                        #ret = ret.squeeze(0)
                        #if len(ret.size()) == 0: print(reward)
                        #returns.append(ret)
                    #returns = [ret.squeeze(0) for ret in reward]
                    #experience.returns = returns
                    experience.returns = reward
                else:    
                    experience.returns = self.get_cumulative_returns(
                        reward,
                        experience.action_mask,
                        experience.retrieve_mask,
                        generate_kwargs["gamma"],
                    )
                experience.advantages = deepcopy(experience.returns)

                experience_num = len(experience.sequences)

            else:
                raise Exception(f"Unkown advantage_estimator {self.advantage_estimator}")
            
            if not getattr(self, "packing_samples", False): #If packing is enabled, the sum of the list will be used instead of the sum of the list.
                return_sums = reward.sum(dim=-1)
            else:
                return_sums = torch.tensor(
                    [each_reward.sum() for each_reward in reward], device=torch.cuda.current_device()
                )
                experience.info["return"] = return_sums
            experience.kl = None
            del experience.info["num_actions"]
            experience.to_device("cpu")

        return experiences


    @torch.no_grad()
    def show_experience(self, experience: Experience):
        """
        打印 experience 中的各项内容，用于调试。
        会显示 tensor 的维度、部分内容，或 list 的长度。
        """

        if isinstance(experience.action_log_probs, list):
            print(
                "list len",
                len(experience.action_log_probs),
                "action_log_probs: ",
                experience.action_log_probs[0].size(),
            )
            # print("action_log_probs: ", experience.action_log_probs[0].size())
        else:
            print("tensor action_log_probs: ", experience.action_log_probs.size())
        if isinstance(experience.sequences, list):
            torch.set_printoptions(threshold=500)
            list_data = experience.sequences[0].tolist()
            print("experience.sequences-try: ", list_data)

            print("experience.sequences-1", experience.sequences[0].size(), "list len", len(experience.sequences))
            torch.set_printoptions(threshold=10)
        elif experience.sequences is not None:
            print("tensor experience.sequences", experience.sequences.size())
        else:
            print("experience.sequences is None")
        if isinstance(experience.values, list):
            print("experience.values", experience.values[0].size(), "list len", len(experience.values))
            # print("experience.values", experience.values[0].size())
        elif experience.values is not None:
            print("tensor experience.values", experience.values.size())
        else:
            print(f"experience.values is None")
        if isinstance(experience.returns, list):
            print("experience.returns: ", experience.returns[0].size(), "list len", len(experience.returns))
            # print("experience.returns: ", experience.returns[0].size())
        elif experience.returns is not None:
            print("tensor experience.returns: ", experience.returns.size())
        else:
            print(f"experience.returns is None")
        if isinstance(experience.advantages, list):
            list_data = experience.advantages[0].tolist()
            print("experience.advantages-try: ", list_data)
            print("experience.advantages-1: ", experience.advantages[0].size(), "list len", len(experience.advantages))
        elif experience.advantages is not None:
            print("tensor experience.advantages: ", experience.advantages.size())
        else:
            print(f"experience.advantages is None")
        if isinstance(experience.attention_mask, list):
            print(
                "experience.attention_mask: ",
                experience.attention_mask[0].size(),
                "list len",
                len(experience.attention_mask),
            )
            # print("experience.attention_mask: ", experience.attention_mask[0].size())
        elif experience.attention_mask is not None:
            print("tensor experience.attention_mask: ", experience.attention_mask.size())
        else:
            print(f"experience.attention_mask is None")
        if isinstance(experience.action_mask, list):
            print(
                "experience.action_mask: ", experience.action_mask[0].size(), "list len", len(experience.action_mask)
            )
            # print("experience.action_mask: ", experience.action_mask[0].size())
        elif experience.action_mask is not None:
            print("tensor experience.action_mask: ", experience.action_mask.size())
        else:
            print(f"experience.action_mask is None")
        if isinstance(experience.kl, list):
            print("experience.kl: ", experience.kl[0].size(), "list len", len(experience.kl))
            # print("experience.kl: ", experience.kl[0].size())
        elif experience.kl is not None:
            print("tensor experience.kl: ", experience.kl.size())
        else:
            print(f"experience.kl is None")

        for k in experience.info:
            if isinstance(experience.info[k], torch.Tensor):
                print(k, experience.info[k].size(), experience.info[k])
                # print(experience.info[k])
            elif isinstance(experience.info[k], list):
                print(f"{k} list len: {len(experience.info[k])}, first: {experience.info[k][0]}")


#    @torch.no_grad()
#    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
#        """
#        Samples are generated using the actor model, with one micro-batch processed at a time.
#
#        输入：
#        - all_prompts: 所有输入的 prompt 文本
#        - generate_kwargs: 生成时使用的参数（如 temperature, top_k 等）
#
#        输出：
#        - samples_list: A list containing multiple Samples (each corresponding to a batch).
#        """
#
#        kill # Placeholder for debugging purposes; must be commented out before training.
#        assert not getattr(self, "packing_samples", False) # The current version does not support the packed format.
#        args = self.strategy.args
#        self.actor.eval() # Switch to evaluation mode to avoid dropout and other operations affecting the generation.
#        # sample multiple response
#        # Make multiple copies of each prompt (to generate multiple samples for the same prompt).
#        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
#        samples_list = []
#        for i in range(0, len(all_prompts), args.micro_rollout_batch_size):
#            prompts = all_prompts[i : i + args.micro_rollout_batch_size]
#            inputs = self.tokenize_fn(prompts, self.prompt_max_len, device="cuda")
#            # Generate a sample (response)
#            sequences, attention_mask, action_mask = self.actor.generate(**inputs, **generate_kwargs)
#            # Construct Samples object
#
#            samples = Samples(
#                sequences=sequences,
#                attention_mask=attention_mask,
#                action_mask=action_mask,
#                num_actions=action_mask.size(1),
#                packed_seq_lens=None,
#                response_length=action_mask.float().sum(dim=-1),
#                total_length=attention_mask.float().sum(dim=-1),
#            )
#            samples_list.append(samples)
#        return samples_list

#    @torch.no_grad()
#    def make_experience(self, samples: Samples) -> Experience:
#        """
#        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
#        """
#        """
#        Transform a Samples object into a complete Experience.
#        This includes calculations of log_probs, values, reward, KL divergence, etc.
#
#        enter:
#        - samples: A Samples object containing the generated sequences and their mask information.
#
#        Output:
#        - experience: An Experience object containing all the information needed for training.
#        """
#        kill
#        # Switch all models to eval mode (disable dropout and randomness in layer normalization).
#        self.actor.eval()
#        self.initial_model.eval()
#        if self.reward_model is not None:
#            self.reward_model.eval()
#        if self.critic is not None:
#            self.critic.eval()
#       
#        # Extracting various types of information from samples
#        # extract values from samples
#        sequences = samples.sequences
#        attention_mask = samples.attention_mask
#        action_mask = samples.action_mask
#        num_actions = samples.num_actions
#
#        # Calculate the log_prob of the action using the current policy model.
#        # log probs
#        action_log_probs = self.actor(sequences, num_actions, attention_mask)
#
#        # Calculate log_prob using the reference strategy (old strategy) for KL divergence calculation.
#        # init log probs
#        base_action_log_probs = self.initial_model(sequences, num_actions, attention_mask)
#
#        # values
#        # If a critic model exists, then obtain the value (used for subsequent advantage and return).
#        if self.critic is not None:
#            value = self.critic(sequences, num_actions, attention_mask)
#        else:
#            value = None
#
#        # Retrieve reward value (supports local and remote reward models)
#        if self.remote_rm_url is not None:
#            # remote RM             # 远程 reward model（将序列 decode 成文本后 POST 请求）
#            queries = self.tokenizer.batch_decode(sequences.cpu(), skip_special_tokens=False)
#            r = remote_rm_fn(self.remote_rm_url, queries=queries).to(device=action_log_probs.device)
#        else:
#            # local RM
#            # Local reward model, direct forward
#            r = self.reward_model(sequences, attention_mask)
#        
#        # Should KL divergence be calculated?
#        if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
#            kl = compute_approx_kl(
#                action_log_probs,
#                base_action_log_probs,
#                action_mask=action_mask,
#                use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
#            )
#        else:
#            kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=action_log_probs.device)
#        
#        # Record information dictionary (which can be used later for logging or reward post-processing).
#        info = {
#            "kl": masked_mean(kl, action_mask, retrieve_mask, dim=-1),
#            "reward": r,
#            "response_length": samples.response_length,
#            "total_length": samples.total_length,
#            "num_actions": num_actions,
#        }
#        # reset model state
#        # 将 actor 和 critic 切换回训练模式
#        self.actor.train()
#        if self.critic is not None:
#            self.critic.train()
#
#        # Returns an Experience object (advantage and return still need to be calculated later).
#        return Experience(
#            sequences,
#            action_log_probs,
#            base_action_log_probs,
#            value,
#            None,
#            None,
#            attention_mask,
#            action_mask,
#            info,
#            kl,
#        )

    @torch.no_grad()
    def filter(self, experiences: List[Experience], global_step: int, wandb:None) -> List[Experience]:
        """
        Filter experiences based on accuracy reward.

        Output:
        - experiences: List of filtered Experience
        """

        args = self.strategy.args
        # as in the original TIRESRAG-R1 code that uses "answer_rewards"
        accuracy_rewards = torch.cat([experience.info["answer_rewards"] for experience in experiences]) 
       # accuracy_rewards = torch.cat([experience.info["reward"] for experience in experiences])
        accuracy_rewards = accuracy_rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
        accuracy_rewards_min = torch.min(accuracy_rewards, dim=-1).values
        accuracy_rewards_max = torch.max(accuracy_rewards, dim=-1).values
        accuracy_counts = Counter(accuracy_rewards_min.tolist())
        print("Accuracy min distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(accuracy_counts.items())))
        accuracy_counts = Counter(accuracy_rewards_max.tolist())
        print("Accuracy max distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(accuracy_counts.items())))

        assert len(experiences) % len(accuracy_rewards) == 0
        group_len = args.n_samples_per_prompt
        grouped_experience = [experiences[i : i + group_len] for i in range(0, len(experiences), group_len)]
        filtered_experiences = []
        min_count = 0 
        max_count = 0 
        zhongjian_count = 0 
        for group, reward_min, reward_max in zip(grouped_experience, accuracy_rewards_min, accuracy_rewards_max):
            # if reward_max.item() - reward_min.item() <= args.accuracy_lower_bound: #如果最大最小值就差0.1，就认为这个group的reward都一样，就丢掉这个group
            #     zhongjian_count += args.n_samples_per_prompt
            if reward_max.item() <= args.accuracy_lower_bound:
                max_count += args.n_samples_per_prompt
                continue 
            elif reward_min.item() >= args.accuracy_upper_bound:
                min_count += args.n_samples_per_prompt
                continue
            else:
            # #if args.accuracy_lower_bound <= reward_min.item() <= args.accuracy_upper_bound:
                filtered_experiences.extend(group)

            # if reward_max.item() <= args.accuracy_lower_bound:
            #     max_count += args.n_samples_per_prompt
            #     continue 
            # elif reward_min.item() >= args.accuracy_upper_bound:
            #     min_count += args.n_samples_per_prompt
            #     continue
        print(f"Filtered { len(experiences) - len(filtered_experiences)} \% {(len(experiences) - len(filtered_experiences))/len(experiences)} experiences.")
        if self.strategy.is_rank_0():
            print('打印filter的step', global_step)
            log = {
                "train/filtered_>0.9_experience": min_count/ len(experiences),
                "train/filtered_<0.1_experience": max_count/ len(experiences),
                "train/(max-min)<0.1_experience": zhongjian_count/ len(experiences)
            }
            wandb.log(log, step=global_step)

        return filtered_experiences


    def filter_after_shaping(self, experiences, rewards, group_retrives_stds, global_step, wandb = None):
        """
        Filter experiences based on accuracy reward.

        Output:
        - experiences: List of filtered Experience
        """

        TH = 0.1 if global_step > 80 else 0.3

        args = self.strategy.args

        accuracy_rewards = torch.cat([experience.info["answer_rewards"] for experience in experiences]) 
        accuracy_rewards = accuracy_rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")

        final_step_rewards = [item for sublist in rewards for item in sublist]
        l = []
        for item in final_step_rewards:
            l.append(item[-1].unsqueeze(0))
        final_step_rewards = torch.cat(l) 
        final_step_rewards = final_step_rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
        final_step_rewards = final_step_rewards.std(-1, keepdim=True)
        
        # statistics based on answer rewards
        accuracy_rewards_min = torch.min(accuracy_rewards, dim=-1).values
        accuracy_rewards_max = torch.max(accuracy_rewards, dim=-1).values
        accuracy_counts = Counter(accuracy_rewards_min.tolist())
        print("Accuracy min distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(accuracy_counts.items())))
        accuracy_counts = Counter(accuracy_rewards_max.tolist())
        print("Accuracy max distribution:", " ".join(f"{k:.2f}:{v}" for k, v in sorted(accuracy_counts.items())))

        assert len(experiences) % len(accuracy_rewards) == 0
        group_len = args.n_samples_per_prompt
        grouped_experience = [experiences[i : i + group_len] for i in range(0, len(experiences), group_len)]
        assert len(grouped_experience) == final_step_rewards.size()[0]
        filtered_experiences = []
        filtered_rewards = []
        min_count = 0 
        max_count = 0 
        middle_count = 0
        zhongjian_count = 0 
        for e, (group, greward_std, reward_min, reward_max) in enumerate(zip(grouped_experience, final_step_rewards, accuracy_rewards_min, accuracy_rewards_max)):
            if greward_std < 0.03:

                if reward_max.item() <= args.accuracy_lower_bound:
                    max_count += args.n_samples_per_prompt
                elif reward_min.item() >= args.accuracy_upper_bound:
                    min_count += args.n_samples_per_prompt
                else:
                    middle_count += args.n_samples_per_prompt
                continue 
            else:
                filtered_experiences.extend(group)
                filtered_rewards.append(rewards[e])

        print(f"Filtered { len(experiences) - len(filtered_experiences)} \% {(len(experiences) - len(filtered_experiences))/len(experiences)} experiences.")
        if self.strategy.is_rank_0():
            print('打印filter的step', global_step)
            log = {
                "train/filtered_>0.9_experience": min_count/ len(experiences),
                "train/filtered_<0.1_experience": max_count/ len(experiences),
                "train/filtered_mid_experience": middle_count/ len(experiences)
            }
            wandb.log(log, step=global_step)

        return filtered_experiences, filtered_rewards


    def get_lets_scores(self, lets_score):
        print("Print the lets score before group_norm:",lets_score)
        max_rows = max(tensor.size(0) for tensor in lets_score)
        padded_scores = [torch.nn.functional.pad(tensor, (0, max_rows - tensor.size(0))) for tensor in lets_score]
        padded_scores = torch.stack(padded_scores, dim=0).to(device="cuda")
        if self.strategy.args.reshape_method == 'potential':
            padded_scores = padded_scores[:,1:] # remove intial element not used by LeTS
        print('lets padded_scores', padded_scores)
        # compute means of these
        mask = padded_scores!=0
        print('lets mask',mask)
        eltos = mask.sum(dim=1)
        print('lets eltos',eltos)
        feltos = eltos.float()
        feltos[feltos==0] = 1e-9
        mean = (padded_scores*mask).sum(dim=1)/feltos
        
        # step norm intermediate rewards
        #compute std masked
        std = padded_scores - mean.unsqueeze(1)
        std = torch.pow(std*mask, 2)
        std = std.sum(dim=1)/feltos
        std = torch.sqrt(std) + 1e-9
        norm_scores = (padded_scores - mean.unsqueeze(1)) / std.unsqueeze(1)
        norm_scores = norm_scores * mask

        return norm_scores

    @torch.no_grad()
    def process_experiences(self, experiences: List[Experience], global_step, total_training_steps) -> Tuple[List[Experience], List[torch.Tensor]]:
        """
        Process experiences, this can be used to filter out some experiences or do some processing on the rewards.

        Output:
        - experiences: List of Experience
        - rewards: List of rewards
        """
        """
        对多个 Experience 进行 reward 的后处理。
        用于根据策略（如 RLOO、group_norm 等）调整 reward，提升训练稳定性。

        输入：
        - experiences: 一个 Experience 对象列表

        输出：
        - 处理后的 experiences（结构不变）
        - 每个 Experience 中的 reward（tensor），可直接用于计算 return 和 advantage
        """
        args = self.strategy.args
        # reward shaping for RLOO
        # 针对 RLOO 策略的 reward 处理
        device = torch.cuda.current_device()
        if args.advantage_estimator == "rloo":
            # 合并所有 reward（形状变为 [batch_size]）
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            # 为每个样本记录是否回答正确（准确率信息）
            for experience in experiences:
                experience.info['acc_info'] = (experience.info["reward"] == 1).float().reshape(-1)
            # reshape 成 [n_prompts, n_samples_per_prompt]（每个 prompt 有多个 sample）
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
            # acc_info = (rewards == 1).float()
            # expe
            # 计算 leave-one-out baseline（去掉自身的平均 reward）
            baseline = (rewards.sum(-1, keepdim=True) - rewards) / (args.n_samples_per_prompt - 1)
            # 计算 advantage = reward - baseline
            rewards = rewards - baseline
            # reshape 回原始 shape，按 batch 切分
            rewards = rewards.flatten().to(device="cpu").chunk(len(experiences)) #将张量分成 len(experiences) 个相等大小的块
            return experiences, rewards, None

        # 针对 group normalization 策略
        elif args.advantage_estimator == "group_norm":
            group_retrives_stds = None
            if args.reward_shape:
                group_retrives_stds = []
                rewards = torch.cat([experience.info["reward"] for experience in experiences])
                rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
                print("Print the reward before group_norm:",rewards[0])
                answer_action  = torch.cat([experience.info["format_rewards"] for experience in experiences])
                answer_action = answer_action.reshape(-1, args.n_samples_per_prompt).to(device="cuda")

                retrieve_score = [rs for experience in experiences for rs in experience.retrieve_score]
                retrieve_score = [x for x in list_chunks(retrieve_score, args.n_samples_per_prompt)]
                print("Print the retriever score before group_norm:",retrieve_score[0])

                if 'glmatch' in args.intrinsic:
                    lets_score = [rs for experience in experiences for rs in experience.lets_score]
                    lets_score = [x for x in list_chunks(lets_score, args.n_samples_per_prompt)]


                if args.reshape_method in ['mtgrpo', 'sgrpo', 'potential', 'average']:
                    sdiversities = [divs for experience in experiences for divs in experience.state_diversity]
                    sdiversities = [x for x in list_chunks(sdiversities, args.n_samples_per_prompt)]

                    state_acoverage = [divs for experience in experiences for divs in experience.state_acoverage]
                    state_acoverage = [x for x in list_chunks(state_acoverage, args.n_samples_per_prompt)]

                    gdiversities = [divs for experience in experiences for divs in experience.group_diversity]
                    gdiversities = [x for x in list_chunks(gdiversities, args.n_samples_per_prompt)]

                for i in range(len(retrieve_score)):

                    if args.reshape_method in ['mtgrpo', 'sgrpo', 'potential', 'average'] and \
                        not ('gola' in args.intrinsic and 'nite' in args.intrinsic) and \
                        not ('rlredun' in args.intrinsic and 'glmatch' in args.intrinsic):

                        print('retrieve_score[i] enabled:', 'rets' in args.intrinsic, retrieve_score[i])
                        print('gdiversities[i]', gdiversities[i])
                        print('sdiversities[i] enabled:', 'sdiv' in args.intrinsic, sdiversities[i])
                        print('state_acoverage[i] enabled:', 'acov' in args.intrinsic, state_acoverage[i])
                        
                        i_gdiversities= []
                        for j in range(len(retrieve_score[i])):
                            #combine retriever scores with diversity scores at step level
                            divs = -1 * torch.log(gdiversities[i][j])
                            divs[-1] = 0 #reset last position reserved for reward step
                            if args.reshape_method in ['potential']:
                                divs = divs[1:-1].mean() if divs.size()[0] > 2 else divs[-1:]
                            else:
                                divs = divs[:-1].mean() if divs.size()[0] > 1 else divs[-1:]
                            i_gdiversities.append(divs * args.lambda_expl \
                                    * (1 / (1 + math.exp((global_step - total_training_steps * 0.9) / 10)))) # this is a decay used in TIRESRAG-R1

                            # combine rets, acov, sdiv
                            tmp_potential = torch.zeros_like(retrieve_score[i][j])
                            if 'rets' in args.intrinsic:
                                tmp_potential += retrieve_score[i][j] * (args.omega_u)
                            if 'sdiv' in args.intrinsic:
                                tmp_potential += (sdiversities[i][j] * (args.omega_d) \
                                    * (1 / (1 + math.exp((global_step - total_training_steps * 0.9) / 10)))) # this is a decay used in TIRESRAG-R1
                            if 'acov' in args.intrinsic:
                                tmp_potential += torch.log(state_acoverage[i][j] + 1) * (else args.omega_c)
                            retrieve_score[i][j] = tmp_potential

                        i_gdiversities = torch.Tensor(i_gdiversities).to(device="cuda")
                        print(f'shaping potentials "£$%"£$^%$%&^%$^&^%&* in [{i},{j}]', retrieve_score[i][j])

                    if args.reshape_method == 'mtgrpo' and \
                        not ('gola' in args.intrinsic and 'nite' in args.intrinsic):
                        i_process_r = []
                        for j in range(args.n_samples_per_prompt):
                            if retrieve_score[i][j].size()[0] > 2:
                                process_mean = retrieve_score[i][j][1:-1].mean(-1, keepdim=True)
                            else:
                                process_mean = retrieve_score[i][j][-1:]
                            i_process_r.append(process_mean)
                        if args.with_process_r:
                            i_process_r = torch.cat(i_process_r, dim=0).to(device="cuda")
                            print('Process rewards', i_process_r)

                        print("Print the retriever score before group_norm:",retrieve_score[i])
                        max_rows = max(tensor.size(0) for tensor in retrieve_score[i])
                        padded_scores = [torch.nn.functional.pad(tensor, (0, max_rows - tensor.size(0))) for tensor in retrieve_score[i]]
                        padded_scores = torch.stack(padded_scores, dim=0).to(device="cuda")
                        mask = padded_scores!=0
                        eltos = mask.sum(dim=0)
                        feltos = eltos.float()
                        feltos[feltos==0] = 1e-9
                        mean = (padded_scores*mask).sum(dim=0)/feltos
                        eltos[eltos<=1] = 0
                        eltos[eltos>1] = 1
                        mean = mean * eltos
                        stdvs = []
                        for s in range(padded_scores.size(1)):
                            if eltos[s] == 0:
                                stdvs.append(1e-9)
                                continue
                            zero_idxs = torch.nonzero(padded_scores[:,s], as_tuple=True)
                            stdvs.append(padded_scores[:,s][zero_idxs].std(dim=0, keepdim=True) + 1e-9)
                        std = torch.Tensor(stdvs).to(device="cuda")
                        padded_scores = (padded_scores - mean) / std
                        # segments in one rollout will receive the reward
                        padded_scores = padded_scores * 0.1
                        padded_scores = padded_scores * mask
                        
                        # combine outcome reward with sample diversity reward
                        # step or trajectory
                        print('i_gdiversities', i_gdiversities)
                        print('rewards[i]', rewards[i])
                        print('i_process_r', i_process_r)
                        rewards[i] += i_gdiversities
                        if args.with_process_r:
                            rewards[i] += (args.process_r_lambda_phi * i_process_r)

                        rewards[i] = (rewards[i] - rewards[i].mean(-1, keepdim=True)) / (rewards[i].std(-1, keepdim=True) + 1e-9)
                        print('normalised outcome reward + diversity (+process)', rewards[i])

                        for s in range(padded_scores.size(1)):
                            if eltos[s] == 0:
                                padded_scores[:, s] = rewards[i]
                            else:
                                padded_scores[:, s] += (0.9 * rewards[i])

                        for e, l in enumerate([x.size(0) for x in retrieve_score[i]]):
                            retrieve_score[i][e] = padded_scores[e][:l]
                            retrieve_score[i][e][-1] = rewards[i][e].item()
                        print("Print the intrinsic and outcome rewards after group_norm:",retrieve_score[i])

                    if args.reshape_method == 'mtgrpo' and \
                        ('gola' in args.intrinsic and 'nite' in args.intrinsic):
                        # baseline from MT-GRPO paper with gold answer in passages intrinsic reward

                        i_masks = []
                        for j in range(args.n_samples_per_prompt):
                            i_masks.append(torch.Tensor(([1] * (retrieve_score[i][j].size()[0]-1)) + [0]))
                        
                        print("Print the retriever score before group_norm:",retrieve_score[i])
                        max_rows = max(tensor.size(0) for tensor in retrieve_score[i])
                        padded_scores = [torch.nn.functional.pad(tensor, (0, max_rows - tensor.size(0))) for tensor in retrieve_score[i]]
                        padded_scores = torch.stack(padded_scores, dim=0).to(device="cuda")
                        print('intrinsic scores', padded_scores, 'device:',padded_scores.is_cuda)
                        i_masks = [torch.nn.functional.pad(tensor, (0, max_rows - tensor.size(0))) for tensor in i_masks]
                        i_masks = torch.stack(i_masks, dim=0).to(device="cuda")
                        i_eltos = i_masks.sum(dim=0)
                        mask = padded_scores!=0
                        eltos = mask.sum(dim=0) 
                        eltos = eltos!=0
                        eltos = eltos * i_eltos
                        feltos = eltos.float()
                        feltos[feltos==0] = 1e-9
                        print('FELTOS', feltos)
                        mean = (padded_scores*i_masks).sum(dim=0)/feltos
                        eltos[eltos<=1] = 0
                        eltos[eltos>1] = 1
                        mean = mean * eltos
                        print('MEAN w/mask', mean)
                        stdvs = []
                        for s in range(padded_scores.size(1)):
                            if eltos[s] == 0:
                                stdvs.append(1e-9)
                                continue
                            col_eltos = padded_scores[:,s][i_masks[:,s].bool()]
                            stdvs.append(col_eltos.std(dim=0, keepdim=True) + 1e-9)
                        std = torch.Tensor(stdvs).to(device="cuda")
                        print('intrinsic std', std)
                        for s in range(padded_scores.size(1)):
                            padded_scores[:,s] = padded_scores[:,s] * eltos[s]
                        padded_scores = (padded_scores - mean) / std
                        # segments in one rollout will receive the reward
                        print('intrinsic after norm', padded_scores)
                        padded_scores = padded_scores * 0.05 * i_masks
                        print('intrinsic after norm masked', padded_scores)
                        
                        # group normalise segment rewards
                        rewards[i] = (rewards[i] - rewards[i].mean(-1, keepdim=True)) / (rewards[i].std(-1, keepdim=True) + 1e-9)
                        print('normalised outcome reward', rewards[i])

                        for s in range(padded_scores.size(1)):
                            if eltos[s] == 0 or padded_scores[:, s][padded_scores[:, s] != 0].numel() == 0:
                                padded_scores[:, s] = rewards[i]
                            else:
                                padded_scores[:, s] += (0.97 * rewards[i])
                        
                        for e, l in enumerate([x.size(0) for x in retrieve_score[i]]):
                            retrieve_score[i][e] = padded_scores[e][:l]
                            retrieve_score[i][e][-1] = rewards[i][e].item()
                        print("Print the intrinsic and outcome rewards after group_norm:",retrieve_score[i])       
                        #time.sleep(20)                 

                    elif args.reshape_method == 'sgrpo' and \
                        not ('rlredun' in args.intrinsic and 'glmatch' in args.intrinsic) : 
                        #step-wise grpo as in LeTS but with our state potentials

                        i_process_r = []
                        for j in range(args.n_samples_per_prompt):
                            if retrieve_score[i][j].size()[0] > 2:
                                process_mean = retrieve_score[i][j][1:-1].mean(-1, keepdim=True)
                            else:
                                process_mean = retrieve_score[i][j][-1:]
                            i_process_r.append(process_mean)
                        if args.with_process_r:
                            i_process_r = torch.cat(i_process_r, dim=0).to(device="cuda")
                            print('Process rewards', i_process_r)

                        print("Print the retriever score before group_norm:",retrieve_score[i])
                        max_rows = max(tensor.size(0) for tensor in retrieve_score[i])
                        padded_scores = [torch.nn.functional.pad(tensor, (0, max_rows - tensor.size(0))) for tensor in retrieve_score[i]]
                        padded_scores = torch.stack(padded_scores, dim=0).to(device="cuda")
                        print('padded_scores', padded_scores)
                        # compute means of these
                        mask = padded_scores!=0
                        print('mask',mask)
                        eltos = mask.sum(dim=1)
                        print('eltos',eltos)
                        feltos = eltos.float()
                        feltos[feltos==0] = 1e-9
                        mean = (padded_scores*mask).sum(dim=1)/feltos
                        
                        # step norm intermediate rewards
                        #compute std masked
                        std = padded_scores - mean.unsqueeze(1)
                        std = torch.pow(std*mask, 2)
                        std = std.sum(dim=1)/feltos
                        std = torch.sqrt(std) + 1e-9
                        norm_scores = (padded_scores - mean.unsqueeze(1)) / std.unsqueeze(1)
                        norm_scores = norm_scores * mask
                        
                        # combine outcome reward with sample diversity reward
                        # step or trajectory
                        print('i_gdiversities', i_gdiversities)
                        print('rewards[i]', rewards[i])
                        print('i_process_r', i_process_r)
                        rewards[i] += i_gdiversities
                        if args.with_process_r:
                            rewards[i] += (args.process_r_lambda_phi * i_process_r)

                        # group normalise segment rewards
                        rewards[i] = (rewards[i] - rewards[i].mean(-1, keepdim=True)) / (rewards[i].std(-1, keepdim=True) + 1e-9)
                        print('normalised outcome reward + diversity (+process)', rewards[i])                        
                        
                        norm_scores = ( 1 + (torch.sign(rewards[i].unsqueeze(1)) * 0.1 * norm_scores)) * rewards[i].unsqueeze(1) # lambda =0.1 as in LeTS paper
                        print('combined', norm_scores)

                        for e, l in enumerate([x.size(0) for x in retrieve_score[i]]):
                            retrieve_score[i][e] = norm_scores[e][:l]
                        print("Print the intrinsic and outcome rewards after group_norm:",retrieve_score[i])
                        #time.sleep(20)

                    elif args.reshape_method == 'sgrpo' and \
                        ('rlredun' in args.intrinsic and 'glmatch' in args.intrinsic) : 
                        # baseline from LeTS paper with roll-out level redundancy and group level match

                        norm_scores = self.get_lets_scores(lets_score[i])
                        
                        print('rewards[i]', rewards[i])
                        rewards[i] = (rewards[i] - rewards[i].mean(-1, keepdim=True)) / (rewards[i].std(-1, keepdim=True) + 1e-9)
                        print('normalised outcome reward', rewards[i])                        
                        
                        norm_scores = ( 1 + (torch.sign(rewards[i].unsqueeze(1)) * 0.1 * norm_scores)) * rewards[i].unsqueeze(1) # lambda =0.1 as in LeTS paper
                        print('combined', norm_scores)

                        for e, l in enumerate([x.size(0) for x in lets_score[i]]):
                            retrieve_score[i][e] = norm_scores[e][:l]
                        print("Print the intrinsic and outcome rewards after group_norm:",retrieve_score[i])
                        #time.sleep(20)

                    
                    elif args.reshape_method == 'average':
                        # here add the retriever scores
                        max_rows = max(tensor.size(0) for tensor in retrieve_score[i])
                        padded_scores = [torch.nn.functional.pad(tensor, (0, max_rows - tensor.size(0))) for tensor in retrieve_score[i]]
                        padded_scores = torch.stack(padded_scores, dim=0).to(device="cuda")
                        mask = padded_scores!=0
                        eltos = mask.sum(dim=1)
                        feltos = eltos.float()
                        feltos[feltos==0] = 1e-9
                        # add average retrieval score to outcome + diversity rewards. Both rewards[i] and 'mean' are 1D vector
                        rewards[i] = rewards[i] + ((padded_scores*mask).sum(dim=1)/feltos) + i_gdiversities
                        print('updated rewards', rewards[i])
                        #reward normalisation below

                    elif args.reshape_method == 'potential':
                        i_process_r = []
                        i_masks = []
                        for j in range(args.n_samples_per_prompt):
                            if retrieve_score[i][j].size()[0] > 2:
                                process_mean = retrieve_score[i][j][1:-1].mean(-1, keepdim=True)
                            else:
                                process_mean = retrieve_score[i][j][-1:]
                            i_process_r.append(process_mean)
                            if not args.goal_state_mod:
                                retrieve_score[i][j][-1] = rewards[i][j].item()
                            i_masks.append(torch.Tensor([1] * (retrieve_score[i][j].size()[0])))
                            # the mask is on all retrieval steps
                        if args.with_process_r:
                            i_process_r = torch.cat(i_process_r, dim=0).to(device="cuda")
                            print('Process rewards', i_process_r)

                        max_rows = max(tensor.size(0) for tensor in retrieve_score[i])
                        padded_scores = [torch.nn.functional.pad(tensor, (0, max_rows - tensor.size(0))) for tensor in retrieve_score[i]]
                        padded_scores = torch.stack(padded_scores, dim=0).to(device="cuda")
                        previous_padded_scores = torch.roll(padded_scores, shifts=1, dims=1)
                        previous_padded_scores[:,0] = 0. #reset first dimension
                        padded_scores = padded_scores - previous_padded_scores
                        padded_scores = padded_scores[:,1:] #remove the first dimension that is only for the initial state related constant
                        print('Potential Functions', padded_scores)
                        norm_scores = padded_scores

                        if args.goal_state_mod:
                            for e, l in enumerate([x.size(0)-2 for x in retrieve_score[i]]):
                                state_reward = rewards[i][e].item()
                                if state_reward > norm_scores[e][l].item():
                                    norm_scores[e][l] = state_reward * (-1 * norm_scores[e][l])
                        print('goal state potential difference MODULATED', norm_scores)

                        # combine outcome reward with sample diversity reward
                        # step or trajectory
                        print('i_gdiversities', i_gdiversities)
                        print('rewards[i]', rewards[i])
                        print('i_process_r', i_process_r)
                        if args.with_group_d:
                            rewards[i] += i_gdiversities #weight already applied
                        if args.with_process_r:
                            rewards[i] += (args.process_r_lambda_phi * i_process_r)

                        # group normalise segment rewards
                        rewards[i] = (rewards[i] - rewards[i].mean(-1, keepdim=True)) / (rewards[i].std(-1, keepdim=True) + 1e-9)
                        print('normalised outcome reward + diversity (+process)', rewards[i])

                        if 'glmatch' in args.intrinsic:
                            # state self-supervision from LeTS combined with PACE potential
                            print('lets_score[i]', lets_score[i])
                            lets_norm_scores = self.get_lets_scores(lets_score[i])
                            lets_norm_scores = ( 1 + (torch.sign(rewards[i].unsqueeze(1)) * 0.05 * lets_norm_scores)) * rewards[i].unsqueeze(1) # lambda =0.05 instead of 0.1 as in LeTS paper
                            print('LeTS combined', lets_norm_scores)
                             # potential based shaping
                            norm_scores = args.alpha * norm_scores + lets_norm_scores
                            print('combined', norm_scores)
                        else:
                            # potential based shaping
                            norm_scores = args.alpha * norm_scores + rewards[i].unsqueeze(1)
                            print('combined', norm_scores)

                        for e, l in enumerate([x.size(0)-1 for x in retrieve_score[i]]):
                            retrieve_score[i][e] = norm_scores[e][:l]
                        print("Print the intrinsic and outcome rewards after group_norm:",retrieve_score[i])
                        #time.sleep(20)

                    else:
                        print('Reward shaping method not implemented', args.reshape_method)
                        kill
                
                # Advantage standarisation for those still missing this step
                if args.reshape_method == 'average':
                    rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
                    for i in range(len(retrieve_score)):
                        for j in range(args.n_samples_per_prompt):
                            retrieve_score[i][j][retrieve_score[i][j]==retrieve_score[i][j]] = rewards[i][j].item()


                rewards = retrieve_score

                
                #if not args.naive_grpo: # apply only if not naive GRPO
                #    difficulty = torch.cat([experience.info["difficulty"] for experience in experiences])
                #    penalty = torch.cat([experience.info["penalty"] for experience in experiences])
                #    rewards = ((rewards - penalty) * difficulty).chunk(len(experiences)) 
                #    print("Print the reward with penalty and difficulty added:",rewards)
                #else:
                #    print("As Naive GRPO... omit penalty and difficulty calculation.")
                #    rewards = rewards.chunk(len(experiences))
                print("As Reward-Shaping GRPO... omit penalty and difficulty calculation. (At least for the moment!)")
                #print('group_retrives_stds', group_retrives_stds)
                #time.sleep(10)
            else:
                rewards = torch.cat([experience.info["reward"] for experience in experiences])
                rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
                print("Print the reward before group_norm:",rewards)
                # Standardize each sample group (mean = 0, variance = 1).
                rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)
                # Reshape back to the original shape, then split by batch.
                rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
                print("Print the reward after group_norm:",rewards)
                rewards = torch.cat(rewards)
                if not args.naive_grpo: # apply only if not naive GRPO
                    difficulty = torch.cat([experience.info["difficulty"] for experience in experiences])
                    penalty = torch.cat([experience.info["penalty"] for experience in experiences])
                    rewards = ((rewards - penalty) * difficulty).chunk(len(experiences)) 
                    print("Print the reward with penalty and difficulty added:",rewards)
                else:
                    print("As Naive GRPO... omit penalty and difficulty calculation.")
                    rewards = rewards.chunk(len(experiences))

                ### Calculate advantage_valued_portion
                # Generate 0/1 mask
                for exp, ri in zip(experiences, rewards):
                    val = (ri != 0).float()
                    exp.info['advantage_valued_porportion'] = val
            return experiences, rewards, group_retrives_stds

        # 针对 REINFORCE + baseline 策略
        elif args.advantage_estimator == "reinforce_baseline":
            # REINFORCE++-baseline removed the / std and K3 kl loss in GRPO.
            # `/ std` is not needed in RL variance reduction theory, and `k3 KL` has a larger variance than `k1 KL` under a categorical distribution.
            rewards = torch.cat([experience.info["reward"] for experience in experiences])
            rewards = rewards.reshape(-1, args.n_samples_per_prompt).to(device="cuda")
           
            # 直接减去均值作为 baseline（不做归一化）
            rewards = rewards - rewards.mean(-1, keepdim=True)
            difficulty = torch.cat([experience.info["difficulty"] for experience in experiences]) #对reinforce++而言，只有一个ppo，所以没有difficulty的计算了.
            penalty = torch.cat([experience.info["penalty"] for experience in experiences])

            # reshape 回原始 shape，按 batch 切分
            rewards = rewards.reshape(-1).to(device="cpu").chunk(len(experiences))
            rewards = torch.cat(rewards)
            rewards = (rewards - penalty).chunk(len(experiences))

            return experiences, rewards, None

        # CZP Update 1229 For GRPO
        # if self.advantage_estimator in ["group_norm"]:
        #     return [sum(experiences)], [experience.info["reward"] for experience in [sum(experiences)]]
        # End of Update

        # default rewards
        # By default, the original reward is returned directly.
        return experiences, [experience.info["reward"] for experience in experiences], None

    @torch.no_grad()
    def get_advantages_and_returns(
        self,
        values: torch.Tensor,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        gamma: float,
        lambd: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Advantage 的计算方式如下：
        Adv_t = δ_t + (γλ) * δ_{t+1} + (γλ)^2 * δ_{t+2} + ...
        其中 δ_t = r_t + γ * V_{t+1} - V_t

        Return 的计算方式是：
        Ret_t = Adv_t + V_t
        参数：
        - values: critic 输出的每个 token 的 V 值，形状为 (B, A)
        - rewards: 每个 token 的即时奖励，形状为 (B, A)
        - action_mask: 用于屏蔽 padding 的二值 mask，形状为 (B, A)
        - gamma: 折扣因子
        - lambd: GAE 的平衡系数

        返回：
        - advantages: GAE 算法计算出的优势函数值，形状为 (B, A)
        - returns: 折扣累计回报（用于 value 回归目标），形状为 (B, A)
        """
        # kill
        if isinstance(values, list):
            # 处理 packed 格式的情况，values 和 rewards 是列表形式
            # packing samples
            # TODO: this is slow...
            advantages = []
            returns = []
            for v, r in zip(values, rewards):
                adv, ret = self.get_advantages_and_returns(v.unsqueeze(0), r.unsqueeze(0), action_mask, gamma, lambd)
                advantages.append(adv.squeeze(0))
                returns.append(ret.squeeze(0))
            return advantages, returns

        lastgaelam = 0  # 用于存储上一个 timestep 的 GAE 累计值
        advantages_reversed = []
        response_length = rewards.size(1)

        # Mask invalid responses
        # Use action_mask to mask out invalid tokens (padding).
        if action_mask is not None:
            values = action_mask * values
            rewards = action_mask * rewards
        
        # Calculate GAE from back to front
        for t in reversed(range(response_length)):
            nextvalues = values[:, t + 1] if t < response_length - 1 else 0.0
            delta = rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lambd * lastgaelam
            advantages_reversed.append(lastgaelam)
        # Reverse the reversed advantage sequence to its normal order.
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values # return = advantage + value（作为 critic 回归目标）
        return advantages.detach(), returns

    @torch.no_grad()
    def get_cumulative_returns( # returns: 累计回报 它是某个状态开始之后，模型可以期望获得的 总奖励,用来训练critic，即模型对未来奖励的预测。而advantage = return - value , value是critic预测当前状态的期望汇报。advantage衡量某个 动作到底比模型期望的好多少。advantage用来训练actor，returns用来训练critic
        self,
        rewards: torch.Tensor,
        action_mask: torch.Tensor,
        retrieve_mask: torch.Tensor,
        gamma: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function that computes advantages and returns from rewards using REINFORCE.
        REINFORCE uses cumulative returns without the GAE (Generalized Advantage Estimation).

        Input:
        - rewards: Tensor of shape (batch_size, response_size)
        - action_mask: Tensor of shape (batch_size, response_size), binary mask
        - gamma: discount factor

        Output:
        - returns: Tensor of shape (batch_size, response_size)
        """
        """
        The cumulative return is calculated using the REINFORCE algorithm (without using GAE).

        For the return value at each position t, the calculation method is as follows:
            R_t = r_t + γ * r_{t+1} + γ^2 * r_{t+2} + ...

        输入：
        - rewards: The instant reward tensor has a shape of (B, A).
        - action_mask: 用于屏蔽 padding 的掩码，形状为 (B, A)
        - retrieve_mask: 用于屏蔽检索区域（不参与 reward），形状为 (B, A)
        - gamma: 折扣因子

        输出：
        - returns: 与 rewards 同形状，表示每个 token 的累计回报
        """
        if isinstance(rewards, list):
            # packing samples
            # TODO: this is slow...
            # 如果是 packed 格式，rewards 是一个 list，每条样本长度不同

            returns = []
            for p, r in enumerate(rewards):
                # Calculate the return value for each sample separately (recursive call).
                ret = self.get_cumulative_returns(r.unsqueeze(0), action_mask, retrieve_mask[p].unsqueeze(0), gamma)
                returns.append(ret.squeeze(0))
            return returns

        response_length = rewards.size(1)
        returns = torch.zeros_like(rewards)
        cumulative_return = torch.zeros(rewards.size(0), device=rewards.device)

        # Mask invalid responses if action_mask is provided
        # 应用 action mask：屏蔽 padding 区域

        if action_mask is not None:
            rewards = action_mask * rewards

        print("cumulative-reward:",rewards)

        # Calculate returns by accumulating
        # discounted rewards
        # The reward is accumulated progressively from the end to the beginning, multiplied by gamma.
        for t in reversed(range(response_length)):
            cumulative_return = rewards[:, t] + gamma * cumulative_return
            returns[:, t] = cumulative_return
        # Apply retrieve mask: mask out the return values ​​of the searched content area (this part is not included in the reward calculation).）
        if retrieve_mask is not None:
            returns = returns * retrieve_mask
        print('returns for printed models:', returns)
        return returns


class RemoteExperienceMaker(NaiveExperienceMaker):
    def __init__(self, *args, vllm_engines: List = None, packing_samples=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.vllm_engines = vllm_engines # The vLLM engine used for generation (optional)
        self.packing_samples = packing_samples # Whether to enable packed format (prompt + response without padding)
        self.question_steps = {}
        args = self.strategy.args
        self.silver_passages_per_idx = {}
        if args.approx_coverage:
            with jsonlines.open(args.approx_coverage, 'r') as jsonl_f:
                train_data = [obj for obj in jsonl_f]
                for item in train_data:
                    self.silver_passages_per_idx[item['idx']] = {"silver_infohash": item["silver_infohash"],
                                "state_acoverage": item['state_acoverage'], #coverage for the zero_initial state
                                "ret_doc_score": item['ret_doc_score'] #retrieval score for the zero_initial state
                                }

    @torch.no_grad()
    def make_experience_list(self, all_prompts: Union[str, List[str]], global_step: int,wandb: None, total_training_steps: int, **generate_kwargs) -> List[Experience]:
        # 如果开启性能评估，记录时间统计项

        if self.strategy.args.perf:
            self.perf_stats = {
                "generate_time": 0,
                "actor_value_rm_time": 0,
                "wait_time": 0,
            }

        # 继承基类生成完整的 experience list
        experiences = super().make_experience_list(all_prompts, global_step, wandb, total_training_steps, **generate_kwargs) #一个list，只有[0]是一个tensor，装64个experience
        # 如果有 critic 模型，则将 experience 异步发送到远程 critic
        if self.critic is not None:
            for experience in experiences:
                # send experience to critic
                experience_cpu = deepcopy(experience)
                experience_cpu.to_device("cpu")
                self._ref = self.critic.append.remote(experience_cpu) #所有远程对象的方法（包括模型方法）都需要用 .remote() 调用。
        return experiences

    @torch.no_grad()
    def generate_samples(self, all_prompts: List[str], **generate_kwargs) -> List[Samples]:
        """
        Generate samples and return in batches.

        When not using vllm, we will fallback to the default implementation,
        in which actor will be used to generate samples.
        """
        if self.vllm_engines is None:
            return super().generate_samples(all_prompts, **generate_kwargs)

        # return self._generate_vllm(all_prompts, **generate_kwargs)
        return self._generate_vllm_with_retrieve(all_prompts, **generate_kwargs)


    @torch.no_grad()
    def make_experience(self, samples_list: Samples|List[Samples], global_step: int, total_training_steps: int) -> Experience:
        # Compared to the base class, this example uses ray for remote distributed model calls to extract information such as logprobs and rewards.
        """
        Turn samples into experience by calculating logprobs, values, rewards, and kl divergence.
        """
        """
        Convert Samples to Experience (remote version)
        Core functions:
        - Use actor to calculate log_probs locally
        - Use the remote initial_model to compute the reference strategy log_probs
        - Calculate value using a remote critic
        - Retrieve rewards using a remote reward model or API.
        - Calculate KL divergence
        """

        self.actor.eval() #self.actor不是remote对象，Actor = ray.remote(Actor) self.actor = Actor.remote(...) # actor 是本地对象，直接切换为 eval 模式

        device = torch.cuda.current_device()
        expeiences = []
        queries_list = [] 
        exit_flags_list = []
        ### Used alone for calculating reward
        r_refs = []
        for samples in samples_list:
        # extract values from samples
            sequences = samples.sequences
          #  print('Print the sequence length (sequences.shape) from make_experience. # This is because micro_rollout_batch_size packs the sequences together, and attention_mask is used to distinguish different examples.
            attention_mask = samples.attention_mask
            action_mask = samples.action_mask
            retrieve_mask = samples.retrieve_mask
            num_actions = samples.num_actions
            packed_seq_lens = samples.packed_seq_lens
            exit_flags = samples.exit_flag.to("cpu").tolist()
            sequences_cpu, attention_mask_cpu = (
                sequences.to("cpu"),
                attention_mask.to("cpu"),
            )
                    # rewards     # reward：支持远程 reward model，也支持 reward_model 为 ray actor 列表     
            # support remote RM API with ray
            if not self.remote_rm_url:
                for rm in self.reward_model:
                    r_refs.append(rm.forward.remote(sequences_cpu, attention_mask_cpu, packed_seq_lens=packed_seq_lens))
            else:
                # remote RM   # The input to the remote RM API is decoded text queries.
                if not self.packing_samples: # If packed format is not enabled, decode directly.
                    queries = self.tokenizer.batch_decode(sequences_cpu, skip_special_tokens=False)
                else:
                    sequences_list = []
                    offset = 0
                    tokens_list = sequences_cpu.tolist()[0]
                    for length in packed_seq_lens: #So this is an operation used to unpack the pack...
                        sequences_list.append(tokens_list[offset : offset + length])
                        offset += length
                    queries = self.tokenizer.batch_decode(sequences_list, skip_special_tokens=False)
                    queries_list.extend(queries)
                    exit_flags_list.extend(exit_flags)
        for rm in self.remote_rm_url:
            r = remote_rm_fn_ray.remote(rm, queries=queries_list, group_size=self.args.n_samples_per_prompt, exit_flags=exit_flags_list, global_step = global_step, total_training_steps=total_training_steps, out_verif=self.args.naive_grpo, reshaping=self.args.reward_shape) #The main modification is here, making the query a query that includes all samples of the same question, instead of just a single query.
            r_refs.append(r)
        r_rewards = ray.get(r_refs) #len(r_rewards) = [micro_rollout_batch_size * n_samples_per_prompt]
        r_rewards = r_rewards[0] #Since there is only one reward function, we only take 0.
        format_rewards = r_rewards["纯format分数"]
        answer_rewards = r_rewards["纯answer分数"]
        previous_right_final_incorrect_results_wait_enough_information = r_rewards["前面对了后面错了有足够信息"]
        previous_incorrect_final_right_results_wait_enough_information = r_rewards["前面错了后面对了有足够信息"]
        previous_right_final_incorrect_results_wait_not_enough_information = r_rewards["前面对了后面错了没有足够信息"]
        previous_incorrect_final_right_results_wait_not_enough_information = r_rewards["前面错了后面对了没有足够信息"]
        no_change_results = r_rewards["没有反思的"]
        no_change_reflect_results = r_rewards["没有改变的reflect"]
        exit_lose_reward = r_rewards["exit_lose_reward"]
        sufficient_premilinary_match = r_rewards["sufficient_premilinary_match"]
        previous_right_final_incorrect = r_rewards["反思前正确的，后面错误的"]
        previous_incorrect_final_right = r_rewards["反思前错误的，后面正确的"]
        thinking_rewards = r_rewards["thinking_reward"]
        difficulty = r_rewards["difficulty"]
        penalty = r_rewards["penalty"]
        # print('打印make experience里面的r_rewards长度', len(r_rewards))
        for index, samples in enumerate(samples_list): #len(samples_list) = n_samples_per_prompt  因为我在外面的make_experience_list的for循环是这么写的
        # extract values from samples
            sequences = samples.sequences
            attention_mask = samples.attention_mask
            action_mask = samples.action_mask
            retrieve_mask = samples.retrieve_mask
            retrieve_score_segments = samples.retrieve_score_segments
            retrieve_score = samples.retrieve_score
            lets_score = samples.lets_score
            retrieve_num = samples.retrieve_num
            num_actions = samples.num_actions
            packed_seq_lens = samples.packed_seq_lens
            # Remote: Get base log_probs and critic value # Set the initial model to remote execution and migrate input to the CPU to avoid GPU memory overflow.
            start = time.time()
            sequences_cpu, attention_mask_cpu = (
                sequences.to("cpu"),
                attention_mask.to("cpu"),
            )

            # init log probs
            base_action_log_probs_ref = self.initial_model.forward.remote(
                sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
            )

            # values     # critic 值，如果启用 colocate_critic_reward，要先强制执行完释放显存
            if self.critic is not None:
                value_ref = self.critic.forward.remote(
                    sequences_cpu, num_actions, attention_mask_cpu, packed_seq_lens=packed_seq_lens
                )
                # avoid CUDA OOM when colocate models #Critic 和 Reward model 在同一个 GPU，必须交替执行，防止显存爆掉
                if self.strategy.args.colocate_critic_reward:
                    ray.get([value_ref]) #ray.get() 会强制等待这个远程任务完成并释放资源
                    ray.get([self.critic.empty_cache.remote()])
            else:
                value_ref = ray.put(None)

            if self.strategy.args.colocate_actor_ref:
                ray.get([base_action_log_probs_ref])
                ray.get([self.initial_model.empty_cache.remote()])


            # log probs     # 本地 actor 计算 log probs
            action_log_probs = self.actor(sequences, num_actions, attention_mask, packed_seq_lens=packed_seq_lens)

            actor_value_rm_time = time.time() - start

            # wait initial/critic/reward model done
            start = time.time()
            # Wait for the remote task to complete, and retrieve log_probs, value, and reward. 
            # ref_values = ray.get([base_action_log_probs_ref, value_ref] + r_refs)
            ref_values = ray.get([base_action_log_probs_ref, value_ref])
            wait_time = time.time() - start

            sum_of_rewards = r_rewards["总的分数"] # this is the 'scores' variable computed in answer_reflection_reward.py (comination of several rewards)
            rewards_list = []
            for jdex in range(len(packed_seq_lens)):
                rewards_list.append(sum_of_rewards[index*len(packed_seq_lens) + jdex])
            # print(f'打印第{index}个rewards_list', rewards_list)
            rewards = [torch.tensor(rewards_list)]      # [torch.tensor([r_rewards[0][index]])] ## 目前这里只考虑一个远程reward 计算方式,所以只取0 这个0目前是放在前面就替代了.,而且要是list[list]格式. 然后第一维度代表奖励计算方式个数 也就是remote_rm_url的个数,第二个维度才代表不同例子的奖励。
            base_action_log_probs, value = ref_values[0], ref_values[1]
            base_action_log_probs = base_action_log_probs.to(device)
            if value is not None:
                value = value.to(device)
            rewards = [r.to(device) for r in rewards]
            # print('打印make_experience里面的rewards长度', len(rewards), len(rewards[0]))
            r = self.reward_fn(rewards) if len(rewards) > 0 else rewards[0] 
            # print('打印make_experience里面的rewards长度', len(r))

            # avoid CUDA OOM when colocate models
            if self.strategy.args.colocate_critic_reward and not self.remote_rm_url:
                ray.get([self.reward_model[0].empty_cache.remote()])

            if self.strategy.args.colocate_actor_ref:
                torch.cuda.empty_cache()

            if (self.initial_model is not None) and (not self.strategy.args.use_kl_loss):
                kl = compute_approx_kl(
                    action_log_probs,
                    base_action_log_probs,
                    action_mask=action_mask,
                    retrieve_mask=retrieve_mask,
                    use_kl_estimator_k3=self.strategy.args.use_kl_estimator_k3,
                )
            else:
                kl = torch.zeros_like(action_log_probs, dtype=action_log_probs.dtype, device=device)

            if not self.packing_samples:
                kl_mean = masked_mean(kl, action_mask,retrieve_mask=retrieve_mask, dim=-1,)
            else:
                # convert tensor into list of tensors so that it's easier to manipulate
                # within dataset.
                sequences = unpacking_samples(sequences, packed_seq_lens)
                attention_mask = None     # In packed format, attention_mask is uniformly set to None (padding position mask is no longer needed).
                action_log_probs = unpacking_samples(action_log_probs, num_actions) #    # The action_log_probs are split into a tensor for each sample (because the action length is different for each sample).
                retrieve_mask = unpacking_samples(retrieve_mask, num_actions)
                retrieve_score_segments = unpacking_samples_list(retrieve_score_segments, [int((x*2)+1) for x in retrieve_num.tolist()])
                #print('UNPACKING MASK SEGMENTS TO CREATE EXPERIENCE DATA', retrieve_score_segments)
                #TODO: we know that we are using only 1 seq, but to be consistent should implement an unpacking for when the list is a list of tensors of differnet sizes
                addit_elto = 2 if (self.args.reshape_method in ['potential']) else 1
                retrieve_score = unpacking_samples(retrieve_score, [int(x+addit_elto) for x in retrieve_num.tolist()]) # is a list of tensors of different lentgths
                lets_score = unpacking_samples(lets_score, [int(x+addit_elto) for x in retrieve_num.tolist()]) # is a list of tensors of different lentgths
                #print('RETRIEVE SCORES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~', index, retrieve_score)

                if value is not None: #    # If there is a value (critic output), it should also be split.
                    value = unpacking_samples(value, num_actions)

                if base_action_log_probs is not None:
                    base_action_log_probs = unpacking_samples(base_action_log_probs, num_actions)

                kl = unpacking_samples(kl, num_actions) #    # KL splits into a tensor for each sample.
                kl_mean = torch.zeros(len(kl), device=device)  # Create a tensor beforehand with a length of kl.
                for i, each_kl in enumerate(kl): #    # Iterate through each sample and calculate the corresponding masked KL average.
                    kl_mean[i] = masked_mean(each_kl, action_mask,retrieve_mask=retrieve_mask[i], dim=-1,)
                #print('kl_mean', kl_mean)

                if not self.strategy.args.use_kl_loss: #   # 如果禁用了 KL 损失，直接将 base_action_log_probs 设为 None（后续不使用）
                    base_action_log_probs = None

            info_ret_score = retrieve_score[0].mean(-1, keepdim=True) if retrieve_score[0].size()[0] > 1 else torch.zeros_like(torch.empty(1), device=device)
            if len(samples.group_diversity.size()) == 2:
                info_group_div = samples.group_diversity.mean(-1, keepdim=True) if samples.group_diversity.size()[1] > 1 else torch.zeros_like(torch.empty(1), device=device)
            else:
                info_group_div = samples.group_diversity #.mean(-1, keepdim=True) if samples.group_diversity.size()[0] > 1 else torch.zeros_like(torch.empty(1), device=device)
            info_acoverage = samples.state_acoverage.mean(-1, keepdim=True) if samples.state_acoverage.size()[1] > 1 else torch.zeros_like(torch.empty(1), device=device)            
            info_state_div = samples.state_diversity.mean(-1, keepdim=True) if samples.state_diversity.size()[1] > 1 else torch.zeros_like(torch.empty(1), device=device)            
 
            info = {
                "kl": kl_mean,
                "reward": r, # The reward value for each sample (which may be a scalar or a vector, depending on the reward model).
                "response_length": samples.response_length,
                "total_length": samples.total_length,
                "num_actions": num_actions, # The number of actions per sample (i.e., the number of tokens in the response, which is similar in meaning to response_length).
                "retrieve_num": samples.retrieve_num, # The number of fragments retrieved in each sample (i.e., how many times <|begin_of_documents|> is included).
                "pure_response_length": samples.pure_response_length, # The number of tokens in the non-retrieval portion of the response for each sample (actual response length).
                "format_rewards": torch.tensor([format_rewards[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "answer_rewards": torch.tensor([answer_rewards[index]], device=device, dtype=torch.float), #   Because the packing needs to be unpacked during buffer appending later, we need to add [] here to form a list.
                "previous_right_final_incorrect_rewards_wait_enough_information": torch.tensor([previous_right_final_incorrect_results_wait_enough_information[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "previous_incorrect_final_right_rewards_wait_enough_information": torch.tensor([previous_incorrect_final_right_results_wait_enough_information[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "previous_right_final_incorrect_rewards_wait_not_enough_information": torch.tensor([previous_right_final_incorrect_results_wait_not_enough_information[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "previous_incorrect_final_right_rewards_wait_not_enough_information": torch.tensor([previous_incorrect_final_right_results_wait_not_enough_information[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "no_reflect_porportion": torch.tensor([no_change_results[index]], device=device, dtype=torch.float), # I didn't do packing here because micro_rollout_batch_size = 1, so I directly called the index.
                "no_change_with_reflect_rewards": torch.tensor([no_change_reflect_results[index]], device=device, dtype=torch.float), # I didn't do packing here because micro_rollout_batch_size = 1, so I directly called the index.
                "exit_lose_reward": torch.tensor([exit_lose_reward[index]], device=device, dtype=torch.float), # I didn't do packing here because micro_rollout_batch_size = 1, so I directly called the index.
                "sufficient_premilinary_match": torch.tensor([sufficient_premilinary_match[index]], device=device, dtype=torch.float), #  I didn't do packing here because micro_rollout_batch_size = 1, so I directly called the index.
                "exit_porpotions": samples.exit_flag,
                "answer_number": samples.answer_num,
                "thinking_reward": torch.tensor([thinking_rewards[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "reflect_previous_right_final_incorrect": torch.tensor([previous_right_final_incorrect[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "reflect_previous_incorrect_final_right": torch.tensor([previous_incorrect_final_right[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "difficulty": torch.tensor([difficulty[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "penalty": torch.tensor([penalty[index]], device=device, dtype=torch.float), #   这里我没做packing, 因为micro_rollout_batch_size = 1, 所以这里直接调用index
                "retrieve_score": info_ret_score,
                "group_diversity": info_group_div,
                "state_acoverage": info_acoverage,
                "state_diversity": info_state_div,
                "shallow": samples.shallow_search,
            }

            if self.strategy.args.perf: #    # 每个 sample 中 response 中非检索部分的 token 数（真实回答长度）
                self.perf_stats["actor_value_rm_time"] += actor_value_rm_time
                self.perf_stats["wait_time"] += wait_time

            experience = Experience(
                sequences,  # A sequence of tokens (including prompt and response).
                action_log_probs,  # 当前策略模型的 log_probs
                base_action_log_probs,  # 初始参考策略的 log_probs
                value,  # Critic 给出的 V 值
                None,  # returns 先留空，之后计算
                None,  # advantages 同上
                attention_mask,  # 注意力掩码（非 packed 时用）
                action_mask,  # 哪些 token 是 action（即 response 部分）
                retrieve_mask,  # Which tokens are not in the search area (KL)?
                retrieve_score_segments,
                retrieve_score,
                lets_score,
                samples.group_diversity,
                samples.state_acoverage,
                samples.state_diversity,
                info,  # 额外信息字典（reward、length 等）
                kl,  # 每个 token 的 KL 散度（后续可选用）
            )
            experience = experience.to_device("cpu")
            expeiences.append(experience)

        self.actor.train()  # reset model state # 将 actor 模型重新设置为 train 模式（因为前面切到 eval）
        return expeiences


    def _generate_vllm(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        from vllm import SamplingParams

        # round-robin load balance
        rank = torch.distributed.get_rank() # 获取当前进程的 rank（用于多卡分配）
        world_size = torch.distributed.get_world_size()  # 获取总的进程数（即 GPU 数量）

        # Select LLM engines: assign each rank an engine, or cycle through engines if world_size < engine_count
        # 根据 rank 分配 vLLM 引擎（若引擎数量少于 world_size，就循环使用）
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args # 获取训练参数

        # 设置采样参数，例如温度、top_p、生成 token 数等
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),  # 默认温度 1.0
            top_p=kwargs.get("top_p", 1.0),  # The default top-p is 1.0 (without truncation).
            top_k=kwargs.get("top_k", -1),  # The default value for top-k is -1 (no truncation).
            max_tokens=kwargs.get("max_new_tokens", 1024),  # 最大生成长度
            min_tokens=kwargs.get("min_new_tokens", 1),  # 最小生成长度
            skip_special_tokens=kwargs.get("skip_special_tokens", False),  # 是否跳过特殊符号
            include_stop_str_in_output=True,  # 是否在输出中包含停止字符串
        )

        # Expand prompt list based on the number of samples per prompt
        # 每个 prompt 扩展成多个 sample（重复 n_samples_per_prompt 次）
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        # 对所有 prompts 进行编码（不使用 padding）
        all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]

        # Distribute requests to engines and collect responses to outputs
        all_output_refs = []  # 存放所有远程生成任务的引用
        batch_size = (len(all_prompt_token_ids) + len(llms) - 1) // len(llms) # 每个引擎处理的 prompt 数
        for i, llm in enumerate(llms):
            prompt_token_ids = all_prompt_token_ids[i * batch_size : (i + 1) * batch_size]  # 拿到该引擎负责的一批 prompt
            if args.random_temperature:  # 若启用随机温度策略
                values = [i / 10 for i in range(5, 11)]  # 可选温度值列表（0.5 ~ 1.0）
                sampling_params.temperature = random.choice(values)  # 随机选择一个温度
            print("temperature", sampling_params.temperature)  # 打印当前温度
            if prompt_token_ids:  # 如果这批 prompt 不为空
                all_output_refs.append(  # 向对应的 vLLM 引擎提交生成任务（异步）
                    llm.generate.remote(sampling_params=sampling_params, prompt_token_ids=prompt_token_ids)
                )

        # Retrieve and combine results from all outputs
        all_outputs = sum(ray.get(all_output_refs), [])  # 获取所有引擎的生成结果，并合并成一个列表

        samples_list = []  # 最终返回的样本列表
        for i in range(0, len(all_outputs), args.micro_rollout_batch_size):  # 每次取一个小 batch
            outputs = all_outputs[i : i + self.strategy.args.micro_rollout_batch_size]  # 当前小 batch 的输出结果
            if not self.packing_samples:  # 如果不使用 packed 格式
                # NOTE: concat all outputs to following format:
                #
                # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
                # | token token token token token | token token [EOS] [PAD] |
                # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
                # |<---------- prompt ----------->|<-------- answer ------->|
                max_input_len, max_output_len = 0, 0  # 初始化输入和输出最大长度
                for output in outputs:  # 计算该 batch 中输入和输出的最大长度
                    max_input_len = max(max_input_len, len(output.prompt_token_ids))
                    max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id  # 获取特殊 token id
                sequences = []  # 用于存放拼接后的输入+输出序列
                for output in outputs:
                    input_len = len(output.prompt_token_ids)  # 当前样本的输入长度
                    input_ids = [pad_token_id] * (max_input_len - input_len) + list(output.prompt_token_ids)  # 左填充
                    output_len = len(output.outputs[0].token_ids)  # 当前样本的输出长度
                    output_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - output_len)  # 右填充
                    sequences.append(input_ids + output_ids)  # 拼接后加入序列列表

                sequences = torch.tensor(sequences)  # 转换为 tensor
                # 处理生成的序列，得到 attention mask 和 action mask
                sequences, attention_mask, action_mask = self.actor.process_sequences(
                    sequences, max_input_len, eos_token_id, pad_token_id
                )
                # 将数据移到 GPU
                sequences = sequences.to("cuda")
                attention_mask = attention_mask.to("cuda")
                action_mask = action_mask.to("cuda")
                retrieve_mask = retrieve_mask.to("cuda")  # retrieve_mask 这里是未定义的，会报错，需上层处理
                # 构建 Samples 对象，并添加到结果列表中
                samples_list.append(
                    Samples(
                        sequences=sequences,  # 输入输出拼接序列
                        attention_mask=attention_mask,  # attention 掩码
                        action_mask=action_mask,  # 标注输出部分的位置
                        retrieve_mask=retrieve_mask,  # 标注非检索区域
                        num_actions=action_mask.size(1),  # 动作数量（输出 token 数）
                        packed_seq_lens=None,  # 非 packed 模式，此处为 None
                        response_length=action_mask.float().sum(dim=-1),  # 每个样本的输出 token 数
                        total_length=attention_mask.float().sum(dim=-1),  # 每个样本的总长度
                    )
                )
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []              # 存储拼接后的所有 token 序列
                packed_seq_lens = []        # 每条样本的长度（prompt + response）
                attention_mask = []         # 用于模型区分不同样本
                retrieve_mask=[]            # 检索部分的掩码（0 表示该 token 属于检索部分）
                num_actions = []            # response 的 token 数（最少为1）                
                for i, output in enumerate(outputs):  # 遍历每个 output 样本
                    input_len = len(output.prompt_token_ids)         # prompt 的长度
                    output_len = len(output.outputs[0].token_ids)    # response 的长度
                    packed_seq_lens.append(input_len + output_len)   # 添加 packed 长度
                    sequences.extend(output.prompt_token_ids + list(output.outputs[0].token_ids))  # 拼接 prompt 和 response
                    attention_mask.extend([i + 1] * (input_len + output_len))  # 相同位置填入样本编号

                    # 构建 retrieve_mask，默认 response 的所有位置都是 1
                    # response_seq = list(output.outputs[0].token_ids)
                    # retrieve_mask_now = [1]*len(response_seq)
                    
                    # 找到 retrieve 的标记位置（开始和结束）
                    # 构建 retrieve_mask，默认 response 的所有位置都是 1
                    response_seq = list(output.outputs[0].token_ids)
                    retrieve_mask_now = [1]*len(response_seq)
                    
                    # 找到 retrieve 的标记位置（开始和结束）
                    start_indices = [g for g, x in enumerate(response_seq) if x == 151657]
                    end_indices = [g for g, x in enumerate(response_seq) if x == 151658]
                    assert len(start_indices) == len(end_indices), "KL: start_indices 和 end_indices 长度不一致"
                    
                    # 将标记区间位置设为 0（属于 retrieve 的部分）
                    for start, end in zip(start_indices, end_indices):
                        for h in range(start, end + 1):
                            retrieve_mask_now[h] = 0

                    retrieve_mask.extend(retrieve_mask_now)  # 添加到总掩码中
                    num_actions.append(max(1, output_len))   # 记录 response 长度
                # 将拼接的内容转为 tensor 并存入 Samples 对象中
                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None
                retrieve_mask = torch.tensor(retrieve_mask, device="cuda").unsqueeze(0)

                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                # print("sequences_size:",sequences.size())
                # print("attention_mask_size:",attention_mask.size())
                # print("vllm-retrieve_mask_size:",retrieve_mask.size())
                # print("response_length_size:",response_length.size())
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)
                # 将处理好的样本加入结果列表

                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        retrieve_mask = retrieve_mask,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                    )
                )
        return samples_list


    def _generate_vllm_with_retrieve(self, all_prompts: List[str], **kwargs) -> List[Samples]:
        url_wiki = "http://0.0.0.0:5003/queries"  # Retrieve server address
        from vllm import SamplingParams
        def get_infohash(passage):
            return " ".join(passage.split(" ")[:10]) # TODO: Fixed to 10 can make it variable, but is an internal representation choice

        # Get the current process rank and total number of processes.
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        print('Print information related to generate_vllm_with_retrieve', rank, world_size) #Sure enough, there is a process for each GPU that calls this method.
        # The vllm instance assigned to the current rank
        if len(self.vllm_engines) <= world_size:
            llms = [self.vllm_engines[rank % len(self.vllm_engines)]]
        else:
            llms = self.vllm_engines[rank::world_size]

        args = self.strategy.args
        adaptive = args.adaptive
        potential = (args.reshape_method in ['potential'])
        print('potential???', potential, args.approx_coverage)

        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=kwargs.get("temperature", 1.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", -1),
            max_tokens=kwargs.get("max_new_tokens", 1024),
            min_tokens=kwargs.get("min_new_tokens", 1),
            skip_special_tokens=kwargs.get("skip_special_tokens", False),
            include_stop_str_in_output=True,
            logprobs=2,
            n=2,
        )
      #  te = kwargs.get("temperature", 1.0)
       # print('打印模型运行的temperature',)
        # 将 prompt 复制多份，每个 prompt 对应多个样本
        all_prompts = sum([[prompt] * args.n_samples_per_prompt for prompt in all_prompts], [])
        # Each prompt is broken down into an idx + prompt dictionary.
        prompts_w_idx_dict = []
        for idx_w_prompt in all_prompts:
            idx, prompt = idx_w_prompt.split("<|idx_prompt_split|>")
            prompt, gold_answer = prompt.split("<pro_answer>")
            prompts_w_idx_dict.append({"idx": idx, "prompt": prompt, "gold_answer": gold_answer})
        # Add search functionality starting here.
        # all_prompt_token_ids = self.tokenize_fn(all_prompts, self.prompt_max_len, padding=False)["input_ids"]
        stop_tokens = ["</search>"] # stop sign
        batch_size = (len(prompts_w_idx_dict) + len(llms) - 1) // len(llms)
        print("llms-len:",len(llms)) #There is only one llms
        all_outputs = []
        for i, llm in enumerate(llms): # Each LLM performs several group inferences (each group performs multiple rollouts).
            # 为每个 LLM 分配一部分 prompt
            print('The i-th llm:',i,rank, len(prompts_w_idx_dict), i * batch_size, (i + 1) * batch_size)
            idx_w_prompt_part = prompts_w_idx_dict[i * batch_size : (i + 1) * batch_size]
            print(f'Print the prompt for the {rank}th LLM.:',idx_w_prompt_part[0])

            # print("idx_w_prompt_part:",idx_w_prompt_part)
            data_keys = ["prompt" , "idx"]
            ds = Dataset.from_dict({key: [d[key] for d in idx_w_prompt_part] for key in data_keys})
            # print("ds:",ds)

            finished_all_list=[]  # All completed samples in the current section
            continued_answer = copy.deepcopy(idx_w_prompt_part) # Used to progressively generate and add retrieved documents.

            for t in range(11): # Each prompt undergoes 11 inferences to arrive at a complete solution, ensuring that all queries required for retrieval appear. The 11 inferences refer to the maximum number of times this prompt is used, as the retrieval inferences are generated multiple times.
                print(f'\nIteration {t} ============================')
                finished_texts = []
                continued_texts = []
                sampling_params = SamplingParams(temperature=1, top_p=0.95, max_tokens=512, stop=stop_tokens) #These are the parameters used by 26_8

                outputs_ray = llm.generate.remote(ds['prompt'], sampling_params)
                outputs = ray.get(outputs_ray)
                query_list=[]
                strategy_list=[]
                incorrect_strategy_list=[]
                ori_strategy = []
                for q, output in enumerate(outputs):
                    prompt = output.prompt
                    idx = continued_answer[q]["idx"]
                    gold_answer = continued_answer[q]["gold_answer"]
                    stop_reason = output.outputs[0].stop_reason
                    generated_text = output.outputs[0].text
                    # Record the prompt token during the initial reasoning.
                    if "prompt_ids" not in continued_answer[q]:
                        input_token_ids = list(output.prompt_token_ids)
                    else:
                        input_token_ids = continued_answer[q]["prompt_ids"]

                    retrieve_num_count = continued_answer[q].get("retrieve_num_count",0)

                    all_token_ids = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
                    output_token_ids = all_token_ids[len(input_token_ids):]

                    if 'retrieval_mask_tokens' not in continued_answer[q]:
                        retrieval_mask_tokens = [[1] * len(output.outputs[0].token_ids)] #The tricky part is that the token ID includes `</search>`, but the text does not.
                    else:
                        retrieval_mask_tokens = continued_answer[q]['retrieval_mask_tokens'].copy()
                        linshi = []
                        for i in range(len(output.outputs[0].token_ids)):
                            linshi.append(1)
                        retrieval_mask_tokens.append(linshi)
                    
                    if 'ret_doc_segment_score' not in continued_answer[q]:
                        ret_doc_segment_score = []
                    else:
                        ret_doc_segment_score = continued_answer[q]['ret_doc_segment_score'].copy()
                    
                    if 'ret_doc_score' not in continued_answer[q]:
                        ret_doc_score = []
                    else:
                        ret_doc_score = continued_answer[q]['ret_doc_score'].copy()
                                        
                    infohash = [] if 'infohash' not in continued_answer[q] else continued_answer[q]['infohash'].copy()
                    infostrategy = [] if 'infostrategy' not in continued_answer[q] else continued_answer[q]['infostrategy'].copy()

                    # will stop at this point if doing Lenrag with all rewards.
                    if t == 8:
                        print('Too many searches :(:(:(:(:(:(:(:(:(:(:(:(:(:(:(')
                        #Too many searches have been performed; the search has been stopped immediately, indicating the search is incomplete.
                        ret_doc_segment_score.append(retrieval_mask_tokens[-1]) 
                        ret_doc_score.append(0)
                        original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "response_ids":output_token_ids,
                            "retrieve_num_count" : retrieve_num_count,
                          #  "actual_response_ids": actual_response_ids,
                            "retrieval_mask_tokens": [item for sublist in retrieval_mask_tokens for item in sublist],
                            "ret_doc_segment_score": ret_doc_segment_score,
                            "ret_doc_score": ret_doc_score,
                            "infohash": infohash,
                            "infostrategy": infostrategy,
                            "gold_answer": gold_answer
                                }
                        
                        assert len([item for sublist in ret_doc_segment_score for item in sublist]) == len(original_data['retrieval_mask_tokens']) \
                                and len(ret_doc_score)==retrieve_num_count + 1, \
                                f"retrieval_mask_tokens:{retrieval_mask_tokens}\nret_doc_segment_score:{ret_doc_segment_score}"

                        if original_data["response_ids"] == None:
                            print(f"Here 1,Why this response_ids is None???,{original_data}")
                            time.sleep(10)
                        finished_texts.append(original_data)
                        continue

                    if "<search>" in generated_text and stop_reason=="</search>":
                        #print('Print query', generated_text)
                        strategy = ''
                        if adaptive:
                            s_pattern = re.compile(r"<strategy>(.*?)</strategy>",re.DOTALL)
                            strate_matches = s_pattern.finditer(generated_text)
                            for e, match in enumerate(strate_matches):
                                print('STRATEGY!!!!!!', match.group(1))
                                strategy = match.group(1).strip().replace('\n','')
                        query = generated_text.split("<search>")[-1].split("</search>")[0]
                        query = query.replace('"',"").strip()
                        query = " ".join(query.split())

                        if query: #Start search
                            #print('generated text', generated_text.strip())
                            if 'qwen' in args.pretrain.lower():
                                m = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(" "+generated_text.strip()))
                                m.extend([690,1836,1339]) #TODO: Qwen2.5 and Qwen3
                            else:
                                assert False
                            retrieval_mask_tokens[-1] = [1] * len(m)
                            all_token_ids = list(output.prompt_token_ids) + m
                            output_token_ids = all_token_ids[len(input_token_ids):] #Because we actually modified the output here, we need to make some changes... 

                            if adaptive: ori_strategy.append(strategy)
                            if (adaptive and strategy in ['Shallow', 'Deep']): # the retriever only understands these modes.
                                strategy_list.append(strategy)
                                incorrect_strategy_list.append(False)
                            elif adaptive:
                                strategy_list.append('Shallow')
                                incorrect_strategy_list.append(True)
                            else:
                                incorrect_strategy_list.append(False)
                                ori_strategy.append('')
                            query_list.append(query)
                            retrieve_num_count += 1
                            original_data = {
                                "idx":idx,
                                "prompt": prompt + " " + generated_text.strip(), #Here, r1-search uses generate_text.strip(). Why is this used...?
                                "prompt_ids":input_token_ids,
                                "response_ids":output_token_ids,
                                "retrieve_num_count" : retrieve_num_count,
                               # "actual_response_ids": actual_response_ids,
                                "retrieval_mask_tokens": retrieval_mask_tokens,
                                "ret_doc_segment_score": ret_doc_segment_score,
                                "ret_doc_score": ret_doc_score,
                                "infohash": infohash,
                                "infostrategy": infostrategy,
                                "gold_answer": gold_answer
                                }
                            if original_data["response_ids"] == None:
                                print(f"Here 2,Why this response_ids is None???,{original_data}")
                                time.sleep(10)
                            continued_texts.append(original_data)
                        else: #This query didn't follow the correct format, so the process stopped abruptly. Optimization may be needed later.
                            ret_doc_segment_score.append(retrieval_mask_tokens[-1]) 
                            ret_doc_score.append(0)
                            original_data = {
                            "idx":idx,
                            "prompt_ids":input_token_ids,
                            "response_ids":output_token_ids,
                            "retrieve_num_count" : retrieve_num_count,
                        #    "actual_response_ids": actual_response_ids,
                            "retrieval_mask_tokens": [item for sublist in retrieval_mask_tokens for item in sublist],
                            "ret_doc_segment_score": ret_doc_segment_score,
                            "ret_doc_score": ret_doc_score,
                            "infohash": infohash,
                            "infostrategy": infostrategy,
                            "gold_answer": gold_answer
                                }
                            assert len([item for sublist in ret_doc_segment_score for item in sublist]) == len(original_data['retrieval_mask_tokens']) \
                                    and len(ret_doc_score)==retrieve_num_count + 1, \
                                    f"retrieval_mask_tokens:{retrieval_mask_tokens}\ret_doc_segment_score:{ret_doc_segment_score}"
                            if original_data["response_ids"] == None:
                                print(f"Here 3,Why this response_ids is None???,{original_data}")
                                time.sleep(10)
                            finished_texts.append(original_data)
                            # print("(2)ori-dict:",original_data)
                    else: #End of generation                   # No query was generated, which means the process has ended.
                        ret_doc_segment_score.append(retrieval_mask_tokens[-1])
                        ret_doc_score.append(0)
                        original_data = {
                        "idx":idx,
                        "prompt_ids":input_token_ids,
                        "response_ids":output_token_ids,
                        "retrieve_num_count" : retrieve_num_count,
                       # "actual_response_ids": actual_response_ids,
                        "retrieval_mask_tokens": [item for sublist in retrieval_mask_tokens for item in sublist],
                        "ret_doc_segment_score": ret_doc_segment_score,
                        "ret_doc_score": ret_doc_score,
                        "infohash": infohash,
                        "infostrategy": infostrategy,
                        "gold_answer": gold_answer
                        }
                        finished_texts.append(original_data)
                        assert len([item for sublist in ret_doc_segment_score for item in sublist]) == len(original_data['retrieval_mask_tokens']) \
                                and len(ret_doc_score)==retrieve_num_count + 1, \
                                f"retrieval_mask_tokens:{retrieval_mask_tokens}\ret_doc_segment_score:{ret_doc_segment_score}"
                        if original_data["response_ids"] == None:
                            print(f"Here 4,Why this response_ids is None???,{original_data}")
                            time.sleep(10)
                        # print("(3)ori-dict:",original_data)

                assert len(query_list) == len(continued_texts), "Error in len of query_list and continued_texts"
                if adaptive:
                    assert len(query_list) == len(strategy_list), "Error in len of query_list and strategy_list"
                #print("query-list:",query_list)
                if len(query_list)!=0:
                    topk = 5

                    response = requests.post(url_wiki, json={"queries": query_list, "k": topk, "rerank": 0, "rescore": 1, "strategy": strategy_list if adaptive else None}) # query_list实际上是query的列表。其它的似乎无法传入列表，要改成循环
                    
                    if response.status_code == 200:
                        print('Query complete, query successful.')
                        result = response.json()
                        answers = result["answers"]
                        distances = result["distances"]
                        if adaptive:
                            deep_penalty = result["deep_penalty"]
                        assert len(query_list) == len(answers)
                        for k in range(len(answers)):
                            print('############ QTEMP', query_list[k], ori_strategy[k])
                            
                            retrieve_docs = answers[k] # answers[0] contains the topk responses. Here k refers to: the kth query, the kth answer, corresponding one to one.
                            retrieve_docs_score = [float(v) for v in distances[k]]
                            continued_text_now = copy.deepcopy(continued_texts[k]) #Why is this the index k
                            if len(retrieve_docs)>0 and not incorrect_strategy_list[k]:
                                doc_content_list = []
                                highest_ret_score = 0
                                infohash_now = []
                                for j in range(len(retrieve_docs)):
                                    doc_now = re.sub(r'^\d+\s+', '', retrieve_docs[j])
                                    doc_content_list.append(f"({j+1}){doc_now}\n")
                                    # search highst score cause might have been re-scored but not re-ordered
                                    if retrieve_docs_score[j] > highest_ret_score:
                                        highest_ret_score = retrieve_docs_score[j]
                                    infohash_now.append(get_infohash(retrieve_docs[j])) #take the first 50 'words' does not matter as is just as a key/hash, what's important is to take the title\ndoc initial string info to differentiate passages
                                doc_content = ''.join(doc_content_list)
                            elif incorrect_strategy_list[k]:
                                doc_content = "No results. The type of search strategy is incorrect, should use <strategy>Shallow</strategy> or <strategy>Deep</strategy>."
                                highest_ret_score = 0.1 #a small reward because the question still may be worth, was just the search strategy
                                infohash_now = [] # should be out: the infohashes should stay so its further penalised if not being diverse
                                print(f'*** Incorrect strategy: {ori_strategy[k]}')
                            else:
                                doc_content = "None"
                                highest_ret_score = 0

                            # if adding None: instead of <information> tags add sthg like: "elaborate your knowledge about this question: "
                            
                            # store highest retrieval score
                            if adaptive and deep_penalty[k]: #the penalty can be computed only when the search strategy is correct and Deep
                                print('Asked for Deep but not needed!')
                                if args.deep_penalty == -1:
                                    doc_content = "No results. Either the sub-query is easy (then re-issue the same query using <strategy>Shallow</strategy>) or the sub-query is too difficult or needs reformulation."
                                    highest_ret_score = 0.1 #a small reward because the question still may be worth, was just the search strategy
                                    infohash_now = [] # out: the infohashes should stay so its further penalised if not being diverse
                                else:
                                    highest_ret_score = highest_ret_score * args.deep_penalty #reduce the score
                                    infohash_now = []
                            elif adaptive and 'Deep' in ori_strategy[k]: #used deep and there was no penalty, but still reduce due to cost of the tool
                                    highest_ret_score = highest_ret_score * 0.9

                            if adaptive:
                                continued_text_now["infostrategy"].append(ori_strategy[k])    
                            if 'gola' in args.intrinsic and 'nite' in args.intrinsic:
                                # answer presence reward is 0.15  as used in the re-implementation by (Xie et al., 2026) see Section E.4 
                                # https://openreview.net/pdf?id=eBMOr6a84z
                                continued_text_now["ret_doc_score"].append( (0.15 \
                                                            if retrieval_precision_([continued_text_now["gold_answer"]], retrieve_docs) else 0))
                            else:
                                continued_text_now["ret_doc_score"].append(highest_ret_score)
                            continued_text_now["infohash"].append("#".join(infohash_now))
                            continued_text_now["prompt"] = continued_text_now["prompt"] + " </search>\n\n"+ "<information>\n" +  doc_content + "</information>\n\n"
                            continued_text_now["ret_doc_segment_score"].append([ v for v in continued_text_now["retrieval_mask_tokens"][-1] ])

                            ### retrieval mask Settings
                            m = []
                            for li in self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize("<information>\n" +  doc_content + "</information>\n\n")):
                                m.append(0)
                            continued_text_now["retrieval_mask_tokens"].append(m)
                            continued_text_now["ret_doc_segment_score"].append(m)
                            continued_texts[k] = continued_text_now
                            ### retrieval mask Settings
                    else:
                        raise Exception("Error in response: the status code is not 200!")

                finished_all_list.extend(finished_texts) # Add the samples completed in this round to the list.
                # If all operations have been completed, exit the loop.
                if len(continued_texts)==0:
                    if len(finished_all_list) != len(idx_w_prompt_part):
                        #time.sleep(20)
                        print("finished_all_list:",finished_all_list)
                        print("idx_w_prompt_part:",idx_w_prompt_part)
                        #time.sleep(20)
                    assert len(finished_all_list) == len(idx_w_prompt_part) , "Error in len of finished_all_list and idx_w_prompt_part"
                    all_outputs.append(finished_all_list)
                    break
                else:  # Otherwise, continue to the next round of generation.
                    data_keys_again = continued_texts[0].keys()
                    if 'ret_doc_segment_score' not in data_keys_again:
                        print('PEPE '*20)
                        print('TBC ', len(continued_texts))
                        print(continued_texts[0])
                        assert False
                    ds = Dataset.from_dict({key: [d[key] for d in continued_texts] for key in data_keys_again})
                    continued_answer = copy.deepcopy(continued_texts)

        # Integrate all outputs and sort them
        all_outputs_cat = [item for sublist in all_outputs for item in sublist]  # Expand all sublists into a single complete list

        if len(all_outputs_cat) != len(prompts_w_idx_dict):   # Check if the number of samples is consistent after unfolding.
            print("[all_outputs_cat,prompts_w_idx_dict]:",[all_outputs_cat,prompts_w_idx_dict])
            #time.sleep(20)
        assert len(all_outputs_cat) == len(prompts_w_idx_dict), "Error in len of all_outputs and prompts_w_idx_dict"
        all_outputs_sorted = sorted(all_outputs_cat, key=lambda x: x['idx'])  # Sort by idx to ensure consistent order.

        # group rollout diversity, compute for all methods to monitor diversity, but only some methods use it as reward
        #if args.diversity_level == 'step': # do always there is a single way implemented
        for idx in range(len(all_outputs_sorted) // args.n_samples_per_prompt):
            group = all_outputs_sorted[idx* args.args.n_samples_per_prompt :(idx+1) * args.args.n_samples_per_prompt]
            retrieval_counts = [output["retrieve_num_count"] for output in group]
            #print('retrieval_counts', retrieval_counts)
            div_pen = [(([0] * output["retrieve_num_count"]) if output["retrieve_num_count"]>0 else [0]) for output in group]
            #print('div_pen', div_pen)
            rets = [(e, s, infohash) for e, output in enumerate(group) for s,infohash in enumerate(output["infohash"])]
            rets = sorted(rets,  key=lambda x: x[2])
            #print(rets)
            if len(rets) > 0:
                cnts = []
                prev = rets[0][2]
                for e, s, infohash in rets:
                    if prev != infohash:
                        for u,v in cnts:
                            if prev == '':
                                div_pen[u][v] = args.n_samples_per_prompt #if was an incorrect action, diversity penalty is not defined here.
                            else:
                                div_pen[u][v] += len(cnts)
                        cnts = []
                    cnts.append((e,s))
                    prev = infohash
                if len(cnts) > 0:
                    for u,v in cnts:
                        div_pen[u][v] += len(cnts)     
            for s in range(args.args.n_samples_per_prompt):
                group[s]['group_diversity'] = ([1] if potential else []) + \
                                        [x/args.args.n_samples_per_prompt for x in div_pen[s]] + \
                                        ([0] if retrieval_counts[s] >0 else []) # if for some reason there is repetition, then clip to max count of 5               

        # add state coverage and diversity summary scores, also complete info for initial state if potential computations
        for output in all_outputs_sorted:
            # profit of this loop to initialised this
            output["lets_score"] = []

            if args.approx_coverage:
                silver = self.silver_passages_per_idx[output['idx']]["silver_infohash"]
                state_acoverage = [0] if potential else []
            else:
                state_acoverage = []
            state_diversity = [1] if potential else []
            info_hashes = []
            prev_info_hash_sets = []
            for info_hash in output["infohash"]:
                current_info_hash_set = info_hash.split('#')
                info_hashes.extend(current_info_hash_set)
                if args.approx_coverage:
                    state_acoverage.append(len(set(silver) & set(info_hashes)))
                if 'rlredun' in args.intrinsic:
                    if prev_info_hash_sets == []:
                        state_diversity.append(1)
                    else:
                        dists = []
                        for prev_h_set in prev_info_hash_sets:
                            dists.append((len(set(prev_h_set) & set(current_info_hash_set)))/(len(prev_h_set) + len(current_info_hash_set)))
                        state_diversity.append(1 - max(dists))
                else:
                    state_diversity.append(len(set(info_hashes))/len(info_hashes))
            
            output["state_acoverage"] = state_acoverage + [0] #add for size compat with ret_score list
            output["state_diversity"] = state_diversity + [0] #add for size compat with ret_score list
            # update ret score with the zero state ones
            output["ret_doc_score"] = ([self.silver_passages_per_idx[output['idx']]["ret_doc_score"]] if potential else []) + output["ret_doc_score"]

            assert len(output["state_acoverage"]) == len(output["state_diversity"]) \
                and len(output["state_acoverage"]) == len(output["ret_doc_score"]), \
                f'state_acoverage:{output["state_acoverage"]} state_diversity:{output["state_diversity"]}' + \
                         f'ret_doc_score:{output["ret_doc_score"]} group_diversity:{output["group_diversity"]}'
            #if args.diversity_level == 'step': # always computed in the same way, so no need to ask
            assert len(output["state_acoverage"]) == len(output["group_diversity"])

        if 'glmatch' in args.intrinsic:
            # TODO: redefine the field output["ret_doc_score"] with LeTS uncer-over weight alingments, so will not to pass additional field everywhere, actually should be nicer....
            for idx in range(len(all_outputs_sorted) // args.args.n_samples_per_prompt):
                group = all_outputs_sorted[idx* args.args.n_samples_per_prompt:(idx+1) * args.args.n_samples_per_prompt]
                overs = []
                unders = []
                for e, output in enumerate(group):
                    # is under or over performing (EM=1)
                    solution = self.tokenizer.decode(output["response_ids"])
                    solution = solution.split("<answer>")[-1].split("</answer>")[0].strip()
                    em = exact_match_score(solution, output["gold_answer"])
                    if em == 1:
                        overs.append(e)
                    else:
                        unders.append(e)
                 
                for e, output in enumerate(group):
                    if e in overs:
                        if potential:
                            #state diversity is part of potentials
                            output["lets_score"] = [0] * len(output["state_diversity"])
                        else:
                            output["lets_score"] = output["state_diversity"]
                        print('Outperforming ==> setting state diversity', output["ret_doc_score"])
                    else:
                        # is in under
                        current_infostrategy = output["infostrategy"]
                        current_info_hash_set = [infohash.split('#') for infohash in output["infohash"]]
                        print(f'\n***len of current infohas_set {len(current_info_hash_set)}')
                        max_KuhnMunkres_computed = False
                        max_KuhnMunkres = None
                        max_total = 0
                        if current_info_hash_set:
                            for o in overs:
                                o_infostrategy = group[o]["infostrategy"]
                                o_info_hash_set = [infohash.split('#') for infohash in group[o]["infohash"]]
                                print(f'len of outperforming infohas_set {len(o_info_hash_set)}')
                                if not o_info_hash_set:
                                    continue
                                G = []
                                for ci, (c_hash_set, c_strat) in enumerate(zip(current_info_hash_set, current_infostrategy)):
                                    ci_row = []
                                    for co, (o_hash_set, o_strat) in enumerate(zip(o_info_hash_set, o_infostrategy)):
                                        if potential and c_strat != o_strat:
                                            #when used w/ tools only compute when the tool is the same
                                            ci_row.append(0.0)
                                        else:
                                            ci_row.append((len(set(c_hash_set) & set(o_hash_set))) / (len(c_hash_set) + len(o_hash_set)))
                                    if len(current_info_hash_set) > len(o_info_hash_set):
                                        ci_row.extend([0.0] * (len(current_info_hash_set) - len(o_info_hash_set)))
                                    G.append(ci_row)                                        
                                print('GGGGGGGGGGG',G)
                                cost = np.array(G)
                                if len(G) >0:
                                    row_ind, col_ind = linear_sum_assignment(cost, True)
                                    total = cost[row_ind, col_ind].sum()
                                    if total > max_total:
                                        max_total = total
                                        max_KuhnMunkres = cost[row_ind, col_ind]
                                        max_KuhnMunkres_computed = True
                                        print('save max munges', max_KuhnMunkres)
                                else:
                                    print('No munges')
                                #time.sleep(10)

                        if max_KuhnMunkres_computed:
                            munges_score = ([0] if potential else []) + list(max_KuhnMunkres) + [0]
                            assert len(output["ret_doc_score"]) == len(munges_score), f'{output["ret_doc_score"]} -- {munges_score}'
                            output["lets_score"] = munges_score
                            print('ANY munge scores?', output["lets_score"])
                        else:
                            output["lets_score"] = [0] * len(output["ret_doc_score"]) # slot for reward is already included

        samples_list = []
        for i in range(0, len(all_outputs_sorted), args.micro_rollout_batch_size): # Divide samples by micro batch
            outputs = all_outputs_sorted[i : i + self.strategy.args.micro_rollout_batch_size]
            if not self.packing_samples:  # The logic of not performing packing (can be extended)
                pass
            else:
                # NOTE: concat all outputs to following format:
                #
                # | token token token | token token [EOS] | token token token token token | token token [EOS] | token token | token token token [EOS] |
                # |<---  prompt ----->|<---- answer ----->|<---------- prompt ----------->|<----- answer ---->|<- prompt -->|<-------- answer ------->|
                # NOTE: By concatenating all prompts and responses into a single long sequence, the model can efficiently process multiple samples at once.
                pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
                sequences = []  # 拼接后的 token 序列
                packed_seq_lens = []  # Total length of each sample
                attention_mask = []  # 标记哪些 token 属于哪个样本
                retrieve_mask = []  # Mark which tokens are search content (set to 0).
                retrieve_score_segments = []  # Retrieval scores for tokens which are search content (set to 0). Highest document score for each retrieve step.
                retrieve_score = []
                lets_score = []
                num_actions = []  # The number of response tokens per sample (minimum 1)）
                exit_flags = []  # The correction flag for each sample records that if this sample is corrected, then it needs to receive a higher reward.
                answer_nums = []
                retrieve_num = []  # Number of searches per sample
                group_diversity = [] # how diverse is the rollout within the group
                state_acoverage = []
                state_diversity = []
                shallow_search = []
                pure_response_length_lis = []  # The number of tokens in the response that are not part of the search results.
                
                for i, output in enumerate(outputs):
                   # try:
                        input_len = len(output["prompt_ids"])  # prompt length
                        #output_len = len(output["response_ids"])  # response length
                        exit_flag = 0 
                        answer_nums.append(0)
                        output_len = len(output["response_ids"])  # response length
                        packed_seq_lens.append(input_len + output_len)  # Total length record

                        sequences.extend(output["prompt_ids"] + output["response_ids"])  # splicing input and output
                        attention_mask.extend([i + 1] * (input_len + output_len))  # Assign a batch id to each token

                        response_seq = output["response_ids"]
                        retrieve_mask_now = [1] * len(response_seq)  # Initially all values ​​are 1, indicating non-search content.

                        # TODO: the following is not really nice using the tokens' IDs
                        # Determine the type of tokenizer used (LLaMA vs Qwen) to decide which token to use.
                        if output["prompt_ids"][0] == 128000:  # llama model
                            start_tokens = [27, 91, 7413, 3659, 77027, 91, 397]# <|begin_of_documents|> Corresponding token
                            end_tokens = [408, 3659, 77027, 91, 1363]# <|end_of_documents|> Corresponding token
                        else:  # qwen model
                            # start_tokens = [27, 91, 7265, 3575, 75927, 91, 397]  # <|begin_of_documents|> 对应的 token
                            # end_tokens = [408, 3575, 75927, 91, 1339]  # <|end_of_documents|>
                            start_tokens = [27, 25069, 397] # <information>\n
                            end_tokens = [522, 25069, 1339] # </information>\n

                        is_in_masking = False  # Is it currently being marked?
                        mask_start_idx = -1
                        start_count = 0
                        end_count = 0
                        # 1. Find all intervals of <information>...</information>
                        # info_spans = []
                        # info_start = None
                        for m in range(len(response_seq)):
                            # Match the start token sequence
                            if (not is_in_masking and m + len(start_tokens) <= len(response_seq) and
                                response_seq[m:m + len(start_tokens)] == start_tokens):
                                start_count += 1
                                is_in_masking = True
                                mask_start_idx = m
                                for n in range(mask_start_idx, mask_start_idx + len(start_tokens)):
                                    retrieve_mask_now[n] = 0  # Set all starting tokens to 0.
                                # info_start = m
                            if is_in_masking:
                                retrieve_mask_now[m] = 0  # The token in the search segment is set to 0.
                                # End of matching token sequence
                                if (m + len(end_tokens) <= len(response_seq) and
                                    response_seq[m:m + len(end_tokens)] == end_tokens):
                                    end_count += 1
                                    is_in_masking = False
                                    for u in range(m, m + len(end_tokens)):
                                        retrieve_mask_now[u] = 0  # Set all end tokens to 0
                                    # if info_start is not None:
                                    #     info_spans.append((info_start, m + len(end_tokens) - 1))
                                    #     info_start = None
                                    mask_start_idx = -1
                        if start_count == end_count == output["retrieve_num_count"]:
                            # if(mask != retrieve_mask_now):
                            #     print('Something went wrong')
                            if( output["retrieval_mask_tokens"] !=retrieve_mask_now ):
                                print("There's a problem with the newly set retrieval mask.")
                                #print('打印模型的response_seq:', self.tokenizer.decode(response_seq))
                                #print('打印模型的response_seq id:', response_seq)
                                #kill
                            pass
                        else:
                            print(f"Important Bug!! Model genearte the retrieve symbols!!, The num:{[start_count,end_count,output['retrieve_num_count']]},The detailed content:{output}")
                            #time.sleep(10)
                            # assert start_count == end_count == output["retrieve_num_count"], "Model genearte the retrieve symbols"

                        retrieve_mask.extend(output["retrieval_mask_tokens"])  # Add mask to the current sample
                        retrieve_score_segments.extend(output["ret_doc_segment_score"])  # Add mask to the current sample
                        retrieve_score.extend(output["ret_doc_score"])
                        lets_score.extend(output["lets_score"])
                        num_actions.append(max(1, output_len))  # Add response length
                        retrieve_num.append(output["retrieve_num_count"])  # Add search times
                        group_diversity.append(output["group_diversity"])
                        state_acoverage.append(output["state_acoverage"])
                        state_diversity.append(output["state_diversity"])
                        if len(output["infostrategy"])>0:
                            shallow_search.append(sum([1 for x in output["infostrategy"] if x=='Shallow'])/len(output["infostrategy"]))
                        else:
                            shallow_search.append(0)
                        exit_flags.append(exit_flag)
                        pure_response_length_lis.append(sum(output["retrieval_mask_tokens"]))  # Count of non-retrieval tokens
                        #print('打印模型的retrieval mask:', retrieve_mask_now)
                        ## print('打印模型的actual_response_ids:', actual_response_ids)
                        ## print('打印模型的mask:', mask)
                        #print('打印模型的output["retrieval_mask_tokens"]:', output["retrieval_mask_tokens"])
                        #print('打印模型的response_seq:', self.tokenizer.decode(response_seq))
                        #print('打印模型的response_seq id:', response_seq)
                        assert len(output["retrieval_mask_tokens"]) == len(response_seq), f"Error in len of retrieval_mask_tokens ({len(output['retrieval_mask_tokens'])})and response_seq ({len(response_seq)})"
                    # except Exception as e:
                    #     print(f"Error occur:{e}, the data: {output},{output['prompt_ids']},{output['response_ids']}")


                # Convert to Tensor form
                #print(self.tokenizer.decode(sequences))
                #query=self.tokenizer.decode(sequences)
                #print('ANY ANSWER', query.split("<answer>")[-1].split("</answer>")[0].strip())
                #time.sleep(20)
                #remove_prefix = query.split("<search>your sub-question</search>.")[1].strip()
                #question = remove_prefix.split("<start_of_turn>model")[0].strip().split("<end_of_turn>")[0]
                #solution = remove_prefix.split("<start_of_turn>model")[1].split("<end_of_turn>")[0].strip()
                #print('question',question)
                #print('solution',solution )
                #kill
                sequences = torch.tensor(sequences, device="cuda").unsqueeze(0)
                attention_mask = torch.tensor(attention_mask, device="cuda").unsqueeze(0)
                action_mask = None  # The action_mask is not currently in use, presumably because the retrieve_mask has been used as a replacement for the action mask.
                retrieve_mask = torch.tensor(retrieve_mask, device="cuda").unsqueeze(0)
                retrieve_score_segments = [torch.tensor(x, device="cuda", dtype=torch.float) for x in retrieve_score_segments]
                retrieve_score = torch.tensor(retrieve_score, device="cuda", dtype=torch.float).unsqueeze(0)
                lets_score = torch.tensor(lets_score, device="cuda", dtype=torch.float).unsqueeze(0)

                response_length = torch.tensor(num_actions, device="cuda", dtype=torch.float)
                total_length = torch.tensor(packed_seq_lens, device="cuda", dtype=torch.float)

                retrieve_nums = torch.tensor(retrieve_num, device="cuda", dtype=torch.float)
                group_diversities = torch.tensor(group_diversity, device="cuda", dtype=torch.float)
                state_acoverages  = torch.tensor(state_acoverage, device="cuda", dtype=torch.float)
                state_diversities = torch.tensor(state_diversity, device="cuda", dtype=torch.float)
                shallow_searches = torch.tensor(shallow_search, device="cuda", dtype=torch.float)

                pure_response_length = torch.tensor(pure_response_length_lis, device="cuda", dtype=torch.float)
                exit_flags = torch.tensor(exit_flags, device="cuda", dtype=torch.float)
                answer_nums = torch.tensor(answer_nums, device="cuda", dtype=torch.float)
                # Create Samples objects and add them to the list.
                #if i==0: 
                #    print(f"Sequences ({packed_seq_lens})\n")
                #    #time.sleep(10)
                samples_list.append(
                    Samples(
                        sequences=sequences,
                        attention_mask=attention_mask,
                        action_mask=None,
                        retrieve_mask=retrieve_mask,
                        retrieve_score_segments=retrieve_score_segments,
                        retrieve_score=retrieve_score,
                        lets_score=lets_score,
                        num_actions=num_actions,
                        packed_seq_lens=packed_seq_lens,
                        response_length=response_length,
                        total_length=total_length,
                        retrieve_num=retrieve_nums,
                        group_diversity=group_diversities,
                        state_acoverage=state_acoverages,
                        state_diversity=state_diversities,
                        shallow_search=shallow_searches,
                        pure_response_length=pure_response_length,
                        exit_flag=exit_flags,
                        answer_num=answer_nums
                    )
                )
        return samples_list

    def flush(self):
        "Ensure all experience has been send to critic"
        if self.critic is not None:
            ray.get(self._ref)
            self._ref = None
