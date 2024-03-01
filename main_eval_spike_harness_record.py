from mistral.cache import RotatingBufferCache
import logging
import torch
import fire
from typing import List
from pathlib import Path
import os
# from mistral.model1 import Transformer
# from mistral.model import Transformer 
from mistral.model_record import Transformer 

from mistral.tokenizer import Tokenizer

import lm_eval
from lm_eval.models import mistral_eval

# from mmlu_test import evaluate_flan_spike
from mmlu_test.evaluate_flan_spike import main_mistral
from dataclasses import dataclass

def sample_top_p(probs: torch.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)


def sample(logits: torch.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


@torch.inference_mode()
def generate(prompts: List[str], model: Transformer, tokenizer: Tokenizer, *, max_tokens: int,  temperature: float, chunk_size: int = None):
    model = model.eval()
    B, V = len(prompts), model.args.vocab_size

    # Tokenize
    encoded_prompts = [tokenizer.encode(prompt, bos=True) for prompt in prompts]
    seqlens = [len(x) for x in encoded_prompts] # [5, 6, 17]
    # print(f'seqlens:{seqlens}')
    # for i in range(len(prompts)):
    #     print(f'prompt:{prompts[i]}; tokenized:{encoded_prompts[i]}')

    # Cache
    cache_window = max(seqlens) + max_tokens #17+35=52
    if model.args.sliding_window is not None and cache_window > model.args.sliding_window: # 4096
        cache_window = model.args.sliding_window
    cache = RotatingBufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache.to(device=model.device, dtype=model.dtype)
    cache.reset()
    
    # Bookkeeping
    logprobs = [[] for _ in range(B)] # [[], [], []]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens) # 17
    if chunk_size is None:
        chunk_size = max_prompt_len # 17

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s:s+chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model.forward(
            torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache
        )
        logits = torch.log_softmax(prelogits, dim=-1)

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(last_token_logits[i_seq, prompt_chunks[i_seq][0]].item())

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            logprobs[i_seq].extend([logits[offset + i, sequence[i + 1]].item() for i in range(len(sequence) - 1)])
            offset += len(sequence)

        last_token_prelogits = prelogits.index_select(0, torch.tensor([len(p) for p in prompt_chunks], device=prelogits.device).cumsum(dim=0) - 1)
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tokens = []
    assert last_token_prelogits is not None
    for i_token in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        generated_tokens.append(next_token[:, None])
        last_token_prelogits = model.forward(next_token, seqlens=[1] * len(prompts), cache=cache)
        assert last_token_prelogits.shape == (B, V)

    generated_words = []
    if generated_tokens:
        generated_tokens = torch.cat(generated_tokens, 1)
        for i, x in enumerate(encoded_prompts):
            generated_words.append(tokenizer.decode(x + generated_tokens[i].tolist()))

    return generated_words, logprobs


def interactive(model_path: str, max_tokens: int = 35, temperature: float = 0.7, instruct: bool = False):
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)

    while True:
        prompt = input("Prompt: ")
        if instruct:
            prompt = f"[INST] {prompt} [/INST]"
        res, _logprobs = generate(
            [prompt],
            transformer,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(res[0])
        print("=====================")


def demo(
    model_path: str, max_tokens: int = 1, temperature: float = 0, num_pipeline_ranks=1
):
    if num_pipeline_ranks > 1:
        torch.distributed.init_process_group()
        torch.cuda.set_device(torch.distributed.get_rank())
        should_print = torch.distributed.get_rank() == 0
    else:
        should_print = True
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(
        Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks
    )

    res, _logprobs = generate(
        [
            '''
           The following are multiple choice questions (with answers) about us foreign policy.

            Question: How did the 2008 financial crisis affect America's international reputation?
            A. It damaged support for the US model of political economy and capitalism
            B. It created anger at the United States for exaggerating the crisis
            C. It increased support for American global leadership under President Obama
            D. It reduced global use of the US dollar
            Answer:
            '''
        ],
        transformer,
        tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if should_print:
        for x,l in zip(res, _logprobs):
            print(x)
            logging.debug('Logprobs: %s',l)
            print("=====================")

@dataclass
class ARGS: 
    ntrain: int = 5
    ngpu: int = 2
    max_tokens: int = 1
    data_dir: str = "/home/ps/sj_files/mistral-src-main/eval_dataset/mmlu/data/"
    save_dir: str = "/home/ps/sj_files/mistral-src-main/mmlu_test/results_mistral/"
    model_name: str = "mistral-7b-spike"
    max_batch_size: int = 1
    temperature: float = 0
    num_pipeline_ranks: int = 1
    chunk_size = None

def mmlu(model_path: str):

    args = ARGS()
    
    assert args.model_name == "mistral-7b-spike", "Only support mistral-7b-spike now"
    # if args.model_name == "mistral-7b":
    #     print('evaluate normal mistral-7b')
    

    # tokenizer
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    logging.info(f'Loading Tokenizer from {str(Path(model_path) / "tokenizer.model")}')
    # transformer: mistral-7b
    transformer = Transformer.from_folder(
        Path(model_path), max_batch_size=args.max_batch_size, num_pipeline_ranks=args.num_pipeline_ranks)
    logging.info(f'Loading Transformer from {Path(model_path)}, max_batch_size={args.max_batch_size}, num_pipeline_ranks={args.num_pipeline_ranks}')
    # cache 


    main_mistral(model=transformer, tokenizer=tokenizer, buffer=RotatingBufferCache,temperature= args.temperature, max_tokens=args.max_tokens, args=args)

    
def eval(
    model_path: str, max_tokens: int = 1, temperature: float = 0.7, num_pipeline_ranks=1
):

    if num_pipeline_ranks > 1:
        torch.distributed.init_process_group()
        torch.cuda.set_device(torch.distributed.get_rank())
        should_print = torch.distributed.get_rank() == 0
    else:
        should_print = True

    # model = model.eval()
    # B, V = len(prompts), model.args.vocab_size

    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(
        Path(model_path), max_batch_size=1, num_pipeline_ranks=num_pipeline_ranks
    )
    Interface = mistral_eval.Mistral(
        transformer, 
        tokenizer, 
        Buffer=RotatingBufferCache, 
        token_chunk=None, 
        temperature=temperature,
        ) # 需要设置max_length吗？
    
    # lm_eval.tasks.initialize_tasks()
    # task_dir = os.path.dirname(os.path.abspath(__file__)) + "/" + "mmlu"
    # task_dir = "/home/ps/sj_files/LargeModelEvaluationHarness/lm-evaluation-harness/lm_eval/tasks/" + "/" + "mmlu"
    # task_dir = "/home/ps/sj_files/LargeModelEvaluationHarness/lm-evaluation-harness/lm_eval/tasks/" + "/" + "hellaswag"
    task_dir = "/home/ps/sj_files/LargeModelEvaluationHarness/lm-evaluation-harness/lm_eval/tasks/" + "/" + "winogrande"
    # task_dir = "/home/ps/sj_files/LargeModelEvaluationHarness/lm-evaluation-harness/lm_eval/tasks/" + "/" + "glue"
    # task_dir = "/home/ps/sj_files/LargeModelEvaluationHarness/lm-evaluation-harness/lm_eval/tasks/" + "/" + "wikitext"
    lm_eval.tasks.include_path(task_dir)
    results = lm_eval.simple_evaluate(
        model=Interface,
        tasks=["winogrande"],
        num_fewshot=5,
        batch_size=transformer.args.max_batch_size,
        max_batch_size=transformer.args.max_batch_size,
        device = transformer.device,

    )
    
    # task_name = task_dir.split('/')[-1]
    for key, value in results.items():
        print(f'{key}: {value}\n')
    # print(f'{task_name} result: {results}')
    # res, _logprobs = generate(
    #     [
    #         '''
    #        The following are multiple choice questions (with answers) about us foreign policy.

    #         Question: How did the 2008 financial crisis affect America's international reputation?
    #         A. It damaged support for the US model of political economy and capitalism
    #         B. It created anger at the United States for exaggerating the crisis
    #         C. It increased support for American global leadership under President Obama
    #         D. It reduced global use of the US dollar
    #         Answer:
    #         '''
    #     ],
    #     transformer,
    #     tokenizer,
    #     max_tokens=max_tokens,
    #     temperature=temperature,
    # )
    # if should_print:
    #     for x,l in zip(res, _logprobs):
    #         print(x)
    #         logging.debug('Logprobs: %s',l)
    #         print("=====================")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({
        "interactive": interactive,
        "demo": demo,
        "mmlu": mmlu,
        "eval": eval,
    })
