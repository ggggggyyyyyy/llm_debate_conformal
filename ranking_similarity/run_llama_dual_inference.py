# run_llama_dual_inference.py
"""
用两个不同的Llama模型分别对同一批prompt做推理，生成llama1_results.csv和llama2_results.csv，便于Rank-Anchor OT-CP实验。
"""
import os
import pickle
import pandas as pd
from tqdm import tqdm
from src.llama_client import LlamaClient

MODEL1_PATH = r'C:/Users/chris/OneDrive/Documents/GUYUpyfile/llm_debate/models/Llama-2-7b-chat-hf'
MODEL2_PATH = r'C:/Users/chris/OneDrive/Documents/GUYUpyfile/llm_debate/models/Llama-3.2-3b'
CACHE1 = 'llama1_responses.pkl'
CACHE2 = 'llama2_responses.pkl'
DATASET_PATH = r'c:/Users/chris/OneDrive/Documents/GUYUpyfile/conformal-safety/data/factscore_final_dataset.pkl'
RESULT1 = 'llama1_results.csv'
RESULT2 = 'llama2_results.csv'

def load_dataset(filepath):
    with open(filepath, 'rb') as fp:
        dataset = pickle.load(fp)
    return dataset

def run_inference(model_path, cache_file, prompts):
    client = LlamaClient(cache_file=cache_file, model_path=model_path)
    results = []
    for prompt in tqdm(prompts):
        output = client._query(prompt, max_tokens=1000)[0]['message']
        results.append({'prompt': prompt, 'llama_response': output})
    return results

def main():
    dataset = load_dataset(DATASET_PATH)
    prompts = [item['prompt'] for item in dataset]
    # Llama-2-7b-chat-hf
    results1 = run_inference(MODEL1_PATH, CACHE1, prompts)
    pd.DataFrame(results1).to_csv(RESULT1, index=False)
    print(f"Llama-2-7b-chat-hf 结果已保存到 {RESULT1}")
    # Llama-3.2-3b
    results2 = run_inference(MODEL2_PATH, CACHE2, prompts)
    pd.DataFrame(results2).to_csv(RESULT2, index=False)
    print(f"Llama-3.2-3b 结果已保存到 {RESULT2}")

if __name__ == '__main__':
    main()
