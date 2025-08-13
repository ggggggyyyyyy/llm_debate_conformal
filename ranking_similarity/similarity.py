from sentence_transformers import SentenceTransformer
import numpy as np

def compute_claim_similarity(claims1, claims2, model_name='all-MiniLM-L6-v2', batch_size=32):
    """
    用 SBERT 计算两个 LLM 输出 claim 的相似度矩阵。
    参数：
        claims1: list[str]，LLM1 的 claim 列表
        claims2: list[str]，LLM2 的 claim 列表
        model_name: str，SBERT 模型名，默认 'all-MiniLM-L6-v2'
        batch_size: int，编码时的 batch size
    返回：
        sim_matrix: np.ndarray，相似度矩阵，shape=(len(claims1), len(claims2))
    """
    model = SentenceTransformer(model_name)
    emb1 = model.encode(claims1, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    emb2 = model.encode(claims2, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    # 归一化
    emb1 = emb1 / np.linalg.norm(emb1, axis=1, keepdims=True)
    emb2 = emb2 / np.linalg.norm(emb2, axis=1, keepdims=True)
    sim_matrix = np.dot(emb1, emb2.T)
    return sim_matrix
