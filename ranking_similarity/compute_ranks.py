import numpy as np
from scipy.stats import rankdata

def compute_ranks(scores):
    """
    将分数数组转为秩（rank），最小为1，最大为N。
    参数：
        scores: np.ndarray 或 list，分数数组
    返回：
        ranks: np.ndarray，秩数组，shape=(N,)
    """
    scores = np.asarray(scores)
    ranks = rankdata(scores, method='average')
    return ranks
