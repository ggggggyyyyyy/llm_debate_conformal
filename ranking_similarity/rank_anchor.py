import numpy as np
from sklearn.linear_model import LinearRegression
from .compute_ranks import compute_ranks

def simple_score(claims):
    return np.array([len(str(c)) for c in claims])

def rank_regression(scores1, scores2, anchor_indices):
    # 先将分数转为秩（rank）
    ranks1 = compute_ranks(scores1)
    ranks2 = compute_ranks(scores2)
    X = ranks1[anchor_indices].reshape(-1, 1)
    y = ranks2[anchor_indices]
    reg = LinearRegression().fit(X, y)
    alpha = reg.coef_[0]
    beta = reg.intercept_
    return alpha, beta, reg

def joint_rank_and_select(claims1, claims2, anchors=None, top_ratio=0.7):
    from .similarity import compute_claim_similarity
    scores1 = simple_score(claims1)
    scores2 = simple_score(claims2)
    # 用 similarity 选 anchor
    if anchors is None:
        # 计算 claims1 和 claims2 的相似度矩阵
        sim_matrix = compute_claim_similarity(claims1, claims2)
        # 展平成一维，找到前20%最大similarity的索引
        num_pairs = sim_matrix.size
        top_n = int(np.ceil(num_pairs * 0.2))
        flat_indices = np.argpartition(-sim_matrix.flatten(), top_n-1)[:top_n]
        # 转换为二维索引
        anchor_pairs = [np.unravel_index(idx, sim_matrix.shape) for idx in flat_indices]
        # 只用 claims1 的索引作为anchor
        anchors = np.array([pair[0] for pair in anchor_pairs])
    alpha, beta, reg = rank_regression(scores1, scores2, anchors)
    ranks1 = compute_ranks(scores1)
    ranks2 = compute_ranks(scores2)
    ranks1_aligned = alpha * ranks1 + beta
    # 合并 ranks2 和 ranks1_aligned
    combined_ranks = np.concatenate([ranks2, ranks1_aligned])
    combined_indices = np.concatenate([np.arange(len(ranks2)), np.arange(len(ranks1_aligned))])
    # 按 rank 降序排序（rank越大越靠前）
    idx_sorted = np.argsort(-combined_ranks)
    # 去重：只保留每个 claim 的第一个出现
    seen = set()
    unique_indices = []
    for idx in idx_sorted:
        claim_idx = combined_indices[idx]
        if claim_idx not in seen:
            seen.add(claim_idx)
            unique_indices.append(claim_idx)
    top_n = max(2, int(len(unique_indices) * top_ratio))
    selected_indices = unique_indices[:top_n]
    return np.array(selected_indices), combined_ranks
