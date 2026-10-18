import numpy as np
from collections import Counter

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    c_len = len(candidate)
    if c_len == 0:
        return 0.0

    ref_lens = [len(ref) for ref in references]
    best_ref_len = min(ref_lens, key=lambda r_len: (abs(r_len - c_len), r_len))

    precisions = []

    for n in range(1, max_n + 1):
        if c_len < n:
            return 0.0

        cand_ngrams = Counter(tuple(candidate[i:i+n]) for i in range(c_len - n + 1))
        
        max_ref_counts = Counter()
        for ref in references:
            ref_ngrams = Counter(tuple(ref[i:i+n]) for i in range(len(ref) - n + 1))
            for ngram, count in ref_ngrams.items():
                max_ref_counts[ngram] = max(max_ref_counts[ngram], count)

        clipped_count = 0
        for ngram, count in cand_ngrams.items():
            clipped_count += min(count, max_ref_counts[ngram])

        total_cand_ngrams = c_len - n + 1
        p_n = clipped_count / total_cand_ngrams

        if p_n == 0:
            return 0.0

        precisions.append(p_n)

    if c_len > best_ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - best_ref_len / c_len)

    geo_mean = np.exp(np.mean(np.log(precisions)))

    return float(bp * geo_mean)