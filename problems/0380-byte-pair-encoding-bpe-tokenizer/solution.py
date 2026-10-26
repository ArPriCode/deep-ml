from collections import defaultdict

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    merges = []
    
    for _ in range(num_merges):
        pair_counts = defaultdict(int)
        
        for word, freq in corpus.items():
            tokens = word.split()
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_counts[pair] += freq
                
        if not pair_counts:
            break
            
        best_pair = max(pair_counts, key=pair_counts.get)
        merges.append(best_pair)
        
        target = f"{best_pair[0]} {best_pair[1]}"
        replacement = f"{best_pair[0]}{best_pair[1]}"
        
        new_corpus = {}
        for word, freq in corpus.items():
            tokens = word.split()
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i + 1] == best_pair[1]:
                    new_tokens.append(replacement)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_corpus[' '.join(new_tokens)] = freq
            
        corpus = new_corpus
        
    return merges