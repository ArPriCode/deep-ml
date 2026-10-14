from collections import Counter

def balance_undersample(data: list) -> list:
    if not data:
        return []

    counts = Counter(label for _, label in data)
    min_count = min(counts.values())

    class_counts = Counter()
    result = []

    for item in data:
        sample, label = item
        if class_counts[label] < min_count:
            result.append(item)
            class_counts[label] += 1

    return result