def compute_throughput(intervals):
    if not intervals:
        return {'step_tps': [], 'cumulative_tps': [], 'overall_tps': 0.0}

    step_tps = []
    cumulative_tps = []
    cum_tokens = 0
    cum_seconds = 0.0

    for tokens, elapsed in intervals:
        if elapsed <= 0:
            raise ValueError("elapsed_seconds must be greater than 0")
        if tokens < 0:
            raise ValueError("tokens_processed cannot be negative")

        step_tps.append(round(tokens / elapsed, 2))

        cum_tokens += tokens
        cum_seconds += elapsed
        cumulative_tps.append(round(cum_tokens / cum_seconds, 2))

    overall_tps = round(cum_tokens / cum_seconds, 2)

    return {
        'step_tps': step_tps,
        'cumulative_tps': cumulative_tps,
        'overall_tps': overall_tps
    }