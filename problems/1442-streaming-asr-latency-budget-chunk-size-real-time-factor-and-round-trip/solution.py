import math

def chunk_latency(chunk_s, rtf, rtt_s):
    return round(chunk_s + rtf * chunk_s + rtt_s, 4)

def largest_chunk_under(budget_s, rtf, rtt_s, step=0.05):
    best_chunk = None
    k = 1
    while True:
        chunk_s = k * step
        latency = chunk_latency(chunk_s, rtf, rtt_s)
        if latency <= budget_s + 1e-9:
            best_chunk = chunk_s
            k += 1
        else:
            break
    
    return round(best_chunk, 2) if best_chunk is not None else None

def concurrent_streams(rtf, gpu_share=1.0):
    return math.floor(gpu_share / rtf)

def end_of_speech_latency(hangover_s, chunk_s, rtf, rtt_s):
    return round(hangover_s + chunk_latency(chunk_s, rtf, rtt_s), 4)

def budget_breakdown(chunk_s, rtf, rtt_s):
    total = chunk_s + rtf * chunk_s + rtt_s
    if total == 0:
        return {"wait": 0.0, "model": 0.0, "network": 0.0}
    return {
        "wait": round(chunk_s / total, 4),
        "model": round((rtf * chunk_s) / total, 4),
        "network": round(rtt_s / total, 4)
    }