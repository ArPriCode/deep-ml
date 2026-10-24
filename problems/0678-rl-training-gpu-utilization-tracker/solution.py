def track_gpu_utilization(timestamps_ms: list, gpu_util_pct: list, phase_labels: list) -> dict:
    total_time_ms = timestamps_ms[-1] - timestamps_ms[0]
    
    if total_time_ms == 0:
        return {
            'total_time_ms': 0.0,
            'avg_gpu_util_pct': 0.0,
            'phase_stats': {},
            'bottleneck_phase': '',
            'gpu_idle_fraction_pct': 100.0
        }

    phase_data = {}
    phase_first_seen = {}
    weighted_util_sum = 0.0

    for i in range(len(gpu_util_pct)):
        duration = timestamps_ms[i+1] - timestamps_ms[i]
        util = gpu_util_pct[i]
        phase = phase_labels[i]

        weighted_util_sum += util * duration

        if phase not in phase_data:
            phase_data[phase] = {'total_time_ms': 0.0, 'weighted_util': 0.0, 'wasted_time': 0.0}
            phase_first_seen[phase] = i

        phase_data[phase]['total_time_ms'] += duration
        phase_data[phase]['weighted_util'] += util * duration
        phase_data[phase]['wasted_time'] += duration * (100.0 - util)

    avg_gpu_util_pct = weighted_util_sum / total_time_ms
    gpu_idle_fraction_pct = 100.0 - avg_gpu_util_pct

    phase_stats = {}
    for phase in sorted(phase_data.keys()):
        p_time = phase_data[phase]['total_time_ms']
        p_avg_util = phase_data[phase]['weighted_util'] / p_time if p_time > 0 else 0.0
        p_time_frac = (p_time / total_time_ms) * 100.0

        phase_stats[phase] = {
            'total_time_ms': round(p_time, 2),
            'avg_util_pct': round(p_avg_util, 2),
            'time_fraction_pct': round(p_time_frac, 2)
        }

    bottleneck_phase = max(
        phase_data.keys(),
        key=lambda p: (phase_data[p]['wasted_time'], -phase_first_seen[p])
    )

    return {
        'total_time_ms': round(total_time_ms, 2),
        'avg_gpu_util_pct': round(avg_gpu_util_pct, 2),
        'phase_stats': phase_stats,
        'bottleneck_phase': bottleneck_phase,
        'gpu_idle_fraction_pct': round(gpu_idle_fraction_pct, 2)
    }