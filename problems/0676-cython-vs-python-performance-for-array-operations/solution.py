import math

def analyze_array_performance(n: int, operations: list) -> dict:
    ops_results = []
    crossover_sizes = {}
    
    total_py_us = 0.0
    total_cy_us = 0.0
    total_np_us = 0.0

    for op in operations:
        name = op['name']
        
        py_pe = op['python_per_elem_ns']
        cy_pe = op['cython_per_elem_ns']
        np_pe = op['numpy_per_elem_ns']
        
        py_oh = op['python_overhead_ns']
        cy_oh = op['cython_overhead_ns']
        np_oh = op['numpy_overhead_ns']
        
        py_us_raw = (py_oh + n * py_pe) / 1000.0
        cy_us_raw = (cy_oh + n * cy_pe) / 1000.0
        np_us_raw = (np_oh + n * np_pe) / 1000.0
        
        total_py_us += py_us_raw
        total_cy_us += cy_us_raw
        total_np_us += np_us_raw
        
        cy_speedup = py_us_raw / cy_us_raw
        np_speedup = py_us_raw / np_us_raw
        
        times = {'cython': cy_us_raw, 'numpy': np_us_raw, 'python': py_us_raw}
        best = min(times, key=lambda k: (times[k], k))
        
        ops_results.append({
            'name': name,
            'python_us': round(py_us_raw, 2),
            'cython_us': round(cy_us_raw, 2),
            'numpy_us': round(np_us_raw, 2),
            'cython_speedup': round(cy_speedup, 2),
            'numpy_speedup': round(np_speedup, 2),
            'best': best
        })
        
        if cy_pe <= np_pe:
            crossover = -1
        else:
            diff_oh = np_oh - cy_oh
            diff_pe = cy_pe - np_pe
            ratio = diff_oh / diff_pe
            if ratio <= 0:
                crossover = 0
            else:
                crossover = math.ceil(ratio)
                
        crossover_sizes[name] = crossover

    pipeline_times = {'cython': total_cy_us, 'numpy': total_np_us, 'python': total_py_us}
    recommended = min(pipeline_times, key=lambda k: (pipeline_times[k], k))

    return {
        'operations': ops_results,
        'pipeline_python_us': round(total_py_us, 2),
        'pipeline_cython_us': round(total_cy_us, 2),
        'pipeline_numpy_us': round(total_np_us, 2),
        'recommended': recommended,
        'crossover_sizes': crossover_sizes
    }