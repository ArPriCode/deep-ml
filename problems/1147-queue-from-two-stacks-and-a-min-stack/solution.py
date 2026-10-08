from collections import deque

def process_operations(operations):
    queue = deque()
    min_stack = [] 
    output = []

    for op in operations:
        op_type = op[0]

        if op_type == "enqueue":
            queue.append(op[1])
        elif op_type == "dequeue":
            output.append(queue.popleft())
        elif op_type == "mpush":
            val = op[1]
            current_min = val if not min_stack else min(val, min_stack[-1][1])
            min_stack.append((val, current_min))
        elif op_type == "mpop":
            output.append(min_stack.pop()[0])
        elif op_type == "mtop":
            output.append(min_stack[-1][0])
        elif op_type == "mmin":
            output.append(min_stack[-1][1])

    return output