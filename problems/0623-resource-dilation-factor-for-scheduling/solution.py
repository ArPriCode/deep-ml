def compute_dilation_factor(tasks: list, capacity: int) -> dict:
    n = len(tasks)
    memo = {}

    def cp(i):
        if i in memo:
            return memo[i]
        if not tasks[i]["dependencies"]:
            memo[i] = tasks[i]["duration"]
        else:
            memo[i] = tasks[i]["duration"] + max(cp(d) for d in tasks[i]["dependencies"])
        return memo[i]

    lower_cp = max(cp(i) for i in range(n))
    work = sum(t["duration"] * t["resources"] for t in tasks)
    lower_bound = float(max(lower_cp, work / capacity))

    start = [None] * n
    finish = [None] * n
    scheduled = [False] * n
    completed = set()
    used = 0
    time = 0.0

    while len(completed) < n:
        for i in range(n):
            if scheduled[i] and i not in completed and finish[i] <= time:
                completed.add(i)
                used -= tasks[i]["resources"]

        for i in range(n):
            if scheduled[i]:
                continue

            if all(d in completed for d in tasks[i]["dependencies"]):
                r = tasks[i]["resources"]

                if used + r <= capacity:
                    start[i] = time
                    finish[i] = time + tasks[i]["duration"]
                    used += r
                    scheduled[i] = True

        if len(completed) == n:
            break

        time = min(
            finish[i]
            for i in range(n)
            if scheduled[i] and i not in completed
        )

    actual = float(max(finish))

    return {
        "dilation_factor": round(actual / lower_bound, 4),
        "actual_makespan": round(actual, 4),
        "lower_bound": round(lower_bound, 4),
        "start_times": [round(float(x), 4) for x in start]
    }