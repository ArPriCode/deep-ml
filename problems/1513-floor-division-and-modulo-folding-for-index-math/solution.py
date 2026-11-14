def term_bounds(terms, ranges):
    vmin = terms.get("", 0)
    vmax = terms.get("", 0)
    for k, v in terms.items():
        if k == "":
            continue
        lo, hi = ranges[k]
        v1, v2 = v * lo, v * (hi - 1)
        vmin += min(v1, v2)
        vmax += max(v1, v2)
    return (vmin, vmax)

def fold_div(terms, c, ranges):
    q = {"": 0}
    r = {"": 0}
    for k, v in terms.items():
        if k == "":
            if v % c == 0:
                q[""] = v // c
            else:
                r[""] = v
            continue
        if v % c == 0:
            if v // c != 0:
                q[k] = v // c
        else:
            r[k] = v
    vmin, vmax = term_bounds(r, ranges)
    if vmin >= 0 and vmax < c:
        return (q, None)
    return (q, r)

def fold_mod(terms, c, ranges):
    rem = {}
    dropped_empty = False
    for k, v in terms.items():
        if v % c == 0:
            if k == "":
                dropped_empty = True
        else:
            rem[k] = v
    if dropped_empty or "" not in rem:
        rem[""] = 0
        
    vmin, vmax = term_bounds(rem, ranges)
    exact = (vmin >= 0 and vmax < c)
    return (rem, exact)

def unflatten(shape, flat_terms, ranges):
    strides = []
    stride = 1
    for d in reversed(shape):
        strides.append(stride)
        stride *= d
    strides.reverse()
    
    res = []
    for s_j, shape_j in zip(strides, shape):
        q, r = fold_div(flat_terms, s_j, ranges)
        rem, exact = fold_mod(q, shape_j, ranges)
        if r is None and exact:
            res.append(rem)
        else:
            res.append("div/mod")
    return res