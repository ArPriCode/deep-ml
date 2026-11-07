def size(expr):
    if not isinstance(expr, tuple):
        return 1
    if expr[0] == "neg":
        return 1 + size(expr[1])
    return 1 + size(expr[1]) + size(expr[2])


def is_const(expr):
    return isinstance(expr, (int, float)) and not isinstance(expr, bool)


def step(expr):
    if not isinstance(expr, tuple):
        return expr

    op = expr[0]
    if op == "neg":
        a = step(expr[1])
        if is_const(a):
            return -a
        if isinstance(a, tuple) and a[0] == "neg":
            return a[1]
        return ("neg", a)

    if op in ("add", "mul"):
        a = step(expr[1])
        b = step(expr[2])

        if is_const(a) and is_const(b):
            return a + b if op == "add" else a * b

        if op == "add":
            if b == 0:
                return a
            if a == 0:
                return b
        elif op == "mul":
            if a == 0 or b == 0:
                return 0
            if b == 1:
                return a
            if a == 1:
                return b

        if is_const(a) and not is_const(b):
            a, b = b, a

        if op == "add":
            if isinstance(a, tuple) and a[0] == "add" and is_const(a[2]) and is_const(b):
                return ("add", a[1], a[2] + b)
            if not is_const(b) and isinstance(b, tuple) and b[0] == "add" and is_const(b[2]):
                return ("add", ("add", a, b[1]), b[2])
            if isinstance(a, tuple) and a[0] == "add" and is_const(a[2]) and not is_const(b):
                return ("add", ("add", a[1], b), a[2])
        elif op == "mul":
            if isinstance(a, tuple) and a[0] == "mul" and is_const(a[2]) and is_const(b):
                return ("mul", a[1], a[2] * b)
            if isinstance(a, tuple) and a[0] == "add" and is_const(b):
                return ("add", ("mul", a[1], b), ("mul", a[2], b))

        return (op, a, b)

    return expr


def simplify(expr):
    while True:
        nxt = step(expr)
        if nxt == expr:
            return nxt
        expr = nxt