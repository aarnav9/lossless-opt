"""One address expression renders to both Lean and Metal; no floating arithmetic."""

ADDRESS = ("+", ("*", ("+", ("*", "row", "capacity"), "time"), "width"), "lane")


def render(expr=ADDRESS, names=None):
    names = names or {}
    if isinstance(expr, str):
        return names.get(expr, expr)
    op, left, right = expr
    if op not in {"+", "*"}:
        raise ValueError("unsupported address operation")
    return f"({render(left, names)} {op} {render(right, names)})"


def evaluate(row, capacity, time, width, lane):
    values = dict(row=row, capacity=capacity, time=time, width=width, lane=lane)

    def go(expr):
        if isinstance(expr, str):
            return values[expr]
        op, a, b = expr
        return go(a) + go(b) if op == "+" else go(a) * go(b)

    return go(ADDRESS)


def lean_source():
    return (
        "-- Generated from indexing.ADDRESS; checked before each round.\n"
        "import Std\nnamespace CacheAddress\n"
        f"def address (row capacity time width lane : Nat) : Nat := {render()}\n"
        "end CacheAddress\n"
    )


def metal_source():
    old = render(names={"capacity": "old_capacity"})
    new = render(names={"capacity": "added", "time": "(time - previous)"})
    return f"""
        uint i = thread_position_in_grid.x;
        uint old_capacity = meta[0], previous = meta[1], added = meta[2];
        uint total = previous + added, width = meta[3], size = meta[4];
        if (i < size) {{
            uint lane = i % width;
            uint time = (i / width) % total;
            uint row = (i / width) / total;
            if (time < previous) {{
                uint source = {old};
                outk[i] = oldk[source];
                outv[i] = oldv[source];
            }} else {{
                uint source = {new};
                outk[i] = newk[source];
                outv[i] = newv[source];
            }}
        }}
    """


def vector_source(width):
    if width not in (2, 4):
        raise ValueError("supported vector widths are 2 and 4")
    old = render(names={"capacity": "old_capacity"})
    new = render(names={"capacity": "added", "time": "(time - previous)"})
    return f"""
        uint i = thread_position_in_grid.x * {width};
        uint old_capacity = meta[0], previous = meta[1], added = meta[2];
        uint total = previous + added, width = meta[3], size = meta[4];
        if (i < size) {{
            uint lane = i % width;
            uint time = (i / width) % total;
            uint row = (i / width) / total;
            uint source = time < previous ? {old} : {new};
            uint{width} k, v;
            if (time < previous) {{
                k = uint{width}({", ".join(f"as_type<uint>(oldk[source+{j}])" for j in range(width))});
                v = uint{width}({", ".join(f"as_type<uint>(oldv[source+{j}])" for j in range(width))});
            }} else {{
                k = uint{width}({", ".join(f"as_type<uint>(newk[source+{j}])" for j in range(width))});
                v = uint{width}({", ".join(f"as_type<uint>(newv[source+{j}])" for j in range(width))});
            }}
            reinterpret_cast<device uint{width}*>(outk)[i/{width}] = k;
            reinterpret_cast<device uint{width}*>(outv)[i/{width}] = v;
        }}
    """


VARIANTS = {"flat", "rows", "fixed_width", "rows_fixed_width", "rows_two"}


def source(variant):
    if variant not in VARIANTS:
        raise ValueError("unsupported copy variant")
    if variant == "flat":
        return metal_source()
    row_grid = variant.startswith("rows")
    fixed = variant in {"fixed_width", "rows_fixed_width"}
    chunk = 2 if variant == "rows_two" else 1
    address_old = render(names={"capacity": "old_capacity"})
    address_new = render(names={"capacity": "added", "time": "(time - previous)"})
    geometry = (
        f"uint row = thread_position_in_grid.y; uint j = thread_position_in_grid.x * {chunk} + offset;\n"
        "uint i = row * total * width + j;\n"
        "if (j < total * width) { uint lane = j % width; uint time = j / width;"
        if row_grid
        else "uint i = thread_position_in_grid.x;\n"
        "if (i < size) { uint lane = i % width; uint time = (i / width) % total; uint row = (i / width) / total;"
    )
    return f"""
      uint old_capacity = meta[0], previous = meta[1], added = meta[2];
      uint total = previous + added, width = {"W" if fixed else "meta[3]"}, size = meta[4];
      for (uint offset = 0; offset < {chunk}; ++offset) {{
        {geometry}
          if (time < previous) {{
            uint source = {address_old};
            outk[i] = oldk[source]; outv[i] = oldv[source];
          }} else {{
            uint source = {address_new};
            outk[i] = newk[source]; outv[i] = newv[source];
          }}
        }}
      }}
    """
