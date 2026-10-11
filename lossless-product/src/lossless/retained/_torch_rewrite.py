"""Narrow checked FX normalization rewrite for immutable research models."""

import operator
import torch


def is_call(node, target):
    return isinstance(node, torch.fx.Node) and node.op == "call_function" and node.target == target


def rewrite(gm, decision):
    changes = []
    for y in list(gm.graph.nodes):
        if not is_call(y, operator.mul):
            continue
        scaled, weight = y.args
        if not isinstance(weight, torch.fx.Node) or weight.op != "get_attr":
            continue
        if not is_call(scaled, operator.mul):
            continue
        z, inv = scaled.args
        if not is_call(inv, torch.rsqrt):
            continue
        eps = inv.args[0]
        if not is_call(eps, operator.add) or len(eps.args) != 2 or eps.args[1] != 1e-5:
            continue
        mean = eps.args[0]
        if not isinstance(mean, torch.fx.Node) or mean.op != "call_method" or mean.target != "mean":
            continue
        if mean.kwargs != {"dim": -1, "keepdim": True}:
            continue
        square = mean.args[0]
        if not is_call(square, operator.mul) or square.args != (z, z):
            continue
        if not is_call(z, operator.add) or len(z.args) != 2:
            continue
        if any(len(node.users) != 1 for node in (scaled, inv, eps, mean, square)):
            continue
        w = gm
        for name in weight.target.split("."):
            w = getattr(w, name)
        if (
            w.dtype != torch.float32
            or w.device.type != "cpu"
            or w.requires_grad
            or not w.is_contiguous()
        ):
            continue
        x, residual = z.args
        if decision.get("fuse_mlp"):
            projections = list(y.users)
            if len(projections) != 2 or any(
                n.op != "call_module" or n.args != (y,) for n in projections
            ):
                continue
            gate = next(
                (
                    n
                    for n in projections
                    if len(n.users) == 1 and is_call(next(iter(n.users)), torch.nn.functional.silu)
                ),
                None,
            )
            if gate is None:
                continue
            up = next(n for n in projections if n is not gate)
            silu = next(iter(gate.users))
            if len(silu.users) != 1 or len(up.users) != 1:
                continue
            product = next(iter(silu.users))
            if (
                not is_call(product, operator.mul)
                or set(product.args) != {silu, up}
                or len(product.users) != 1
            ):
                continue
            down = next(iter(product.users))
            if down.op != "call_module" or down.args != (product,) or len(down.users) != 1:
                continue
            tail = next(iter(down.users))
            if (
                not is_call(tail, operator.add)
                or set(tail.args) != {down, z}
                or set(z.users) != {square, scaled, tail}
            ):
                continue
            modules = [gm.get_submodule(n.target) for n in (gate, up, down)]
            if any(
                type(m) is not torch.nn.Linear
                or m.training
                or m._forward_hooks
                or m._forward_pre_hooks
                or any(
                    p.requires_grad or p.dtype != torch.float32 or p.device.type != "cpu"
                    for p in m.parameters()
                )
                for m in modules
            ):
                continue
            with gm.graph.inserting_before(y):
                params = []
                extra_bytes = 0
                for index, node in enumerate((gate, up, down)):
                    weight_target = node.target + ".weight"
                    if decision.get("fold_weights") and index < 2:
                        folded = (modules[index].weight * w.reshape(1, -1)).contiguous()
                        weight_target = "_folded_" + str(len(changes)) + "_" + str(index)
                        gm.register_buffer(weight_target, folded)
                        extra_bytes += folded.numel() * folded.element_size()
                    params.extend(
                        [
                            gm.graph.get_attr(weight_target),
                            gm.graph.get_attr(node.target + ".bias")
                            if modules[index].bias is not None
                            else None,
                        ]
                    )
                operation = (
                    torch.ops.lossless_retained.rms_mlp_folded.default
                    if decision.get("fold_weights")
                    else torch.ops.lossless_retained.rms_mlp.default
                )
                fused = gm.graph.call_function(
                    operation,
                    (
                        x,
                        residual,
                        weight,
                        *params,
                        decision.get("algorithm", 0),
                        decision.get("grain", 0),
                    ),
                )
            tail.replace_all_uses_with(fused)
            for obsolete in (
                tail,
                down,
                product,
                silu,
                gate,
                up,
                y,
                scaled,
                inv,
                eps,
                mean,
                square,
                z,
            ):
                gm.graph.erase_node(obsolete)
            changes.append(
                {
                    "region": y.name,
                    "operation": str(operation),
                    "linear_shapes_preserved": True,
                    "algorithm": decision.get("algorithm", 0),
                    "extra_parameter_bytes": extra_bytes,
                }
            )
            continue
        if decision.get("fuse_region_output"):
            users = list(y.users)
            if (
                len(users) != 1
                or not is_call(users[0], operator.add)
                or set(users[0].args) != {y, z}
            ):
                continue
            tail = users[0]
            if set(z.users) != {square, scaled, tail}:
                continue
            with gm.graph.inserting_before(y):
                fused = gm.graph.call_function(
                    torch.ops.lossless_retained.rms_region.default,
                    (x, residual, weight, decision.get("algorithm", 0), decision.get("grain", 0)),
                )
            tail.replace_all_uses_with(fused)
            for obsolete in (tail, y, scaled, inv, eps, mean, square, z):
                gm.graph.erase_node(obsolete)
            changes.append(
                {
                    "region": y.name,
                    "weight": weight.target,
                    "fused_output": True,
                    "algorithm": decision.get("algorithm", 0),
                }
            )
            continue
        with gm.graph.inserting_before(y):
            fused = gm.graph.call_function(
                torch.ops.lossless_retained.rms_pre.default,
                (x, residual, weight, decision.get("algorithm", 0), decision.get("grain", 0)),
            )
            normalized = gm.graph.call_function(operator.getitem, (fused, 0))
            added = gm.graph.call_function(operator.getitem, (fused, 1))
        y.replace_all_uses_with(normalized)
        z.replace_all_uses_with(added)
        # Remove the obsolete chain before DCE's topological-order check.
        # Its old users now temporarily refer to the later tuple result.
        for obsolete in (y, scaled, inv, eps, mean, square, z):
            gm.graph.erase_node(obsolete)
        changes.append(
            {
                "region": y.name,
                "weight": weight.target,
                "algorithm": decision.get("algorithm", 0),
                "grain": decision.get("grain", 0),
            }
        )
    gm.graph.eliminate_dead_code()
    gm.graph.lint()
    gm.recompile()
    return changes
