"""Bounded relative RGB gains estimated only from shared observed surface samples."""

import numpy as np


def estimate_gains(samples, visible):
    count = len(samples)
    gains = np.ones((count, 3), np.float32)
    report = {
        "applied": False,
        "reason": "No trustworthy photometric overlap",
        "pairs": 0,
    }
    edges = []
    for i in range(count):
        for j in range(i + 1, count):
            mask = (
                visible[i]
                & visible[j]
                & np.all(
                    (samples[i] > 0.05)
                    & (samples[i] < 0.95)
                    & (samples[j] > 0.05)
                    & (samples[j] < 0.95),
                    axis=1,
                )
            )
            ids = np.flatnonzero(mask)
            if len(ids) < 128:
                continue
            train, heldout = ids[::2], ids[1::2]
            ratio = np.log(samples[i, train] / samples[j, train])
            median = np.median(ratio, axis=0)
            if np.max(np.median(np.abs(ratio - median), axis=0)) > 0.08:
                continue
            edges.append((i, j, median, heldout))
    if not edges:
        return gains, report
    matrix, values = [], []
    neighbours = [set() for _ in range(count)]
    for i, j, ratio, _ in edges:
        row = np.zeros(count)
        row[j] = 1
        row[i] = -1
        matrix.append(row)
        values.append(ratio)
        neighbours[i].add(j)
        neighbours[j].add(i)
    visited = set()
    for anchor in range(count):
        if anchor in visited:
            continue
        stack = [anchor]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(neighbours[node] - visited)
        row = np.zeros(count)
        row[anchor] = 1
        matrix.append(row)
        values.append(np.zeros(3))
    logs = np.linalg.lstsq(np.array(matrix), np.array(values), rcond=None)[0]
    proposed = np.clip(np.exp(logs), 2 / 3, 1.5)
    before, after = [], []
    for i, j, _, ids in edges:
        before.append(float(np.mean(np.abs(samples[i, ids] - samples[j, ids]))))
        after.append(
            float(
                np.mean(
                    np.abs(
                        samples[i, ids] * proposed[i] - samples[j, ids] * proposed[j]
                    )
                )
            )
        )
    report.update(
        pairs=len(edges),
        validation_before=float(np.mean(before)),
        validation_after=float(np.mean(after)),
    )
    if np.mean(before) < 0.005:
        report["reason"] = "Views already photometrically consistent"
    elif np.mean(after) >= 0.9 * np.mean(before) or any(
        a > b + 0.005 for a, b in zip(after, before)
    ):
        report["reason"] = "Held-out overlap did not improve consistently"
    else:
        gains = proposed.astype(np.float32)
        report.update(
            applied=True, reason="Bounded gains improved held-out surface samples"
        )
    report["gains"] = gains.tolist()
    return gains, report
