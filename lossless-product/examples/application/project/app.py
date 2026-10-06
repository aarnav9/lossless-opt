"""An ordinary numerical application: no Lossless imports or adapter methods."""

import json
import sys

import numpy as np


def analyze(rows, scale=1.0):
    values = np.asarray(rows, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    features = np.tanh(centered * scale)
    return {"features": features, "energy": float(np.sum(features * features))}


if __name__ == "__main__":
    request = json.load(sys.stdin)
    result = analyze(request["rows"], request.get("scale", 1.0))
    print(json.dumps({"features": result["features"].tolist(), "energy": result["energy"]}))
