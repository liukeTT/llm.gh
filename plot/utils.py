import json
import argparse
import numpy as np


def load_json(filename):
    data = None
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def write_json(filename, data):
    with open(filename, "w") as f:
        f.write(json.dumps(data))


