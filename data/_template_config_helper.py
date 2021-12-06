#!/usr/bin/env python
import itertools
from pathlib import Path

import numpy as np
import yaml


def strings_from_values(values, name):
    strings = []
    values = np.sort(values)
    assert len(values) > 0
    strings.append(f"{name} <= {values[0]}")
    for i in range(len(values) - 1):
        strings.append(f"({name} > {values[i]}) & ({name} <= {values[i + 1]})")
    strings.append(f"{name} > {values[-1]}")
    return strings


def get_categories():
    template_ranges = {
        "b_tag1": np.arange(1, 5) / 5,
        "n_charged_hadrons": np.concatenate(
            [np.arange(10), [10, 12, 15, 20, 25, 30, 40]]
        ),
    }
    strings_1d = {k: strings_from_values(v, k) for k, v in template_ranges.items()}
    cat_strings = {}
    for criteria in itertools.product(*strings_1d.values()):
        name = "(" + ") & (".join(criteria) + ")"
        cat_strings[name] = list(criteria)
    print(f"{len(cat_strings)} categories for the variables {list(template_ranges)}")
    return cat_strings


def categories_to_string(categories):
    as_str = yaml.dump(categories)
    as_str = "    " + as_str.replace("\n", "\n    ")
    return as_str


def read_write_config(categories):
    skeleton_path = Path(__file__).parent / "_skeleton_without_categories.yaml"
    filled_config_path = skeleton_path.parent / "higgstables-config.yaml"

    with skeleton_path.open("r") as f:
        skeleton = f.read()

    categories_string = categories_to_string(categories)
    filled_config = skeleton.replace("# CATEGORIES GO HERE", categories_string)
    with filled_config_path.open("w") as f:
        f.write(filled_config)


if __name__ == "__main__":
    categories = get_categories()
    read_write_config(categories)
