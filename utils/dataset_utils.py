"""
Dataset utility functions for Mauritian Creole NLP thesis project.

Used by:
- BiLSTM
- T5-small
- ByT5-small
- Qwen2.5-0.5B
"""

import json
import os


def load_json(file_path):
    """
    Load a JSON dataset file.

    Args:
        file_path (str): Path to JSON file

    Returns:
        list: Dataset examples
    """

    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_splits(data_folder):
    """
    Load train, valid and test datasets.

    Expected structure:

    data/
        train.json
        valid.json
        test.json

    Returns:
        dict:
            {
                "train": [...],
                "valid": [...],
                "test": [...]
            }
    """

    datasets = {}

    for split in ["train", "valid", "test"]:

        file_path = os.path.join(
            data_folder,
            f"{split}.json"
        )

        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"{split}.json not found in {data_folder}"
            )

        datasets[split] = load_json(file_path)

        print(
            f"{split}: {len(datasets[split])} samples loaded"
        )

    return datasets