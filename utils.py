import os
import json
from argparse import ArgumentParser


def load_label_dict(label_path: str = "model/coco_labels.txt"):
    model_dict = {}
    file_exist = os.path.exists(label_path)
    if file_exist:
        with open(label_path, "r") as txt_file:
            labels = txt_file.read().splitlines()

            model_dict = {i: label for i, label in enumerate(labels)}
    else:
        print("File not found")

    return model_dict
