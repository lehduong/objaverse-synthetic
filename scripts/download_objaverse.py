import argparse
import json
import random
import os
from dataclasses import dataclass

import boto3
import objaverse
import tyro
from tqdm import tqdm


@dataclass
class Args:
    """We will sort a list of uid from start_i to end_i and select n_objects most starred"""
    
    start_i: int
    """total number of files uploaded"""

    end_i: int
    """total number of files uploaded"""

    n_objects: int = 10
    """total number of objects will be rendered."""
    
    skip_completed: bool = False
    """whether to skip the files that have already been downloaded"""
    
    uid_json_path: str = None
    """path to json file contains all uids that need to be download"""


def get_completed_uids():
    # get all the files in the objaverse-images bucket
    s3 = boto3.resource("s3")
    bucket = s3.Bucket("objaverse-images")
    bucket_files = [obj.key for obj in tqdm(bucket.objects.all())]

    dir_counts = {}
    for file in bucket_files:
        d = file.split("/")[0]
        dir_counts[d] = dir_counts.get(d, 0) + 1

    # get the directories with 12 files
    dirs = [d for d, c in dir_counts.items() if c == 12]
    return set(dirs)


# set the random seed to 42
if __name__ == "__main__":
    args = tyro.cli(Args)
    object_paths = objaverse._load_object_paths()
    
    if os.path.exists(args.uid_json_path):
        with open(args.uid_json_path) as f:
            uids = json.load(f)
    else:
        random.seed(42)
        uids = objaverse.load_uids()
        random.shuffle(uids)
        uids = uids[args.start_i : args.end_i]
        annotation = objaverse.load_annotations(uids)
        uids = sorted(uids, key=lambda x: annotation[x]['likeCount'], reverse=True)[: args.n_objects]
    
    # get the uids that have already been downloaded
    if args.skip_completed:
        completed_uids = get_completed_uids()
        uids = [uid for uid in uids if uid not in completed_uids]

    uid_object_paths = [
        f"https://huggingface.co/datasets/allenai/objaverse/resolve/main/{object_paths[uid]}"
        for uid in uids
    ]

    with open("input_models_path.json", "w") as f:
        json.dump(uid_object_paths, f, indent=2)
