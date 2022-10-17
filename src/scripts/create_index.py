import os, sys


source_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(source_path)

from utils import SentencesIndex, file_dir_or_pathlist_to_fid_fpath_dict
from config import DEFAULT_INDEX_DIR
from typing import List
from argparse import ArgumentParser
from pathlib import Path
from typing import Union


def build_index(fileordirpaths_list: List[str], save_path: Union[str, Path], device: str):
    fid2fpath_dict = file_dir_or_pathlist_to_fid_fpath_dict(fileordirpaths_list)
    index = SentencesIndex.from_fid_fpath_dict(
        fid2fpath_dict=fid2fpath_dict, indexing_device=device
    )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    index.save_sentence_index(save_path=save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("fileordirpaths", action="store", type=str, nargs="+")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_path", type=str, default=DEFAULT_INDEX_DIR)
    args = parser.parse_args()
    build_index(
        fileordirpaths_list=args.fileordirpaths, save_path=args.save_path, device=args.device
    )
