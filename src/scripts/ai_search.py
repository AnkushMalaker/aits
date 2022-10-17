import os, sys


source_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(source_path)
from utils import SentencesIndex
from config import DEFAULT_INDEX_DIR
from typing import List, Optional
from argparse import ArgumentParser


def safe_query(sentence_index: SentencesIndex, query: str, top_k: int = 1):
    res_list = sentence_index.extract_query_match_from_index(query, sim_thres=0.1, top_k=top_k)
    if res_list is not None:
        for res in res_list:
            match, similarity, sentence_position = res
            print(f"Match: {match}")
            print(f"Similarity: {similarity}")
    else:
        print("No match found.")


def main(
    index_path: str,
    device: str,
    continous_queries: bool,
    top_k: int,
    query: Optional[str] = None,
):
    S = SentencesIndex.from_save(sentence_index_pkl_path=index_path, query_device=device)
    if query:
        safe_query(S, query, top_k=top_k)
    else:
        query = input("Enter search query: ")
        safe_query(S, query, top_k=top_k)
    if continous_queries:
        while True:
            query = input()
            if query == "quit" or query == "q":
                break
            else:
                safe_query(S, query, top_k=top_k)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--index_path", type=str, default=DEFAULT_INDEX_DIR.as_posix())
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--query", type=str)
    parser.add_argument("--continous_queries", "-c", default=False, action="store_true")
    parser.add_argument("--top_k", "-k", type=int, default=1)
    args = parser.parse_args()

    main(
        index_path=args.index_path,
        device=args.device,
        continous_queries=args.continous_queries,
        query=args.query,
        top_k=args.top_k,
    )
