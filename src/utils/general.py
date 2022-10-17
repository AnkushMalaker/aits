from typing import Union, List, Dict, cast, Tuple
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import torch
from sentence_transformers import SentenceTransformer
from pathlib import Path
import os
from dataclasses import dataclass
import nltk
import textract


def read_filepaths(dir_path: Union[str, Path], recurse=False) -> List[Path]:
    # FIXME : Add recurse functionality?

    dir_path = Path(dir_path)
    file_paths = [fname.resolve() for fname in dir_path.iterdir() if fname.is_file()]
    return file_paths


def file_dir_or_pathlist_to_fid_fpath_dict(
    filedir_path_or_filepath_list: Union[List[Union[str, Path]], Union[str, Path]]
):
    """
    Return fid2fpath dict
    The input can be list of files, list of files and dirs, list of dirs or single string dir
    """
    file_paths = []
    if isinstance(filedir_path_or_filepath_list, list):
        for fp in filedir_path_or_filepath_list:
            if os.path.isfile(fp):
                file_paths.append(fp)
            elif os.path.isdir(fp):
                file_paths.extend(read_filepaths(fp))
    elif isinstance(filedir_path_or_filepath_list, str):
        file_paths = read_filepaths(filedir_path_or_filepath_list)
    else:
        raise TypeError

    fid2fpath: Dict[int, str] = {}
    for f_id, fpath in enumerate(file_paths):
        fid2fpath[f_id] = fpath.as_posix()

    # fpath2fid = {fpath: f_id for (f_id, fpath) in fid2fpath.items()}

    return fid2fpath


def sentence2embedding():
    ...


def paths_to_embedding():
    ...


@dataclass
class SentenceCorpusPosition:
    """
    Represents FILE_ID, SENTENCE_POSIITON
    """

    fileid: int
    sentence_position_in_file: int


class SentenceIndexEntry:
    def __init__(
        self,
        sentence_id: int,
        sentence: str,
        sentence_position_in_corpus: SentenceCorpusPosition,
        sentence_embedding: torch.Tensor,
    ) -> None:
        self.sentence_id = sentence_id
        self.sentence = sentence
        self.sentence_position_in_corpus = sentence_position_in_corpus
        self.sentence_embedding = sentence_embedding

    def __repr__(self) -> str:
        return f"id: {self.sentence_id}\nsentence: {self.sentence}\nembedding: {self.sentence_embedding}"


@dataclass
class QueryResult:
    similarity: float
    sent_id: int


@dataclass
class SentenceIndex:
    fid2fpath: Dict[int, str]
    sentid2sententry: Dict[int, SentenceIndexEntry]


class SentencesIndex:
    def __init__(
        self,
        sentence_index: SentenceIndex,
        embedding_model: Union[SentenceTransformer, str] = "all-MiniLM-L6-v2",
        indexing_device: str = "cuda",
        query_device: str = "cpu",
    ) -> None:
        self.sentence_index = sentence_index
        self.indexing_device = torch.device(indexing_device)
        if isinstance(embedding_model, SentenceTransformer):
            self.sentence_embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.sentence_embedding_model = SentenceTransformer(
                embedding_model, device=indexing_device
            )
        else:
            raise TypeError(
                "Please provide sentence_embedding_model or type str or SentenceTransformer"
            )
        self.query_device = torch.device(query_device)

    def query(self, text: str, top_k=1, sim_thres: float = 0.1) -> List[QueryResult]:
        text_embedding = cast(
            torch.Tensor, self.sentence_embedding_model.encode(text, convert_to_tensor=True)
        )
        text_embedding = text_embedding.to(self.query_device)
        # FIXME : MAybe some vectorization possbile for speed up ?

        top_k_results: List[QueryResult] = []

        for sent_id, sent_index_entry in self.sentence_index.sentid2sententry.items():
            cur_sim = cosine_similarity(
                text_embedding.reshape(1, -1).cpu(),
                sent_index_entry.sentence_embedding.to(self.query_device).reshape(1, -1).cpu(),
            )
            if cur_sim > sim_thres:
                if len(top_k_results) != top_k:
                    top_k_results.append(QueryResult(similarity=cur_sim, sent_id=sent_id))
                else:
                    # check with last element (lowest_sim) if this sentence is better
                    if cur_sim > top_k_results[-1].similarity:
                        top_k_results.pop(-1)
                        top_k_results.append(QueryResult(similarity=cur_sim, sent_id=sent_id))
                top_k_results = sorted(top_k_results, key=lambda x: x.similarity, reverse=True)
        return top_k_results

    def extract_query_match_from_index(
        self, text: str, top_k: int = 1, sim_thres: float = 0.5
    ) -> Union[List[Tuple[str, float, SentenceCorpusPosition]], None]:
        query_result = self.query(text, top_k=top_k, sim_thres=sim_thres)

        if not query_result:
            return None
        query_result = query_result[0 : min(top_k, len(query_result))]
        top_k_results = []
        for res in query_result:
            top_k_results.append(
                [
                    self.sentence_index.sentid2sententry[res.sent_id].sentence,
                    res.similarity,
                    self.sentence_index.sentid2sententry[res.sent_id].sentence_position_in_corpus,
                ]
            )
        return top_k_results

    def get_file_position_from_index(
        self, sentence_position_in_corpus: SentenceCorpusPosition
    ) -> Tuple[str, int]:
        """
        Returns filepath and sentence position in file
        """
        return (
            self.sentence_index.fid2fpath[sentence_position_in_corpus.fileid],
            sentence_position_in_corpus.sentence_position_in_file,
        )

    @staticmethod
    def create_sentence_index(
        fid_fpath_dict,
        tokenizer,
        sentence_embedding_model,
        filter_min_length: int = 0,
        split_on_new_line: bool = True,
    ) -> SentenceIndex:
        encoding = "utf-8"

        sentid2sententry: Dict[int, SentenceIndexEntry] = {}
        sentence_id = 0
        # self.position2sentence =
        for fid, fpath in fid_fpath_dict.items():

            raw_sentences = textract.process(fpath, encoding=encoding)
            raw_sentences = raw_sentences.decode(encoding)
            sentences_parsed: List[str] = tokenizer.tokenize(raw_sentences)
            if split_on_new_line:
                filtered_lines = []
                for sent in sentences_parsed:
                    filtered_lines.extend([s.strip() for s in sent.splitlines() if s.strip() != ""])
                sentences_parsed = filtered_lines
            for i, sent in enumerate(sentences_parsed):
                if len(sent) < filter_min_length:
                    continue
                sent_position = SentenceCorpusPosition(fileid=fid, sentence_position_in_file=i)
                sentence_embedding = torch.tensor(
                    sentence_embedding_model.encode(sent), dtype=torch.float32
                )
                sentence_index_entry = SentenceIndexEntry(
                    sentence_id=sentence_id,
                    sentence=sent,
                    sentence_position_in_corpus=sent_position,
                    sentence_embedding=sentence_embedding,
                )
                sentid2sententry[sentence_id] = sentence_index_entry
                sentence_id += 1
        sentence_index = SentenceIndex(fid_fpath_dict, sentid2sententry=sentid2sententry)

        return sentence_index

    def save_sentence_index(self, save_path: Union[str, Path]):
        with open(save_path, "wb") as f:
            pickle.dump(self.sentence_index, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def from_save(
        cls,
        sentence_index_pkl_path: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        indexing_device: str = "cuda",
        query_device: str = "cpu",
    ):
        with open(sentence_index_pkl_path, "rb") as f:
            sentence_index = pickle.load(f)

        sentence_embedding_model = SentenceTransformer(embedding_model_name, device=indexing_device)
        return cls(
            sentence_index=sentence_index,
            embedding_model=sentence_embedding_model,
            indexing_device=indexing_device,
            query_device=query_device,
        )

    @classmethod
    def from_fid_fpath_dict(
        cls,
        fid2fpath_dict: Dict[int, str],
        embedding_model_name: str = "all-MiniLM-L6-v2",
        indexing_device: str = "cuda",
        query_device: str = "cpu",
        filter_min_length: int = 0,
        split_on_new_line: bool = True,
    ):
        tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
        sentence_embedding_model = SentenceTransformer(embedding_model_name, device=indexing_device)
        sentence_index = SentencesIndex.create_sentence_index(
            fid_fpath_dict=fid2fpath_dict,
            tokenizer=tokenizer,
            sentence_embedding_model=sentence_embedding_model,
            filter_min_length=filter_min_length,
            split_on_new_line=split_on_new_line,
        )
        return cls(
            sentence_index=sentence_index,
            embedding_model=sentence_embedding_model,
            indexing_device=indexing_device,
            query_device=query_device,
        )


def test_sentence_index_and_query():
    TEST_INDEX_DIR = Path("/tmp/test_index.pkl")
    x = SentencesIndex.from_fid_fpath_dict(
        file_dir_or_pathlist_to_fid_fpath_dict("/home/ginger/Projects/ai-text-search/DocsToSearch")
    )
    x.save_sentence_index(TEST_INDEX_DIR.as_posix())
    t = SentencesIndex.from_save(TEST_INDEX_DIR.as_posix())

    query = "This is the best day!"
    result = t.query(query)
    top_result = result[0]
    print(f"Sentence: {query}")
    print(f"Top Similarity: {top_result.similarity}")
    print(f"Similar Sentence: {t.sentence_index[top_result.sent_id].sentence}")

    query = "This is the worst day ever!"
    result = t.query(query)
    top_result = result[0]
    print(f"Sentence: {query}")
    print(f"Top Similarity: {top_result.similarity}")
    print(f"Similar Sentence: {t.sentence_index[top_result.sent_id].sentence}")

    query = "LMFAO I can't deal with these bugs, bye"
    result = t.query(query)
    top_result = result[0]
    print(f"Sentence: {query}")
    print(f"Top Similarity: {top_result.similarity}")
    print(f"Similar Sentence: {t.sentence_index[top_result.sent_id].sentence}")


if __name__ == "__main__":
    test_sentence_index_and_query()
