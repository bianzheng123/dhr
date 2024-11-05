################################################################################################################
# The evaluation code is revised from SPLADE repo: https://github.com/naver/splade/blob/main/src/beir_eval.py


import argparse
import os
import time
import json

from tevatron.datasets.beir.sentence_bert import Retriever, SentenceTransformerModel
from transformers import AutoModelForMaskedLM, AutoTokenizer

from tevatron.arguments import ModelArguments

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir import util, LoggingHandler


def write_answer(username: str, dataset: str, results: dict, topk: int):
    answer_path = f"/home/{username}/Dataset/vector-set-similarity-search/end2end/Result/answer"
    answer_fname = os.path.join(answer_path, f"{dataset}-HNSW-Aggretriever-top{topk}--.tsv")
    print(results)
    with open(answer_fname, 'w') as f:
        for query_id in results.keys():
            result = results[query_id]
            for idx, doc_id in enumerate(result.keys()):
                score = result[doc_id]
                f.write(f"{query_id}\t{doc_id}\t{idx + 1}\t{score}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--model", type=str, default='dhr', help='dhr, agg, dense')
    parser.add_argument("--agg_dim", type=int, default=640, help='for agg model')
    parser.add_argument("--semi_aggregate", action='store_true', help='for agg model')
    parser.add_argument("--skip_mlm", action='store_true', help='for agg model')
    parser.add_argument("--pooling_method", type=str, default='cls', help='for dense model')
    parser.add_argument("--username", type=str, default='bianzheng', help='username')
    args = parser.parse_args()

    model_type_or_dir = args.model_name_or_path
    model_args = ModelArguments
    model_args.model = args.model.lower()
    # agg method
    model_args.agg_dim = args.agg_dim
    model_args.semi_aggregate = args.semi_aggregate
    model_args.skip_mlm = args.skip_mlm
    model_args.pooling_method = args.pooling_method
    # loading model and tokenizer
    model = Retriever(model_type_or_dir, model_args)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_type_or_dir, use_fast=False)
    sentence_transformer = SentenceTransformerModel(model, tokenizer, args.max_length)

    username = args.username
    dataset = args.dataset

    # url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    # out_dir = "dataset/{}".format(dataset)
    # data_path = util.download_and_unzip(url, out_dir)
    # print("---------------------------data_path",data_path)
    data_path = f'dataset/{dataset}/{dataset}'

    #### Provide the data path where nfcorpus has been downloaded and unzipped to the data loader
    # data folder would contain these files:
    # (1) nfcorpus/corpus.jsonl  (format: jsonlines)
    # (2) nfcorpus/queries.jsonl (format: jsonlines)
    # (3) nfcorpus/qrels/test.tsv (format: tsv ("\t"))

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    # from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
    from beir.retrieval.search.dense import HNSWFaissSearch as DRFS
    from beir.retrieval.evaluation import EvaluateRetrieval

    topk = 10

    if username == 'bianzheng':
        M = 16
        hnsw_ef_construction = 30
        hnsw_ef_search_l = [10, 20, 30, 40]
    else:
        assert username == 'zhengbian'
        M = 32
        hnsw_ef_construction = 200
        hnsw_ef_search_l = [10, 20, 30, 40, 60, 80, 120, 160, 300, 500, 1000, 2000, 4000, 8000]

    test_ef_search = 30
    dres = DRFS(sentence_transformer, hnsw_store_n=M, hnsw_ef_construction=hnsw_ef_construction,
                hnsw_ef_search=test_ef_search)

    assert not dres.faiss_index
    start_time = time.time()
    dres.index(corpus)
    print(dres.faiss_index.index.hnsw.efSearch)
    build_index_time = time.time() - start_time
    assert dres.faiss_index
    for hnsw_ef_search in hnsw_ef_search_l:
        dres.faiss_index.index.hnsw.efSearch = hnsw_ef_search
        retriever = EvaluateRetrieval(dres, score_function="dot", k_values=[topk])
        start_time = time.time()
        results = retriever.retrieve(corpus, queries)
        retrieval_time = time.time() - start_time
        write_answer(username, dataset, results, topk)
        ndcg, map_, recall, p = EvaluateRetrieval.evaluate(qrels, results, [topk])
        results2 = EvaluateRetrieval.evaluate_custom(qrels, results, [topk], metric="mrr")
        print(ndcg)
        print(recall)
        print(results2)
        res = {"NDCG@10": ndcg["NDCG@10"],
               "Recall@10": recall["Recall@10"],
               "MRR@10": results2["MRR@10"]}
        print("res for {}:".format(dataset), res, flush=True)

        performance_json = {
            "n_query": len(queries),
            "topk": topk,
            "build_index": {
                "efConstruction": hnsw_ef_construction,
                "time(s)": build_index_time
            },
            "retrieval": {
                "efSearch": hnsw_ef_search
            },
            "search_time": {
                "total_query_time_ms": retrieval_time * 1e3,
                "retrieval_time_average(ms)": retrieval_time / len(queries) * 1e3,
            },
            "search_accuracy": {
                "mrr_mean": results2["MRR@10"],
                "e2e_recall_mean": recall["Recall@10"],
                "ndcg_mean": ndcg["NDCG@10"]
            }
        }

        performance_path = f"/home/{username}/Dataset/vector-set-similarity-search/end2end/Result/performance"
        fname = f"{dataset}-retrieval-HNSW-Aggretriever-top{topk}-M_{M}-efConstruction_{hnsw_ef_construction}-efSearch_{hnsw_ef_search}.json"
        with open(os.path.join(performance_path, fname), "w") as f:
            json.dump(performance_json, f)


if __name__ == "__main__":
    main()
