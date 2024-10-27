import os
import json
import jsonlines


def proc_collection(username: str, dataset: str):
    rawdata_path = f"/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}/document"
    document_fname = os.path.join(rawdata_path, 'collection.tsv')
    doc_l = []
    with open(document_fname, 'r') as f:
        for line in f:
            if line == '':
                continue
            doc_id, doc = line.split('\t')
            doc_l.append({'_id': doc_id, 'title': '', 'text': doc, 'metadata': {}})

    out_path = f"/home/{username}/dhr/dataset/{dataset}/{dataset}/"
    out_fname = os.path.join(out_path, 'corpus.jsonl')

    with jsonlines.open(out_fname, 'w') as writer:
        writer.write_all(doc_l)


def proc_query(username: str, dataset: str):
    rawdata_path = f"/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}/document"
    query_fname = os.path.join(rawdata_path, 'queries.dev.tsv')
    query_l = []
    with open(query_fname, 'r') as f:
        for line in f:
            if line == '':
                continue
            query_id, query = line.split('\t')
            query_l.append({'_id': query_id, 'text': query, 'metadata': {}})

    out_path = f"/home/{username}/dhr/dataset/{dataset}/{dataset}/queries.jsonl"

    with jsonlines.open(out_path, 'w') as writer:
        writer.write_all(query_l)


def proc_qrel(username: str, dataset: str):
    out_path = f"/home/{username}/dhr/dataset/{dataset}/{dataset}/qrels"
    out_fname = os.path.join(out_path, 'test.tsv')
    os.makedirs(out_path, exist_ok=True)

    rawdata_path = f"/home/{username}/Dataset/vector-set-similarity-search/RawData/{dataset}/document"
    gnd_fname = os.path.join(rawdata_path, 'queries.gnd.jsonl')
    with open(gnd_fname, 'r') as f_in, open(out_fname, 'w') as f_out:
        f_out.write('query-id\tcorpus-id\tscore\n')
        for line in f_in:
            if line == '':
                continue
            j_ins = json.loads(line)

            query_id = j_ins['query_id']
            passage_id_l = j_ins['passage_id']

            for passage_id in passage_id_l:
                f_out.write(f'{query_id}\t{passage_id}\t1\n')


if __name__ == '__main__':
    username = 'bianzheng'
    dataset = 'lotte-500-gnd'

    out_path = f"/home/{username}/dhr/dataset/{dataset}/{dataset}/"
    os.makedirs(out_path, exist_ok=True)
    proc_collection(username, dataset)
    proc_query(username, dataset)
    proc_qrel(username, dataset)
