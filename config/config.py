path = {
    # path to compressed files from TREC disks 4 & 5 (minus cr)
    # "trec45": "data/compr/trec45/",
    "trec45": "/input/collections/robust04/",
    # path to compressed files from AQUAINT corpus
    # "aquaint_compr": "data/compr/aquaint/",
    "aquaint_compr": "/input/collections/robust05/",
    # path to compressed files from New York Times corpus
    # "times_compr": "data/compr/nyt/",
    "times_compr": "/input/collections/core17/",
    # path to (cleaned) text-docs from Washington Post corpus
    "wapo": "data/clean/wapo/",
    # path to raw text-docs from Washington Post corpus
    "wapo_raw": "data/raw/wapo/",
    # path to (cleaned) text-docs from New York Times corpus
    "times": "data/clean/times/",
    # path to raw text-docs from New York Times corpus
    "times_raw": "data/raw/times/",
    # path to (cleaned) text-docs from Robust04 corpus
    "robust": "data/clean/robust/",
    # path to raw text-docs from Robust04 corpus
    "robust_raw": "data/raw/robust/",
    # path to (cleaned) text-docs from AQUAINT corpus
    "aquaint": "data/clean/aquaint/",
    # path to raw text-docs from AQUAINT corpus
    "aquaint_raw": "data/raw/aquaint/",
    # path to union corpus of Washington Post and Robust04
    "union_wapo_robust": "data/clean/union_wapo_robust/",
    # path to union corpus of Washington Post, AQUAINT and Robust04
    "union_wapo_robust_aquaint": "data/clean/union_wapo_robust_aquaint/",
    # path to union corpus of New York Times and Robust04
    "union_times_robust": "data/raw/union_times_robust/",
    # path to union corpus of New York Times, AQUAINT and Robust04
    "union_times_robust_aquaint": "data/clean/union_times_robust_aquaint/",
    # path to union corpus of Robust04 and AQUAINT
    "union_robust_aquaint": "data/clean/union_robust_aquaint/",
    # path to union corpus of Robust04 and AQUAINT with raw texts
    "union_robust_aquaint_raw": "data/raw/union_robust_aquaint/",
    # path to directory where training features will be written in svm-light format
    "train_feat": "artifact/feat/topic-based/",
    # path to temporal directory; this folder will be deleted after successful execution
    "tmp": "tmp/",
    # path for temporal directory; this folder will be deleted after successful execution
    "tmp_extract": "tmp_extract/",
    # path to directory where resulting complete run will be written
    # "complete_run": "artifact/runs/",
    "complete_run": "/output/",
    # path to directory where resulting single runs will be written
    "single_runs": "artifact/runs/single/",
    # path to directory where scores of a topic run will be written
    "score": "artifact/score/",
    # path to directory where tfidf-vectorizers will be stored
    "tfidf": "artifact/tfidf/"
}

file = {
    # path to json lines file of Washington Post corpus
    "wapo_jl": "data/compr/w/TREC_Washington_Post_collection.v2.jl",
    # path to file where tfidf-vectorizer will be dumped
    "vectorizer_wapo_robust04": "artifact/tfidf/tfidfvectorizer_wapo_robust04.pk",
    # path to file where tfidf-vectorizer will be dumped
    "vectorizer_wapo_robust0405": "artifact/tfidf/tfidfvectorizer_wapo_robust0405.pk",
    # path to file where tfidf-vectorizer will be dumped
    "vectorizer_times_robust04": "artifact/tfidf/tfidfvectorizer_times_robust04.pk",
    # path to file where tfidf-vectorizer will be dumped
    "vectorizer_times_robust0405": "artifact/tfidf/tfidfvectorizer_times_robust0405.pk",
    # path to shelve where tf-idf-features will be dumped
    "feat_wapo_robust04": "artifact/feat/feat_wapo_robust04",
    # path to shelve where tf-idf-features will be dumped
    "feat_wapo_robust0405": "artifact/feat/feat_wapo_robust0405",
    # path to shelve where tf-idf-features will be dumped
    "feat_times_robust04": "artifact/feat/feat_times_robust04",
    # path to shelve where tf-idf-features will be dumped
    "feat_times_robust0405": "artifact/feat/feat_times_robust0405",
    # path to file with scores of topic run
    "score_wapo_robust04": "artifact/score/score_wapo_robust04",
    # path to file with scores of topic run
    "score_wapo_robust0405": "artifact/score/score_wapo_robust0405",
    # path to file with scores of topic run
    "score_times_robust04": "artifact/score/score_times_robust04",
    # path to file with scores of topic run
    "score_times_robust0405": "artifact/score/score_times_robust0405",
    # path to qrel-file of Robust 2004
    "qrel_robust": "qrels/qrels2004.txt",
    # path to qrel-file of Common Core 2018
    "qrel_wapo": "qrels/qrels2018.txt",
    # path to qrel-file of Common Core 2017
    "qrel_times": "qrels/qrels2017.txt",
    # path to qrel-file of Robust 2005
    "qrel_aquaint": "qrels/qrels2005.txt",
    # path to merged qrel file of Robust 2004/05
    "qrel_robust_aquaint": "qrels/qrels0405.txt",
    # path to compiled trec_eval
    "trec_eval": "./trec_eval"
}
