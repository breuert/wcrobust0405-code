import shelve

import core.data_preparation as dprep
import core.feature_preparation as fprep
import core.tfidf_vectorization as tfidf_vec
import core.util as util
import core.training as train

from config.config import path, file

data_prep = True
robust_only = True  # train only with robust data

paths_to_check = [
    path['times'],
    path['robust'],
    path['union_times_robust'],
    path['train_feat'],
    path['tmp'],
    path['complete_run'],
    path['single_runs'],
    path['tfidf']
]


def main():

    print("Start indexing for WCRobust04")
    # Setup directories
    util.check_path(paths_to_check)
    # Delete old shelve with features, if existent
    # util.delete_shelve(file['feat_times_robust04'])

    if data_prep:
        # Extract single raw text document files from New York Times
        dprep.raw_text_from_times(path['times_compr'], path['tmp_extract'], path['times_raw'])
        dprep.clean_raw_text(path['times_raw'], path['times'])
        util.clear_path([path['times_raw']])

        # Extract single raw text document files from TREC Disks 4 & 5
        dprep.raw_text_from_trec(path['trec45'], path['tmp'], path['robust_raw'])
        dprep.clean_raw_text(path['robust_raw'], path['robust'])
        util.clear_path([path['robust_raw']])

        # Make union corpus
        if not robust_only:
            corpora = [path['times'], path['robust']]
            dprep.unify(path['union_times_robust'], corpora)

    # Generate tfidf-vectorizer
    if robust_only:
        tfidf_vec.dump_tfidf_vectorizer(file['vectorizer_times_robust04'], path['robust'])
    else:
        tfidf_vec.dump_tfidf_vectorizer(file['vectorizer_times_robust04'], path['union_times_robust'])

    # Prepare tfidf-features
    fprep.prepare_corpus_feature(file['vectorizer_times_robust04'], path['times'], file['feat_times_robust04'])

    # Find intersecting topics
    qrel_files = [file['qrel_times'], file['qrel_robust']]
    topics = util.find_inter_top(qrel_files)

    topic_curr = 1
    if topics is not None:

        meta = shelve.open('artifact/feat/train_meta')

        for topic in topics:
            print("Preparing features for topic " + str(topic_curr) + " of " + str(len(topics)))

            if robust_only:
                n_feat = train.prep_train_feat(
                    file['vectorizer_times_robust04'],
                    file['qrel_robust'],
                    topic,
                    path['robust'],
                    path['train_feat'])
            else:
                n_feat = train.prep_train_feat(
                    file['vectorizer_times_robust04'],
                    file['qrel_robust'],
                    topic,
                    path['union_times_robust'],
                    path['train_feat'])

            meta[str(topic)] = n_feat

            topic_curr += 1

        meta.close()

    util.clear_path([path['tmp_extract'],
                     path['tmp'],
                     path['times'],
                     path['robust'],
                     path['union_times_robust']])


if __name__ == '__main__':
    main()
