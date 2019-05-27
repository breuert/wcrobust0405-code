import shelve

import core.data_preparation as dprep
import core.evaluation as evaluation
import core.feature_preparation as fprep
import core.prediction as pred
import core.ranking as rank
import core.tfidf_vectorization as tfidf_vec
import core.util as util
import core.training as train

from config.config import path, file

robust_only = True  # train only with robust data


def main():

    print("Start ranking for WCRobust0405")
    # Find intersecting topics
    qrel_files = [file['qrel_times'], file['qrel_robust'], file['qrel_aquaint']]
    topics = util.find_inter_top(qrel_files)

    # Merge qrel files from Robust04 and Robust05
    util.merge_qrels(qrel_files[1:], file['qrel_robust_aquaint'])

    topic_curr = 1
    if topics is not None:

        meta = shelve.open('artifact/feat/train_meta')

        for topic in topics:
            print("Processing topic " + str(topic_curr) + " of " + str(len(topics)))

            n_feat = meta[str(topic)]

            model = train.train(path['train_feat'], topic, n_feat, model_type='logreg-scikit')

            pred.predict(model, file['feat_times_robust0405'], file['score_times_robust0405'])

            rank.rank(file['score_times_robust0405'], topic, path['single_runs'])

            topic_curr += 1

        meta.close()

        complete_run_file = path['complete_run'] + 'wcrobust0405'
        evaluation.merge_single_topics(path['single_runs'], complete_run_file)
        util.clear_path([path['single_runs'], path['train_feat']])

    else:
        print("No intersecting topics")


if __name__ == '__main__':
    main()
