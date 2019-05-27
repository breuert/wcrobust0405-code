import shelve

import core.evaluation as evaluation
import core.prediction as pred
import core.ranking as rank
import core.util as util
import core.training as train

from config.config import path, file

robust_only = True  # train only with robust data


def main():

    print("Start ranking for WCRobust04")
    # Find intersecting topics
    qrel_files = [file['qrel_times'], file['qrel_robust']]
    topics = util.find_inter_top(qrel_files)

    topic_curr = 1
    if topics is not None:

        meta = shelve.open('artifact/feat/train_meta')

        for topic in topics:
            print("Processing topic " + str(topic_curr) + " of " + str(len(topics)))

            n_feat = meta[str(topic)]

            model = train.train(path['train_feat'], topic, n_feat, model_type='logreg-scikit')

            pred.predict(model, file['feat_times_robust04'], file['score_times_robust04'])

            rank.rank(file['score_times_robust04'], topic, path['single_runs'])

            topic_curr += 1

        meta.close()

        complete_run_file = path['complete_run'] + 'wcrobust04'
        evaluation.merge_single_topics(path['single_runs'], complete_run_file)
        util.clear_path([path['single_runs'], path['train_feat']])

    else:
        print("No intersecting topics")


if __name__ == '__main__':
    main()
