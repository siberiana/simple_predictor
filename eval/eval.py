""" Utils for ranking evaluation. """
from __future__ import print_function

BESTIND = 1


def evaluate_ranking(predictions, testlabels, testids):
    """ Calculates P@1 and MRR.

    Assumes that there is only one relevant document per query. """
    if len(set([a[0] for a in predictions])) == 1:
        print ('All got equal probabilities. Something is wrong')
        return 0.0, 0.0
    p_at_1 = .0
    mrr = .0
    question_indices = dict()  # which lines correspond to which question
    question_best = dict()  # best answer index for the question
    for_mrr = []
    for i in xrange(len(testids)):
        qid = testids[i].split()[0]
        if qid not in question_indices:
            question_indices[qid] = set()
        question_indices[qid].add(i)
        if testlabels[i] == BESTIND:
            question_best[qid] = i
    without_best = 0
    for qid in question_indices:
        curr_pred = [(i, predictions[i]) for i in sorted(list(question_indices[qid]))]
        best_prob = [(curr_pred[i][0], curr_pred[i][1][BESTIND]) for i in xrange(len(curr_pred))]
        best_prob.sort(key=lambda x: (-x[1], x[0]))
        best_prob = [i[0] for i in best_prob]  # remove probabilities, now only sorted indices
        if qid not in question_best:
            without_best += 1
            print ("no best answer for", qid)
            continue
        best_ranked_as = best_prob.index(question_best[qid]) + 1
        for_mrr.append(best_ranked_as)
        if best_ranked_as == 1:
            p_at_1 += 1
        mrr += (1.0 / float(best_ranked_as))
    if len(question_indices) == without_best:
        p_at_1 = 1.0
        mrr = 1.0
    else:
        p_at_1 /= float(len(question_indices) - without_best)
        mrr /= float(len(question_indices) - without_best)
    return p_at_1, mrr
