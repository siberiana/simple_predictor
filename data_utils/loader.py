""" Methods to load data from a file. """


def load(path_to_features, has_aid=False):
    """ This method loads features from a file.

    It assumes that the file contains:
    label, question id, answer id and feature vector,
    separated by spaces or tabs.

    Args:
        path_to_features (str): path to the text file with features
    Returns:
        features (list): list of feature vectors
        labels (list)
    """
    fin = open(path_to_features)
    features = []
    labels = []
    ids = []
    for line in fin:
        tok = line.split()
        qid = tok[1]
        if has_aid:
            aid = tok[2]
            vect = map(float, tok[3:])
            ids.append(qid + ' ' + aid)
        else:
            vect = map(float, tok[2:])
            ids.append(qid)
        label = int(tok[0])
        features.append(vect)
        labels.append(label)
    return features, labels, ids
