#!/usr/bin/python

# Copyright (C) 2014 Allison Chaney

import cPickle, sys
from os.path import join

import onlineldavb
import generalrandom


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description = \
        'Fit LDA to a set of documents with online VB.')

    parser.add_argument('fit_path', type=str, \
        help = 'path to the fit directory')
    parser.add_argument('tmv_path', type=str, \
        help = 'path to tmv source template (e.g. \'../tmv/BasicBrowser\')')

    # parse the arguments
    args = parser.parse_args()
    sys.path.append(args.tmv_path)
    import db


    # load model settings: vocab, K, docgen
    print "loading model settings"
    f = open(join(args.fit_path, 'settings.pickle'))
    vocab, K, docgen = cPickle.load(f)
    f.close()

    # load model itself, the olda object
    print "loading model"
    f = open(join(args.fit_path, 'olda.pickle'))
    olda = cPickle.load(f)
    f.close()

    # Add terms and topics to the DB
    print "initializing db"
    db.init()
    print "adding vocab terms"
    db.add_terms(vocab)
    print "adding",K,"topics"
    db.add_topics(K)


    # write out the final topics to the db
    print "writing out final topics to tmv db"
    for topic in range(len(olda._lambda)):
        topic_terms_array = []
        lambda_sum = sum(olda._lambda[topic])

        for term in range(len(olda._lambda[topic])):
            topic_terms_array.append((term, \
                olda._lambda[topic][term]/lambda_sum))

        db.update_topic_terms(topic, topic_terms_array)


    # do a final pass over all documents
    print "doing a final E step over all documents"
    per_time = dict()
    i = 0
    import time
    s = time.time()
    D = 1850000 #TODO: this should be read in from settings
    for filename, alltxt, title, subtitle in docgen:
        length = 0
        for word in alltxt.split():
            if word in vocab:
                length += 1

        # TODO: this should be done less hackishly
        t = int(filename.split('/')[6])

        if length == 0:
            continue

        db.add_doc(title, subtitle, length, filename, t)

        (gamma, ss) = olda.do_e_step(alltxt)
        if t not in per_time:
            per_time[t] = ss
        else:
            per_time[t] += ss
        db.add_doc_topics(filename, gamma.tolist()[0])
        if i % 100 == 0:
            tn = (time.time() - s) / 3600
            rem = D - i
            time_rem = rem * (tn) / (i+1)
            print "doc %d (%d)" % (i, t), tn, str(time_rem)+'h', \
                str(time_rem/24)+'d'
        i += 1

    # slice up topics by time
    print "calculating time-slice topics"
    for t in per_time:
        per_time[t] += olda._eta
        db.add_time_topics(t, per_time[t])
