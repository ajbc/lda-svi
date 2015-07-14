#!/usr/bin/python

# olda_general.py: online VB for LDA
#
# Copyright (C) 2014 Allison Chaney
# adapted from Wikipedia demo by Matthew D. Hoffman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cPickle, string, numpy, getopt, sys, random, time, re, pprint
from os.path import join, exists
from os import makedirs
import numpy as np

import onlineldavb
import generalrandom

def print_topics(num_topics, num_terms, vocab, lambdas, anchors, f=None):
    for k in range(0, num_topics):
        lambdak = lambdas[k, :]
        lambdak = lambdak / sum(lambdak)
        lambdak = np.array(lambdak)
        lambdak[lambdak==np.inf] = 0
        lambdak = list(lambdak)
        temp = zip(lambdak, range(0, len(lambdak)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        term = lambda i: vocab[temp[i][1]] + ('*' if len(anchors) > k and \
            vocab[temp[i][1]] in anchors[k] else '')
        terms = ' '.join([term(i) for i in range(num_terms)])
        print '    topic %d: %s' % (k, terms)
        if f:
            f.write('    topic %d: %s\n' % (k, terms))

def fit_olda(parse, doc_path, doc_file, vocab_file, outdir, K, batch_size, \
    iterations, verbose_topics, anchors, tmv_pickle, lemmatize, final_pass, \
    full_doc_topics):
    """
    Analyzes a set of documents using online VB for LDA.
    """
    # instance to generate radom documents
    if parse == "live": # read and parse docs on the fly using vocab
        docgen = generalrandom.LiveparseDocGen(doc_path)
    else: # alternative: preparsed
        docgen = generalrandom.PreparseDocGen(doc_file)

    # The total number of documents in Wikipedia
    D = docgen.getDocCount()
    if iterations == 0:
        iterations = max(D / batch_size, 10)

    # Our vocabulary
    if parse == "live" or verbose_topics:
        vocab = [term.strip() for term in file(vocab_file).readlines()]
        W = len(vocab)
    else:
        W = docgen.getTermCount()
        vocab = ["term " + str(w) for w in range(W)]

    # write out general settings to pickle file for use by TMV later
    if tmv_pickle:
        # save model settings: vocab, K, docgen
        f = open(join(outdir, 'settings.pickle'), 'w+')
        cPickle.dump((vocab, K, docgen, lemmatize), f)
        f.close()

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7, anchors, \
        lem = lemmatize, preparsed = (parse == "preparsed"))
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)

    iteration = 0
    old_perplexity = 1.0 * sys.maxint
    delta_perplexity = 1.0 * sys.maxint
    delta_perplexities = [old_perplexity] * 10
    logfile = open(join(outdir, 'log.out'), 'w+')


    while iteration < iterations and sum(delta_perplexities)/10 > 0.001: # 0.1% change in sample perplexity

        iter_start = time.time()

        # Download some articles
        docset = docgen.get_random_articles(batch_size)

        # Give them to online LDA
        (gamma, bound) = olda.update_lambda(docset)


        # Compute an estimate of held-out perplexity
        if parse == "live":
            (wordids, wordcts) = onlineldavb.parse_doc_list(docset, \
                olda._vocab, lemmatize)
        else:
            (wordids, wordcts) = docset

        # estimate perpexity with the current batch
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        perplexity = numpy.exp(-perwordbound)
        delta_perplexity = abs(old_perplexity - perplexity) / perplexity
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f (%.2f%%)' % \
            (iteration, olda._rhot, perplexity, delta_perplexity * 100)
        logfile.write('%d:  rho_t = %f,  held-out perplexity estimate = %f (%.2f%%)\n' % (iteration, olda._rhot, perplexity, delta_perplexity * 100))
        old_perplexity = perplexity
        delta_perplexities.pop(0)
        delta_perplexities.append(delta_perplexity)


        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 10 == 0):
            numpy.savetxt(join(outdir, 'lambda-%d.dat' % iteration), \
                olda._lambda)
            numpy.savetxt(join(outdir, 'gamma-%d.dat' % iteration), gamma)

            if verbose_topics:
                print_topics(K, 7, vocab, olda._lambda, anchors)

        iteration += 1

    logfile.close()

    if tmv_pickle:
        f = open(join(outdir,'olda.pickle'), 'w+')
        cPickle.dump(olda, f)
        f.close()

    # save final iters
    numpy.savetxt(join(outdir, 'lambda-final.dat'), olda._lambda)
    numpy.savetxt(join(outdir, 'gamma-%d.dat' % iteration), gamma)

    # do a final pass on all documents
    if (final_pass):
        fout = open(join(outdir, "gamma-final.dat"), 'w+')
        if not full_doc_topics:
            fout.write("doc.lda.id\ttopic.id\tscore\n")

        i = 0
        for doc in docgen:
            if parse == 'live': #TODO: the parsers should return same order...
                doc = doc[1]
            (gamma, ss) = olda.do_e_step(doc)
            j = 0
            if not full_doc_topics:
                for g in gamma.tolist()[0]:
                    if g > 0.051:
                        fout.write("%d\t%d\t%f\n" % (i,j,g))
                    j += 1
                i += 1
            else:
                gf = gamma.tolist()[0]
                fout.write(('\t'.join(["%f"]*len(gf))+'\n') % tuple(gf))
        fout.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description = \
        'Fit LDA to a set of documents with online VB.')

    parser.add_argument('--K', metavar='K', type=int, \
        default=100, help = 'number of LDA components, default 100')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, \
        default=1000, help = 'batch size (# of random docs per iteration), default 1000')
    parser.add_argument('--iterations', metavar='iterations', type=int, \
        default=0, help = 'number of iterations; default # doc / batch size')
    parser.add_argument('--vocab', metavar='vocab', type=str, \
        default='', help = 'input vocabulary file')


    # Two methods of obtaining the corpus
    group = parser.add_mutually_exclusive_group(required=True)
    # option A: parse into vocab tokens on the fly
    group.add_argument('--doc_path', metavar='doc_path', type=str, \
        default='', help='a path to a corpus, one doc per file')

    # option B: read pre-processed wordcounts
    group.add_argument('--doc_file', metavar='doc_file', type=str, \
        default='', help='a corpus in a single file, one doc per line')


    # verbosity / printing arguments
    parser.add_argument('--print-topics', dest='print_topics', \
        action='store_true', help='print topic terms every 10 iterations')
    parser.add_argument('--no-print-topics', dest='print_topics', \
        action='store_false', help='default; don\'t print topic terms')
    parser.set_defaults(print_topics=False)

    # output arguments
    parser.add_argument('--out', dest='outdir', type=str, \
        default='', help = 'output directory')
    parser.add_argument('--tmv-pickle', dest='tmv_pickle', \
        action='store_true', help='save pickles for tmv later')
    parser.add_argument('--final-pass', dest='final_pass', \
        default=False, action='store_true', help='do a final pass over all documents')
    parser.add_argument('--full-doc-topics', dest='full_doc_topics', \
        default=False, action='store_true', help='write out every doc topic value; no thresholding')


    # extensions of LDA
    parser.add_argument('--anchors', metavar='anchors', type=str, \
        default='', help = 'anchor words file, one topic per line, terms separated by commas')
    parser.add_argument('--lemmatize', dest='lemmatize', default=False, \
        action='store_true', help='Lemmitize vocabulary for live parse.')


    # parse the arguments
    args = parser.parse_args()
    if args.doc_path != '' and args.vocab is '':
        parser.error("--doc_path requires --vocab.")
    if args.anchors != '' and args.vocab is '':
        parser.error("--anchors requires --vocab.")
    if args.outdir != '' and not exists(args.outdir):
        makedirs(args.outdir)

    if args.print_topics and args.vocab == '':
        raise Exception("cannot print topics without vocabulary; use --vocab to specify")

    # anchor words, if applicable
    anchors = []
    if args.anchors != '':
        anchors = [set([term.strip() for term in line.split(',')]) \
            for line in file(args.anchors).readlines()]
        if len(anchors) > args.K:
            raise Exception("K < # of anchored topics")

    # run the fits
    if args.doc_file == '':
        # option A: live parse
        parse = "live"
        lemmatize = args.lemmatize
    else:
        # option B
        parse = "preparsed"
        if args.lemmatize:
            print "ignoring lemmatize; invalid argument for preparsed docs."
        lemmatize = False # (the vocab is predefined)
    fit_olda(parse, args.doc_path, args.doc_file, args.vocab, args.outdir, \
        args.K, args.batch_size, args.iterations, args.print_topics, \
        anchors, args.tmv_pickle, lemmatize, args.final_pass, \
        args.full_doc_topics)
