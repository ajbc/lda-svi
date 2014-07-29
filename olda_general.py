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
from os.path import join

import onlineldavb
import generalrandom

def print_topics(num_topics, num_terms, vocab, lambdas, f=None):
    for k in range(0, num_topics):
        lambdak = list(lambdas[k, :])
        lambdak = lambdak / sum(lambdak)
        temp = zip(lambdak, range(0, len(lambdak)))
        temp = sorted(temp, key = lambda x: x[0], reverse=True)
        terms = ' '.join([vocab[temp[i][1]] for i in range(num_terms)])
        print '    topic %d: %s' % (k, terms)
        if f:
            f.write('    topic %d: %s\n' % (k, terms))

def fit_olda_liveparse(doc_path, vocab_file, outdir, K, batch_size, iterations):
    """
    Analyzes a set of documents using online VB for LDA.
    """
    
    # instance to get random documents
    docgen = generalrandom.LiveparseDocGen(doc_path)

    # The total number of documents in Wikipedia
    D = docgen.getDocCount()

    # define number of iterations; TODO: add a better convergence check
    if iterations == 0:
        iterations = int(D/batch_size)

    # Our vocabulary
    vocab = [term.strip() for term in file(vocab_file).readlines()]
    W = len(vocab)

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(vocab, K, D, 1./K, 1./K, 1024., 0.7)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)
    for iteration in range(0, iterations):
        # Download some articles
        (docset, articlenames) = \
            docgen.get_random_articles(batch_size)
        # Give them to online LDA
        (gamma, bound) = olda.update_lambda(docset)
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = onlineldavb.parse_doc_list(docset, olda._vocab)
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f' % \
            (iteration, olda._rhot, numpy.exp(-perwordbound))

        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 10 == 0):
            numpy.savetxt(join(outdir, 'lambda-%d.dat' % iteration), \
                olda._lambda)
            numpy.savetxt(join(outdir, 'gamma-%d.dat' % iteration), gamma)
            print_topics(K, 7, vocab, olda._lambda)


def fit_olda_preparse(doc_file, vocab_file, outdir, K, batch_size, iterations):
    """
    Analyzes a set of documents using online VB for LDA.
    """
    print batch_size, " ia the bathc_sze"
    
    # instance to get random documents
    docgen = generalrandom.PreparseDocGen(doc_file)

    # The total number of documents in Wikipedia
    D = docgen.getDocCount()
    vocab = [line.strip() for line in open(args.vocab).readlines()]

    # Initialize the algorithm with alpha=1/K, eta=1/K, tau_0=1024, kappa=0.7
    olda = onlineldavb.OnlineLDA(len(vocab), K, D, 1./K, 1./K, 1024., 0.7)
    # Run until we've seen D documents. (Feel free to interrupt *much*
    # sooner than this.)

    iteration = 0
    old_perplexity = 1.0 * sys.maxint
    delta_perplexity = 1.0 * sys.maxint
    old_perplexities = [old_perplexity] * 10
    logfile = open(join(outdir, 'log.out'), 'w+')
    while (iterations != 0 and iteration < iterations) or \
        sum(old_perplexities)/10 > 0.0001: # 0.1% change in sample perplexity
        # Download some articles
        docset = docgen.get_random_articles(batch_size)
        # Give them to online LDA
        (gamma, bound) = olda.update_lambda(docset)
        # Compute an estimate of held-out perplexity
        (wordids, wordcts) = docset
        perwordbound = bound * len(docset) / (D * sum(map(sum, wordcts)))
        perplexity = numpy.exp(-perwordbound)
        delta_perplexity = abs(old_perplexity - perplexity) / perplexity
        print '%d:  rho_t = %f,  held-out perplexity estimate = %f (%.2f%%)' % \
            (iteration, olda._rhot, perplexity, delta_perplexity * 100)
        logfile.write('%d:  rho_t = %f,  held-out perplexity estimate = %f (%.2f%%)\n' % (iteration, olda._rhot, perplexity, delta_perplexity * 100))
        old_perplexity = perplexity
        old_perplexities.pop(0)
        old_perplexities.append(old_perplexity)


        # Save lambda, the parameters to the variational distributions
        # over topics, and gamma, the parameters to the variational
        # distributions over topic weights for the articles analyzed in
        # the last iteration.
        if (iteration % 10 == 0):
            #TODO: add outdir
            numpy.savetxt(join(outdir, 'lambda-%d.dat' % iteration), \
                olda._lambda)
            numpy.savetxt(join(outdir, 'gamma-%d.dat' % iteration), gamma)
            print_topics(K, 7, vocab, olda._lambda, logfile)
        iteration += 1
    f.close()
    
    # save final iters
    numpy.savetxt(join(outdir, 'lambda-%d.dat' % iteration), olda._lambda)
    numpy.savetxt(join(outdir, 'gamma-%d.dat' % iteration), gamma)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description = \
        'Fit LDA to a set of documents with online VB.')
    
    parser.add_argument('--out', metavar='outdir', type=str, \
        default='', help = 'output directory')
    parser.add_argument('--K', metavar='K', type=int, \
        default=100, help = 'number of LDA components, default 100')
    parser.add_argument('--batch_size', metavar='batch_size', type=int, \
        default=1000, help = 'batch size (# of random docs per iteration), default 1000')
    parser.add_argument('--iterations', metavar='iterations', type=int, \
        default=0, help = 'number of iterations; default # doc / batch size')
    

    # Two methods of obtaining the corpus
    group = parser.add_mutually_exclusive_group(required=True)
    # option A: parse into vocab tokens on the fly
    group.add_argument('--doc_path', metavar='doc_path', type=str, \
        default='', help='a path to a corpus, one doc per file')
    parser.add_argument('--vocab', metavar='vocab', type=str, \
        default='', help = 'input vocabulary file')
    
    # option B: read pre-processed wordcounts
    group.add_argument('--doc_file', metavar='doc_file', type=str, \
        default='', help='a corpus in a single file, one doc per line')

    args = parser.parse_args()
    if args.doc_path != '' and args.vocab is '':
        parser.error("--doc_path requires --vocab.")

    if args.doc_file == '':
        # option A
        fit_olda_liveparse(args.doc_path, args.vocab, args.out, \
            args.K, args.batch_size, args.iterations)
    else:
        # option B
        fit_olda_preparse(args.doc_file, args.vocab, args.out, \
            args.K, args.batch_size, args.iterations)
