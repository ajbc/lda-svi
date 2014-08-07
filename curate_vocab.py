# TODO: add lemmitization

import argparse, os, fnmatch
from collections import defaultdict
import numpy as np
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description = \
    'curate a vocabulary .')

parser.add_argument('doc_path', metavar='doc_path', type=str, \
    help='a path to a corpus, one doc per file')
parser.add_argument('--out', metavar='out', type=str, \
    default='vocab.dat', help = 'output file')

parser.add_argument('--min_doc_count', metavar='min_doc_count', type=int, \
    default=10, help = 'the minimum # of documents a word must be in')
parser.add_argument('--max_doc_percent', metavar='max_doc_per', type=float, \
    default=30, help = 'the maximum % of documents a word can be in')
parser.add_argument('--min_tfidf', metavar='min_tfidf', type=float, \
    default=np.exp(8), help = 'the minimumm tfidf for a word')
#TODO add log tfidf option (same as above, but in log form), and make them
# mutually exclusive

args = parser.parse_args()


tf = defaultdict(int)
df = defaultdict(int)
dc = 0
for (root,dirnames,filenames) in os.walk(args.doc_path):
    for filename in fnmatch.filter(filenames, '*'):
        docfile = os.path.join(root,filename)
    
        xml = open(docfile, 'r')
        soup = BeautifulSoup(xml) 
        xml.close()

        # find all the text
        if soup.find("block", {"class":"full_text"}) is None:
            continue
        fulltext = soup.find("block", {"class":"full_text"})
        paras = fulltext.findAll("p")
        all = ' '.join([p.contents[0] for p in paras])

        all = all.lower()
        all = re.sub(r'-', ' ', all)
        all = re.sub(r'[^a-z ]', '', all)
        all = re.sub(r' +', ' ', all)
        words = string.split(all)

        found = set()
        for word in words:
            if len(word) < 3:
                continue
            if word not in found:
                df[word] += 1
                found.add(word)
            tf[word] += 1
        dc += 1
 
fout = open(args.out, 'w+')
for word in tf:
    tfidf = tf[word] * np.log(dc * 1.0 / df[word])
    if df[word] > args.min_doc_count and \
        df[word]*1.0 / dc <= args.max_doc_per and \
        tfidf > args.min_tfidf:
        fout.write(word + '\n')
fout.close()
