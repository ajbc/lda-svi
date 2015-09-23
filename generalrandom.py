# wikirandom.py: Functions for downloading random articles from Wikipedia
#
# Copyright (C) 2010  Matthew D. Hoffman
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

import sys, os, urllib2, re, string, time, threading, fnmatch
from random import randint
from bs4 import BeautifulSoup

class LiveparseDocGen():
    def __init__(self, path):
        print "initializing live parse doc gen with path " + path

        self.doclist = []
        for (root,dirnames,filenames) in os.walk(path):
            for filename in fnmatch.filter(filenames, '*'):
                #print os.path.join(root,filename)
                self.doclist.append(os.path.join(root,filename))

    def get_article(self, id):
        docfile = self.doclist[id]
        xml = open(docfile, 'r')
        soup = BeautifulSoup(xml)
        xml.close()

        # find all the text
        if soup.find("block", {"class":"full_text"}) is None:
            return self.get_random_article()
        fulltext = soup.find("block", {"class":"full_text"})
        paras = fulltext.findAll("p")
        alltxt = ' '.join([p.contents[0] for p in paras])
        alltxt = alltxt.lower()
        alltxt = re.sub(r'-', ' ', alltxt)
        alltxt = re.sub(r'[^a-z ]', '', alltxt)
        alltxt = re.sub(r' +', ' ', alltxt)

        title = soup.find("title")
        title = title.contents[0] if title else ""

        byline = soup.find("byline")
        subtitle = byline.contents[0] if byline and len(byline.contents) != 0 \
            else ""

        return (alltxt, title, subtitle, docfile)

    def get_random_article(self):
        id = randint(0, len(self.doclist) - 1)
        return self.get_article(id)

    def get_random_articles(self, n):
        docs = []
        for i in range(n):
            (doc, title, subtitle, link) = self.get_random_article()
            docs.append(doc)
        return docs

    def getDocCount(self):
        return len(self.doclist)

    def __iter__(self):
        self.current = 0
        return self

    def next(self):
        if self.current >= len(self.doclist):
            raise StopIteration
        else:
            (all, title, subtitle, docfile) = self.get_article(self.current)
            link = self.doclist[self.current]
            self.current += 1
            return (link, all, title, subtitle)

class PreparseDocGen():
    def __init__(self, filename):
        print "initializing preparsed doc gen with file " + filename
        lines = open(filename).readlines()
        self.docs = []
        self.terms = set()
        for line in lines:
            wordids = []
            wordcts = []

            for token in line.split(' ')[1:]:
                tokens = token.split(':')
                wordids.append(int(tokens[0])-1)
                wordcts.append(int(tokens[1]))
                self.terms.add(int(tokens[0]))

            self.docs.append((wordids, wordcts))
        self.D = len(self.docs)

        #''The first, wordids, says what vocabulary tokens are present in
        #each document. wordids[i][j] gives the jth unique token present in
        #document i. (Dont count on these tokens being in any particular
        #order.)
        #The second, wordcts, says how many times each vocabulary token is
        #present. wordcts[i][j] is the number of times that the token given
        #by wordids[i][j] appears in document i.

    def get_random_articles(self, n):
        wordids = []
        wordcts = []
        i = 0
        while (i < n):
            doc = self.docs[randint(0, self.D - 1)]

            # omit short docs from training (to speed things up)
            if len(doc[0]) < 5:
                continue

            wordids.append(doc[0])
            wordcts.append(doc[1])

            i += 1

        return((wordids, wordcts))

    def getDocCount(self):
        return len(self.docs)

    def getTermCount(self):
        return len(self.terms)

    def __iter__(self):
        self.current = 0
        return self

    def next(self):
        if self.current >= len(self.docs):
            raise StopIteration
        else:
            doc = self.docs[self.current]
            self.current += 1
            return(([doc[0]], [doc[1]]))


def get_random_wikipedia_article():
    """
    Downloads a randomly selected Wikipedia article (via
    http://en.wikipedia.org/wiki/Special:Random) and strips out (most
    of) the formatting, links, etc.

    This function is a bit simpler and less robust than the code that
    was used for the experiments in "Online VB for LDA."
    """
    failed = True
    while failed:
        articletitle = None
        failed = False
        try:
            req = urllib2.Request('http://en.wikipedia.org/wiki/Special:Random',
                                  None, { 'User-Agent' : 'x'})
            f = urllib2.urlopen(req)
            while not articletitle:
                line = f.readline()
                result = re.search(r'title="Edit this page" href="/w/index.php\?title=(.*)\&amp;action=edit" /\>', line)
                if (result):
                    articletitle = result.group(1)
                    break
                elif (len(line) < 1):
                    sys.exit(1)

            req = urllib2.Request('http://en.wikipedia.org/w/index.php?title=Special:Export/%s&action=submit' \
                                      % (articletitle),
                                  None, { 'User-Agent' : 'x'})
            f = urllib2.urlopen(req)
            all = f.read()
        except (urllib2.HTTPError, urllib2.URLError):
            print 'oops. there was a failure downloading %s. retrying...' \
                % articletitle
            failed = True
            continue
        print 'downloaded %s. parsing...' % articletitle

        try:
            all = re.search(r'<text.*?>(.*)</text', all, flags=re.DOTALL).group(1)
            all = re.sub(r'\n', ' ', all)
            all = re.sub(r'\{\{.*?\}\}', r'', all)
            all = re.sub(r'\[\[Category:.*', '', all)
            all = re.sub(r'==\s*[Ss]ource\s*==.*', '', all)
            all = re.sub(r'==\s*[Rr]eferences\s*==.*', '', all)
            all = re.sub(r'==\s*[Ee]xternal [Ll]inks\s*==.*', '', all)
            all = re.sub(r'==\s*[Ee]xternal [Ll]inks and [Rr]eferences==\s*', '', all)
            all = re.sub(r'==\s*[Ss]ee [Aa]lso\s*==.*', '', all)
            all = re.sub(r'http://[^\s]*', '', all)
            all = re.sub(r'\[\[Image:.*?\]\]', '', all)
            all = re.sub(r'Image:.*?\|', '', all)
            all = re.sub(r'\[\[.*?\|*([^\|]*?)\]\]', r'\1', all)
            all = re.sub(r'\&lt;.*?&gt;', '', all)
        except:
            # Something went wrong, try again. (This is bad coding practice.)
            print 'oops. there was a failure parsing %s. retrying...' \
                % articletitle
            failed = True
            continue

    return(all, articletitle)

class WikiThread(threading.Thread):
    articles = list()
    articlenames = list()
    lock = threading.Lock()

    def run(self):
        (article, articlename) = get_random_wikipedia_article()
        WikiThread.lock.acquire()
        WikiThread.articles.append(article)
        WikiThread.articlenames.append(articlename)
        WikiThread.lock.release()

def get_random_wikipedia_articles(n):
    """
    Downloads n articles in parallel from Wikipedia and returns lists
    of their names and contents. Much faster than calling
    get_random_wikipedia_article() serially.
    """
    maxthreads = 8
    WikiThread.articles = list()
    WikiThread.articlenames = list()
    wtlist = list()
    for i in range(0, n, maxthreads):
        print 'downloaded %d/%d articles...' % (i, n)
        for j in range(i, min(i+maxthreads, n)):
            wtlist.append(WikiThread())
            wtlist[len(wtlist)-1].start()
        for j in range(i, min(i+maxthreads, n)):
            wtlist[j].join()
    return (WikiThread.articles, WikiThread.articlenames)

if __name__ == '__main__':
    t0 = time.time()

    (articles, articlenames) = get_random_wikipedia_articles(1)
    for i in range(0, len(articles)):
        print articlenames[i]

    t1 = time.time()
    print 'took %f' % (t1 - t0)
