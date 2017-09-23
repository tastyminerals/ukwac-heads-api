#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Simple ukWaC corpus semantic roles API.
Allows you to parse ukWac extracted head files.

ukWaC corpus: http://wacky.sslmit.unibo.it/doku.php?id=corpora

"A large corpus automatically annotated with semantic role information"
http://dgfs2017.uni-saarland.de/wordpress/abstracts/clposter/cl_5_saye.pdf

"An exploration of semantic features in an unsupervised thematic fit evaluation
 framework"
http://www.coli.uni-saarland.de/~asayeed/roledm-journal.pdf


Usage examples:

    from ulwac_api import Corpus
    corpus = Corpus('ukwac_corpus')  # set corpus dir with gzipped files
    corpus.create('ukwac.h5', mode='set')  # generates .pickles and .h5 file

CorpusReader is a subclass of Corpus, so you might prefer:

    from ukwac_api import CorpusReader
    reader = CorpusReader('ukwac_corpus')
    reader.create()  # generated .pickles only! since no .h5 fname is given
    reader.connect('ukwac.h5')  # set the generated .h5 file
    reader.read()  # returns iterator

You can read from corpus files directly without conversion to .h5.

    from ukwac_api import CorpusReader
    reader = CorpusReader('ukwac_corpus')
    reader.read()  # returns iterator

"""

from __future__ import division
import argparse
from itertools import chain, izip_longest
from collections import defaultdict as dd
from datetime import datetime
import gzip
import math
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from nltk.stem.wordnet import WordNetLemmatizer
import xml.etree.cElementTree as cet
import os
import pickle
import re
import sys
import tables


class Corpus:
    """Converts ukWaC head xml file(s) to .h5 file."""
    def __init__(self, corpus_path, filterfile='', algs=[]):
        """
        Args:
            | *corpus_path* (str) -- directory with ukWac head files
            | *filterfile* (str) -- a file of allowed words
            | *algs* (list) -- a list of allowed algorithms

        """
        self.path = corpus_path
        self.wordfilter = filterfile  # list of words to include
        self.algs = set(algs) or set(['malt', 'malt_span', 'linear', 'failed'])
        self._selfcheck()
        self._setup()

    def _setup(self):
        """Setting constants required for xml parsing."""
        self.lem = WordNetLemmatizer()
        # wordnet postags
        self.wn = dd(lambda: u'n')
        self.wn.update({'v': u'v', 'n': u'n', 'j': u'a', 'r': u'r'})
        self.lemmatize = False
        self.h5path = None  # .h5 file path, default None
        self.files = [os.path.join(self.path, p) for p in os.listdir(self.path)
                      if p.endswith(('.xml', '.gz'))]
        # init wordfilter
        if self.wordfilter:
            self.set_wordfilter(self.wordfilter)
        # couple of filtering regex for bad heads
        self.junk = re.compile('|'.join(['&amp;amp;',
                                         '@ord@',
                                         '^[^\w]+.*',
                                         '.*[^\w]+$',
                                         '[0-9*]+',
                                         '.*@[\w]+.[\w]+',
                                         '_+',
                                         '[\w]+/'
                                         ]))
        # alg map
        self.ALGS = {'malt': 1, 'malt_span': 2, 'linear': 3, 'failed': 4}
        # roles map
        self.ROLES = {'a0': 5,
                      'a1': 1,
                      'a2': 4,
                      'a3': 10,
                      'a4': 14,
                      'a5': 28,
                      'am': 25,
                      'am-adv': 12,
                      'am-cau': 8,
                      'am-dir': 18,
                      'am-dis': 6,
                      'am-ext': 24,
                      'am-loc': 2,
                      'am-mnr': 7,
                      'am-mod': 21,
                      'am-neg': 13,
                      'am-pnc': 15,
                      'am-prd': 27,
                      'am-tmp': 3,
                      'c-a1': 26,
                      'c-v': 9,
                      'r-a0': 11,
                      'r-a1': 17,
                      'r-a2': 20,
                      'r-a3': 29,
                      'r-am-cau': 23,
                      'r-am-loc': 16,
                      'r-am-mnr': 19,
                      'r-am-tmp': 22
                      }

    def _selfcheck(self):
        assert isinstance(self.path, str)
        assert os.path.exists(self.path)
        assert os.path.isdir(self.path)
        if self.wordfilter:
            assert isinstance(self.wordfilter, str)
            assert os.path.exists(self.wordfilter)
            assert os.path.isfile(self.wordfilter)
        assert isinstance(self.algs, set)

    def _valid(self, head, pos=None, alg=None, role=None):
        """
        1) Check if a given head is in a word filter if the filter is set.
        2) Check if alg parameter is in the allowed algorithms list. This way
        you can control which results  of  malt, malt_span or linear algorithms
        to include in the output.
        3) Do various "junk" checking.
        4) Check whether xml element is a singleton and should be included
        into the output.

        <Even though we were able to successfully find a phrase head it
        does not mean that the head or its governor can be a valid word.
        It might happen so, that instead of a phrase head we receive a
        chunk of punctuation chars or a gardbled word or something we do
        not want to see in our tensor. For this reason we need to do some
        light "junk" filtering.>

        <TODO:
        We  need to check if a role contains infinitivals like "to live to
        dance". Remove  "to" token  and  reconnect each verb with its
        governor. This is relatively rare case that is why I decided skip
        it for now. Besides, it drags in unecessary comlexity.>

        Args:
            | *head* (str) -- phrase head or a governor token
            | *pos* (str) -- word POS-tag
            | *alg* (str) -- algorithm (strategy) used to retrive the token
              (if the token is a phrase head, ``None`` if it is a governor)
            | *role* (str) -- dependency type

        Returns:
            *bool* -- ``True`` if a given token is good, else ``False``

        """
        # check if word is in filtered list
        if self.wordfilter and head not in self.wordfilter:
            return False
        # heads allowed algs check
        if alg is not None and alg not in self.algs:
            return False
        # heads consistency check
        if ' ' in head:
            return False
        if len(head) > 53:  # max possible length in ukWaC
            return False
        elif len(re.sub(r'\w', '', head)) > 3:
            return False
        elif not re.sub(r'\W', '', head):
            return False
        elif len(head) <= 5:  # filtering short words that end with numbers
            if re.match(r'.*[0-9]+', head):
                return False
        elif len(re.sub(r'[^\.]', '', head)) > 1:  # filter head with > 1 dots
            return False
        elif re.search(r'[;:&]', head):
            return False
        # heads junk check
        if self.junk.match(head):
            return False
        # head singletons check
        if alg == 'failed':
            # singletons allowed postags
            if pos not in {'jj', 'rb', 'vv', 'vb', 'vh'}:
                return False
            elif pos != 'md' and role == 'am-mod':
                return False
            elif role != 'am-neg':
                return False
        return True

    def _lem(self, head, pos):
        """Lemmatize given head."""
        if self.lemmatize:
            return self.lem.lemmatize(head, self.wn[pos[0]])
        else:
            return head

    def create(self, h5name=None, compress=0, lemmatize=False, mode='single'):
        """
        Create three .h5 files: ${h5name}_train.h5, ${h5name}_valid.h5 and
        ${h5name}_test.h5.
        The ukwac head files will be split according to the following
        train/valid/test ratio --> 70/20/10.

        Args:
            | *h5name* (str) -- .h5 file name
            | *compress* (int) -- level of compression [0-9]
            | *lemmatize* (bool) -- use nltk lemmatizer to lemmatize
                govs and their deps (doubles processing time)
            | *mode* (str) -- store mode, 'single' {gov,dep}, 'set' {gov,deps}

                'single' mode stores in .h5
                    (sid, gov, gov pos, arg, arg pos, role, alg)

                'set' mode stores in .h5
                    (sid, gov, gov pos, arg0, arg0 pos, role0, alg0,
                                        arg1, arg1 pos, role1, alg1,
                                        ...)

            WARNING!

            .h5 files are created using "single" mode by default.
            If you attempt to use "set" read mode on such .h5 files,
            it will fail.
            In other words, if you plan to read .h5 files using "single" mode,
            do `create('ukwac_single.h5')`.
            If you plan to read .h5 using "set" or "random_set" mode,
            do `create('ukwac_set.h5', mode='set')`.

        """
        # split files into train/valid/test groups
        total = len(self.files)
        s1 = int(round(total * 0.7))  # train sample size
        s2 = int(round(total * 0.2))  # valid sample size
        groups = [self.files[:s1], self.files[s1:s1+s2], self.files[s1+s2:]]

        # word and postag mappings
        wordfreq = dd(int)
        wmap = dd()  # word mapping from word --> int id
        posmap = dd()

        for name, gr in zip(['train', 'valid', 'test'], groups):
            h5name = name + '_' + h5name if h5name else None
            if h5name:
                print("Creating {0} file...".format(h5name))
            wordfreq, wmap, posmap = self._create(gr, [wordfreq, wmap, posmap],
                                                  h5name, compress, lemmatize,
                                                  mode)
            print('')

        self._pickle('wordfreq.pickle', wordfreq)
        self._pickle('wmap.pickle', wmap)
        self._pickle('posmap.pickle', posmap)

    def _create(self, files, maps, h5name, compress, lemmatize, mode):
        """
        Iteratively parse ukWaC head files and convert them to .h5 file.
        If h5name is None, generate .pickle word and postag mappings only.

        """
        def update_progress():
            progress = round((100 - (left / total * 100)), 1)
            total0 = int(math.floor(total/100))
            left0 = int(math.floor(left/100))
            out = '[{0}] {1}%'.format('#'*(total0-left0)+'-'*(left0), progress)
            elapsed = ' Elapsed: {}'.format(datetime.now() - start_time)
            out = '\r' + out + ' {0}/{1}'.format(total-left, total) + elapsed
            sys.stdout.write(out)
            sys.stdout.flush()
            sys.stdout.write('\n') if total-left == 0 else None

        if lemmatize:
            self.lemmatize = True

        # set vars for progress bar
        total = len(files)
        left = len(files)
        start_time = datetime.now()

        # continue from last time
        wordfreq = maps[0]
        wmap = maps[1]
        posmap = maps[2]

        # setup id number generators
        wit = iter(xrange(int(3e6)))  # approximate vocab size
        pit = iter(xrange(80))  # approximate postag vocab size

        # set pytables EArray
        if h5name is not None:
            h5file = tables.open_file(h5name, mode='w')
            atom = tables.Int32Atom()
            filters = None
            if compress:
                filters = tables.Filters(complevel=compress, complib='blosc',
                                         fletcher32=True)
            if mode == 'single':
                expectedrows = int(4e9)
            elif mode == 'set':
                expectedrows = int(2.022e8)

            earr = h5file.create_earray(h5file.root, 'ukwac', atom, (0,),
                                        'earr', filters=filters,
                                        expectedrows=expectedrows)

            # we add -1 at the beginning of .h5 to delimit 1-st govset
            if mode == 'set':
                earr.append(tuple([-1]))  # adding first set delimiter

        # parse all corpus files
        sid = 0  # sentence id num
        cache = []  # gov,deps accumulator
        seen = False  # keep track of <s> tag
        skip = 0  # number of (gov,deps) to skip in order to set correct sid
        for fname in iter(files):
            fname = fname if fname.endswith('.xml') else gzip.open(fname, 'r')
            xmliter = cet.iterparse(fname, events=('end',))
            for _, elem in xmliter:
                if elem.tag == 'governor':
                    if mode == 'set' and cache and cache[1]:
                        if h5name is not None:
                            # sid, gov, gov pos, (arg, arg pos, role, alg)+
                            earr.append(tuple(cache[0] +
                                              [i for i in chain(*cache[1])] +
                                              [-1]))  # -1 is set delimiter

                    cache = []  # reset
                    head, pos = elem.text.rsplit('/', 2)[:2]

                    if not self._valid(head, pos):
                        continue

                    head = self._lem(head, pos)
                    wordfreq[head] += 1
                    wmap[head] = wmap.get(head) or wit.next()
                    posmap[pos] = posmap.get(pos) or pit.next()

                    cache.append([sid, wmap.get(head), posmap.get(pos)])
                    cache.append([])  # padding for dep, to keep row pair

                elif elem.tag == 'dep' and elem.attrib.get('algorithm'):
                    head, pos = elem.text.rsplit('/', 2)[:2]
                    alg = elem.attrib.get('algorithm')
                    role = elem.attrib.get('type')
                    if cache and self._valid(head, pos, alg, role):
                        if pos in {'prp', 'vbg'}:
                            pos = 'nn'

                        if mode == 'single':
                            head = self._lem(head, pos)
                            wordfreq[head] += 1
                            wmap[head] = wmap.get(head) or wit.next()
                            posmap[pos] = posmap.get(pos) or pit.next()
                            cache[-1] = [wmap.get(head), posmap.get(pos),
                                         self.ROLES[role], self.ALGS[alg], sid]
                            if h5name is not None:
                                # sid, gov, gov pos, arg, arg pos, role, alg
                                earr.append(tuple(chain(*cache)))  # flatten
                            continue

                        elif mode == 'set':
                            head = self._lem(head, pos)
                            wordfreq[head] += 1
                            wmap[head] = wmap.get(head) or wit.next()
                            posmap[pos] = posmap.get(pos) or pit.next()
                            cache[-1].append((wmap.get(head),
                                              posmap.get(pos),
                                              self.ROLES[role],
                                              self.ALGS[alg]))

                # single 'end' event shifts <s> index, we need to adjust
                if elem.tag == 's' and not seen:
                    seen = True
                if seen and elem.tag != 'text':
                    skip += 1
                    if skip > 1:
                        seen = False
                        skip = 0
                        sid += 1

                elem.clear()
            del xmliter
            fname.close() if isinstance(fname, gzip.GzipFile) else None
            left -= 1
            update_progress()

        h5file.close() if h5name is not None else None
        return wordfreq, wmap, posmap

    def _pickle(self, fname, obj):
        with open(fname, 'wb') as f:
            pickle.dump(obj, f)

    def _load_pickle(self, fname):
        try:
            with open(fname, 'rb') as f:
                data = pickle.load(f)
        except (IOError, OSError) as ex:
            print 'ERROR! Pickle could not be loaded!'
            print ex
            data = {}
        return data

    def clean_up(self):
        """WARNING! All *.h5, *.pickle files in current dir will be removed"""
        files = [fname for fname in os.listdir(os.getcwd())
                 if fname.endswith(('.h5', '.pickle'))]
        for fname in files:
            try:
                os.remove(fname)
            except OSError as ex:
                print 'ERROR! Can not remove "%s"'.format(fname)
                print ex

    def set_wordfilter(self, wfilter):
        """Word filter setter."""
        with open(self.wordfilter, 'r') as f:
            fdata = f.read().strip().splitlines()
        self.wordfilter = frozenset(fdata)

    def generate_freqlist(self, wordfreq='wordfreq.pickle', size=50000):
        """
        Generate and write a list of most frequent words. You can then use this
        list as a word filter to restrict the words included into the output.

        Args:
            | *wordfreq* (pickle) -- a word frequency dict
            | *size* (int) -- number of words in the list

        """
        wordfreq = self._load_pickle(wordfreq)
        if not wordfreq:
            sys.exit(1)
        try:
            os.remove('wordfreq.txt')
        except OSError:
            pass
        sorted_freqs = sorted(wordfreq, key=wordfreq.get, reverse=True)[:size]
        with open('wordfreq.txt', 'a') as f:
            for w in sorted_freqs:
                f.write(w)
                f.write('\n')
        print 'Successfully generated "wordfreq.txt", size:', size


class CorpusReader(Corpus):
    """Reads ukWaC head xml file(s) directly or from .h5 file."""
    def __init__(self, corpus='.', filterfile='', algs=[]):
        """
        Args:
            | *corpus_path* (str) -- directory with ukWac head files
            | *filterfile* (str) -- a file of allowed words
            | *algs* (list) -- a list of allowed algorithms

        """
        self.path = corpus  # directory with gzipped ukWaC head files
        self.wordfilter = filterfile  # list of words to include
        self.algs = set(algs) or set(['malt', 'malt_span', 'linear', 'failed'])
        self._selfcheck()
        self._setup()

    def connect(self, h5file):
        """Set previously generated .h5 file and use it for reading.
        Args:
            | *h5file* (str) -- .h5 file name

        """
        self.h5path = h5file

    def disconnect(self):
        """Unset previously generated .h5 file. CorpusReader will switch into
        direct reading mode."""
        self.h5path = None

    def _read_from_xml(self, mode, delex):
        """Read the corpus directly by iterating through its files and return
        a corpus iterator. If you already generated .h5 file, load *.pickle
        maps to delexicalize (if True) the output, else return xml strings.

        <Make sure self.path contains ukwac files and set correctly upon
        class instantiation!>

        Args:
            | *mode* (str) -- reading mode 'set' or 'single'
            | *delex* (bool) -- use delexicalization if True

        """
        # load *.pickle maps
        # wordfreq = self._load_pickle('wordfreq.pickle') if delex else {}
        wmap = self._load_pickle('wmap.pickle') if delex else {}
        posmap = self._load_pickle('posmap.pickle') if delex else {}

        # parse all corpus files
        sid = 0  # sentence id num
        cache = []  # gov,deps accumulator
        seen = False  # keep track of <s> tag
        skip = 0  # number of (gov,deps) to skip in order to set correct sid
        for fname in iter(self.files):
            fname = fname if fname.endswith('.xml') else gzip.open(fname, 'r')
            xmliter = cet.iterparse(fname, events=('end',))

            for _, elem in xmliter:
                if elem.tag == 'governor':
                    if mode == 'set' and cache and cache[1]:
                        yield (tuple(cache[0]), tuple(cache[1]))

                    cache = []  # reset
                    head, pos = elem.text.rsplit('/', 2)[:2]
                    if not self._valid(head, pos):
                        continue
                    head = self._lem(head, pos)
                    cache.append([sid, wmap.get(head) or head,
                                  posmap.get(pos) or pos])
                    cache.append([])  # padding for dep, to keep row pair

                elif elem.tag == 'dep' and elem.attrib.get('algorithm'):
                    head, pos = elem.text.rsplit('/', 2)[:2]
                    alg = elem.attrib.get('algorithm')
                    role = elem.attrib.get('type')
                    if cache and self._valid(head, pos, alg, role):
                        if pos in {'prp', 'vbg'}:
                            pos = 'nn'

                        if mode == 'single':
                            head = self._lem(head, pos)
                            cache[-1] = (wmap.get(head) or head,
                                         posmap.get(pos) or pos,
                                         self.ROLES[role],
                                         self.ALGS[alg])
                            yield tuple(chain(*cache))
                            # continue

                        elif mode == 'set':
                            head = self._lem(head, pos)
                            cache[-1].append((wmap.get(head) or head,
                                              posmap.get(pos) or pos,
                                              self.ROLES[role],
                                              self.ALGS[alg]))

                # single 'end' event shifts <s> index, we need to adjust
                if elem.tag == 's' and not seen:
                    seen = True
                if seen and elem.tag != 'text':
                    skip += 1
                    if skip > 1:
                        seen = False
                        skip = 0
                        sid += 1

                elem.clear()
            del xmliter
            fname.close() if isinstance(fname, gzip.GzipFile) else None

    def _read_from_h5(self, mode):
        """Read from previously generated .h5 file.

        Args:
            mode (str) -- reading mode: "single", "set", "random_set"

            WARNING!

            .h5 file is created using "single" mode by default.
            If you attempt to use "set" read mode on such .h5 file,
            this function will crash.
            In other words, if you plan to read .h5 file using "single" mode,
            do create('ukwac_single.h5').
            If you plan to read .h5 using "set" or "random_set" mode,
            do create('ukwac_set.h5', mode='set').

        """
        h5 = tables.open_file(self.h5path, mode='r')
        ukwac = h5.root.ukwac

        if mode == "single":
            chunk = [iter(ukwac)] * 7  # chunk size is constant
            for row in izip_longest(fillvalue=None, *chunk):
                yield row

        elif mode == "set":
            cache = []
            # in 'set' mode chunk size is not constant
            itukwac = iter(ukwac)
            itukwac.next()  # we skip the first -1
            for elem in itukwac:
                if elem == -1:
                    yield tuple(cache)
                    cache = []
                else:
                    cache.append(elem)

        elif mode == "random_set":
            sids = set()
            while len(sids) < 202193377:  # number of govsets in ukwac
                start = random.randint(0, int(2.022e8))  # "set" earray size
                end = start + 170  # extracted from ukwac
                cache = []
                begin = False
                first = False
                for i in ukwac.iterrows(start, end):
                    if i == -1 and not begin:
                        begin = True
                        first = True

                    elif i != -1 and begin:
                        # check sent id if we already seen it
                        if first and i in sids:
                            begin = False
                            first = False
                            continue

                        elif first and i not in sids:
                            cache.append(i)
                            first = False
                            continue

                        cache.append(i)

                    elif i == -1 and begin:
                        begin = False
                        sids.add(cache[0])
                        yield tuple(cache)
                        cache = []

        h5.close()

    def _extract_elems(self, mode, delex, extract):
        """
        Read and extract given xml elements from corpus.

        Args:
            | *mode* (str) -- reading mode, "set"/"single"
            | *delex* (bool) -- apply delexicalization (words -> digits)
            | *extract* (list) -- extract only the elements of attributes from
                the provided list.

        """
        # load *.pickle maps
        # wordfreq = self._load_pickle('wordfreq.pickle') if delex else {}
        wmap = self._load_pickle('wmap.pickle') if delex else {}
        posmap = self._load_pickle('posmap.pickle') if delex else {}

        elems = frozenset(extract)
        attrs = dd()

        # parse all corpus files
        sid = 0  # sentence id num
        cache = []  # gov,deps accumulator
        seen = False  # keep track of <s> tag
        skip = 0  # number of (gov,deps) to skip in order to set correct sid
        for fname in iter(self.files):
            fname = fname if fname.endswith('.xml') else gzip.open(fname, 'r')
            xmliter = cet.iterparse(fname, events=('end',))

            for _, elem in xmliter:
                if elem.tag == 'governor':
                    if mode == 'set' and cache and cache[1]:
                        yield (tuple(cache[0]), tuple(cache[1]))

                    cache = []  # reset
                    head, pos = elem.text.rsplit('/', 2)[:2]
                    if not self._valid(head, pos):
                        continue
                    head = self._lem(head, pos)
                    cache.append([sid, wmap.get(head) or head,
                                  posmap.get(pos) or pos])
                    cache.append([])  # padding for dep, to keep row pair

                elif elem.tag == 'dep' and elem.attrib.get('algorithm'):
                    head, pos = elem.text.rsplit('/', 2)[:2]
                    alg = elem.attrib.get('algorithm')
                    role = elem.attrib.get('type')
                    if cache and self._valid(head, pos, alg, role):
                        if pos in {'prp', 'vbg'}:
                            pos = 'nn'

                        # collecting extract attributes
                        if elems is not None:
                            attrs = dd()
                            for el in elems:
                                attrs[el] = tuple(tuple(e.rsplit('/', 2))
                                                  for e in
                                                  elem.attrib.get(el).split())

                        if mode == 'single':
                            head = self._lem(head, pos)
                            cache[-1] = (wmap.get(head) or head,
                                         posmap.get(pos) or pos,
                                         self.ROLES[role],
                                         self.ALGS[alg],
                                         attrs)
                            yield tuple(chain(*cache))

                        elif mode == 'set':
                            head = self._lem(head, pos)
                            cache[-1].append((wmap.get(head) or head,
                                              posmap.get(pos) or pos,
                                              self.ROLES[role],
                                              self.ALGS[alg],
                                              attrs))

                # single 'end' event shifts <s> index, we need to adjust
                if elem.tag == 's' and not seen:
                    seen = True
                if seen and elem.tag != 'text':
                    skip += 1
                    if skip > 1:
                        seen = False
                        skip = 0
                        sid += 1

                elem.clear()
            del xmliter
            fname.close() if isinstance(fname, gzip.GzipFile) else None

    def read(self, mode='set', delex=False, extract=[], lemmatize=False):
        """
        Read the corpus using one of the given modes.
        Reading modes: 'single', 'set', 'random_set'
        'single' mode returns a tuple of a governor and its dependant argument:

        ('send id', 'gov', 'gov postag', 'arg', 'arg postag', 'role', 'alg')


        'set' mode returns a tuple of tuples of a governor and all its
        dependant arguments found within one sentence:

        ('sent id', 'governor', 'gov postag',
            (('arg', 'arg postag', 'role', 'alg'), (...), (...))
        )

        'random_set' mode (.h5 only) returns a randomly (without replacement)
        chosen part of the .h5 file, looks for a valid chunk of gov and its
        args and returns it.

        Delexicalization (here) is the process of replacing words with ints.
        If h5 corpus file is connected, read from it. Keep in mind that mode
        and delexicalization won't work when reading from h5. Reading from h5
        is intended to be fast and its elements are already delexicalized.

        If extract list of additional xml attributes is provided, read and
        extract these attributes from the corpus and return together with the
        main tuple. The attributes will be collected in a dict and supplied as
        the last element of the output tuple.
        Mode will be applied accordingly if set.
        Delexicalization will be applied as well if delexicalize is set and
        .pickle files exist.

        If lemmatize is set, use nltk lemmatizer on govs and deps. The "source"
        and "text" attributes are coming from malt parser and should be
        lemmatized already.

        Args:
            | *mode* (str) -- reading mode, "set"/"single"
            | *delex* (bool) -- apply delexicalization (words -> digits)
            | *extract* (list) -- extract only the elements of attributes from
                the provided list. Xml head files contain 'source' and 'text'
                attributes whose values can be extracted in addition to already
                existing ones.
            | *lemmatize* (bool) -- use nltk lemmatizer to lemmatize
                govs and their deps (doubles processing time)

        Returns:
            | *iterator* -- corpus line iterator

        """
        if lemmatize:
            self.lemmatize = True

        if mode != 'random_set':
            if self.h5path is None and extract is None:
                return self._read_from_xml(mode, delex)
            elif self.h5path is None and extract is not None:
                return self._extract_elems(mode, delex, extract)
        return self._read_from_h5(mode)

    # debug only
    def decode(self, batch, wmap):
        """Decode batch ids into str"""
        selor = dict([(v, k) for k, v in self.ROLES.items()])
        pamw = dict([v, k] for k, v in wmap.items())
        return dict([(selor.get(k, 'UNK_ROLE'), pamw.get(v))
                    for k, v in batch.items()])

    def get_minibatch(self, filtered_roles, batch_size=1, random=False, neg=1):
        """
        Generate k-noise samples for target role + 1 positive sample from data.
        Noise and positive samples share inputs.

        TODO: implement random reading

        Args:
            filtered_roles (set) -- roles included into batch
            batch_size (int) -- batch size
            random (bool) -- randomly pick data from .h5 file
            neg (int) -- number of negative samples

        """
        # wmap.pickle is generated by self.create() function
        assert os.path.isfile('wmap.pickle'), '"wmap.pickle" is not found!'
        with open('wmap.pickle', 'rb') as f:
            wmap = pickle.load(f)

        # unk_word_id, unk_role_id, nn_missing_word_id, nn_unk_word_id
        unk_word_id = sorted(wmap.values())[-1] + 1
        nn_missing_word_id = unk_word_id
        # nn_unk_word_id = unk_word_id + 1
        unk_role_id = sorted(self.ROLES.values())[-1] + 1

        # get unigram counts
        # n_words = len(wmap)  # used for "unigram_counts"
        n_roles = len(self.ROLES)
        # unigram_counts = dict((r, [0]*n_words) for r in xrange(n_roles+1))

        v_id = 0

        assert type(filtered_roles) == set, '"filtered_roles" arg must be of set type!'
        filtered_roles = filtered_roles ^ {unk_role_id}

        # only set and random_set modes are available for minibatch
        mode = 'set' if not random else 'random_set'
        for govset in self.read(mode=mode):
            # {role_id: word_id}
            batch = dict([(self.ROLES.get(i, unk_role_id), unk_word_id)
                     for i in filtered_roles])
            # govset=(sent_id, gov, gov_pos, arg0, arg0_pos, arg0_role, alg0,)
            govid = govset[1]
            # extract only [arg0, arg0_role, arg1, arg1_role, ...] ids
            ids = [govset[3:][i] for i in xrange(0, len(govset[3:]), 2)]
            # iterate over id pairs, fill batch with matching arg,role pairs
            for arg, role in izip_longest(*[iter(ids)] * 2):
                if batch.get(role):
                    batch[role] = arg
            batch[v_id] = govid

            # print batch
            # print self.decode(batch,wmap)

            # rest is taken from batcher.py

            non_missing_inputs = [n for n, v in enumerate(batch.values())
                                  if v != nn_missing_word_id]
            roles, words = map(list, zip(*batch.items()))
            x_w_i, x_r_i, y_w_i, y_r_i = [], [], [], []
            n_total_samples, n_neg_samples = 0, 0

            # Generate samples for each given (non-missing-word) role-word pair
            for role_idx in non_missing_inputs:
                # Positive sample

                # Remove current role-word pair from context ...
                input_words = words[:]
                input_roles = roles[:]
                del input_words[role_idx]
                del input_roles[role_idx]

                # ... and set it as target
                target_word = words[role_idx]
                target_role = roles[role_idx]

                target_words = [target_word]
                target_roles = [target_role]

                x_w_i.append(input_words)
                x_r_i.append(input_roles)

                n_total_samples += 1

                # generate k neg samples corrupting one non missing input role
                for _ in xrange(neg):
                    noise_role = target_role
                    noise_word = np.random.randint(nn_missing_word_id)
                    # noise_word = np.random.choice(word_ids, p=unigram_counts[noise_role])
                    target_words.append(noise_word)
                    target_roles.append(noise_role)
                    n_neg_samples += 1
                    n_total_samples += 1

                y_w_i.append(target_words)
                y_r_i.append(target_roles)

                if len(x_w_i) >= batch_size:
                    yield (
                        np.asarray(x_w_i, dtype=np.int32),
                        np.asarray(x_r_i, dtype=np.int32),
                        np.asarray(y_w_i, dtype=np.int32),
                        np.asarray(y_r_i, dtype=np.int32),
                        np.asarray(to_categorical(y_w_i, nn_missing_word_id+2),
                                   dtype=np.int32),
                        # added +2 otherwise fails
                        np.asarray(to_categorical(y_r_i, n_roles+2),
                                   dtype=np.int32),
                        n_total_samples,
                        n_neg_samples
                        )

                    x_w_i, x_r_i, y_w_i, y_r_i = [], [], [], []
                    n_total_samples, n_neg_samples = 0, 0


def main():
    corpus = Corpus(args.dir, args.filter, args.algs)
    if args.out and os.path.isdir(args.out):
        h5file = os.path.join(args.out, 'ukwac.h5')
    else:
        h5file = args.out
    corpus.create(h5file, args.compress, args.lemmatize)


if __name__ == "__main__":
    prs = argparse.ArgumentParser(description="""Convert annotated ukWaC\
    corpus to .h5 format.""")
    prs.add_argument('-d', '--dir', required=True,
                     help='Specify the directory where annotated ukWaC corpus \
                     files are located.')
    prs.add_argument('-o', '--out', default=None, required=False,
                     help='Specify the output file name. If not specified do \
                     not create .h5 file and generate only .pickle dicts.')
    prs.add_argument('-filter', '--filter', default=[], required=False,
                     help='Specify the word-per-line filter. Words not in the \
                     file will be skipped.')
    prs.add_argument('-alg', '--algs', required=False, default=[], nargs='*',
                     help='Specify the name of the head extraction algorithm \
                     whose results will be included in the output \
                     (default malt, malt_span, linear, failed).')
    prs.add_argument('-lemmatize', '--lemmatize', required=False,
                     help='Apply lemmatization for govs and deps.',
                     action='store_true')
    prs.add_argument('-c', '--compress', default=0, required=False, type=int,
                     help='Specify the level of compression [0-9] for h5 file\
                     (default 0, no compression).')
    args = prs.parse_args()
    main()
