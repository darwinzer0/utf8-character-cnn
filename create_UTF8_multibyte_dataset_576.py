#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Functions for creating a text classification dataset out of tab-delimited files
The expected file structure is:
<Class>\t<Text Field>
"""

import argparse
import caffe
import csv
from collections import defaultdict
import h5py
import lmdb
import numpy as np
import os
import PIL.Image
import random
import re
import shutil
import sys
import time
from random import randint

np.set_printoptions(threshold='nan')

# Find the best implementation available
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO

DB_BATCH_SIZE = 1024

FEATURE_LEN = 576 # must have integer square root 12 * 12 * 4 (sqrt of 24)
max_len = 0
#need to remove all tabs from text (make it one column for text)
class UnicodeDictReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, delimiter='\t', quoting=csv.QUOTE_NONE, encoding="utf-8", fieldnames=['class'], restkey='fields', **kwds):
        self.encoding = encoding
        self.reader = csv.DictReader(f, delimiter=delimiter, quoting=quoting, fieldnames=fieldnames, restkey=restkey, **kwds)

    def next(self):
        global max_len
        row = self.reader.next()
        row['fields'][0] = unicode(row['fields'][0], "utf-8")
        row['fields'][0] = row['fields'][0].lower()
        row['fields'][0] = bytearray(row['fields'][0].encode('utf-8'))
        if len(row['fields'][0]) > max_len:
            max_len = len(row['fields'][0])
        return row

    def __iter__(self):
        return self

def _save_image(image, filename):
    # convert from (channels, heights, width) to (height, width)
    image = image[0]
    image = PIL.Image.fromarray(image)
    image.save(filename)

def create_dataset(folder, input_file_name, db_batch_size=None, create_images=False, labels_file=None):
    """
    Creates LMDB database and images (if create_images==True)
    """

    if db_batch_size is None:
        db_batch_size = DB_BATCH_SIZE

    # open output LMDB
    output_db = lmdb.open(folder, map_async=True, max_dbs=0)

    labels = None
    if labels_file is not None:
        labels = map(str.strip,open(labels_file, "r").readlines())
    print "Class labels: %s" % repr(labels)
    if create_images:
        for label in labels:
            os.makedirs(os.path.join(args['output'], label))


    print "Reading input file %s..." % input_file_name
    samples = {}
    batch = []
    with open(input_file_name, "rb") as f:

        global max_len

        reader = UnicodeDictReader(f)
        ii = 0
        for row in reader:
            ii = ii + 1
            if (ii % 10000) == 0:
              print ii
            label = int(row['class']) - 1
            if label not in samples:
                samples[label] = []
            sample = np.ones(FEATURE_LEN) # one by default (i.e. 'other' character)
            # randomize start index
            str_len = 0
            for field in row['fields']:
                str_len += len(field)
            max_start = FEATURE_LEN - str_len
            if max_start < 0:
                max_start = 0
            count = 0

            ## uncomment following lines if you want to randomly set start index for text in input
            #if max_start > 0:
                #count = randint(0,max_start)
            for field in row['fields']:
                for char in field:
                    sample[count] = char
                    count += 1
                    if count >= FEATURE_LEN-1:
                        break
            ##
            class_id = label + 1
            sample = sample.astype('uint8')
            sample = sample[np.newaxis, np.newaxis, ...]
            sample = sample.reshape((1,np.sqrt(FEATURE_LEN),np.sqrt(FEATURE_LEN)))
            if create_images:
                filename = os.path.join(args['output'], labels[label], '%d.png' % ii)
                _save_image(sample, filename)
            datum = caffe.io.array_to_datum(sample, class_id)
            batch.append(('%d_%d' % (ii,class_id), datum))
            if len(batch) >= db_batch_size:
                _write_batch_to_lmdb(output_db, batch)
                batch = []

            samples[label].append(sample)

    output_db.close()

    return

def _write_batch_to_lmdb(db, batch):
    """
    Write a batch of (key,value) to db
    """
    try:
        with db.begin(write=True) as lmdb_txn:
            for key, datum in batch:
                lmdb_txn.put(key, datum.SerializeToString())
    except lmdb.MapFullError:
        # double the map_size
        curr_limit = db.info()['map_size']
        new_limit = curr_limit*2
        print('Doubling LMDB map size to %sMB ...' % (new_limit>>20,))
        try:
            db.set_mapsize(new_limit) # double it
        except AttributeError as e:
            version = tuple(int(x) for x in lmdb.__version__.split('.'))
            if version < (0,87):
                raise Error('py-lmdb is out of date (%s vs 0.87)' % lmdb.__version__)
            else:
                raise e
        # try again
        _write_batch_to_lmdb(db, batch)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Create Dataset tool')

    ### Positional arguments

    parser.add_argument('input', help='Input .csv file')
    parser.add_argument('output', help='Output Folder')
    parser.add_argument('--create-images', action='store_true')
    parser.add_argument('--labels', default=None)

    args = vars(parser.parse_args())

    if os.path.exists(args['output']):
        shutil.rmtree(args['output'])

    os.makedirs(args['output'])

    start_time = time.time()

    create_dataset(args['output'], args['input'], create_images = args['create_images'], labels_file = args['labels'])

    print 'Done after %s seconds' % (time.time() - start_time,)


