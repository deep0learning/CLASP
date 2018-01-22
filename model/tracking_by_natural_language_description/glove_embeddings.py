import logging
import sys

import gflags
import numpy as np

FLAGS = gflags.FLAGS

gflags.DEFINE_string('glove_pretrained_data', '/scratch/data/glove/glove.test.txt',
                     'location of the glove pretrained data file.')


class GloveEmbeddings:
    def __init__(self):
        self.embeddings = {}
        logging.info("Loading embeddings from file " + FLAGS.glove_pretrained_data)
        logging.info("Will take upto several minutes...")
        with open(FLAGS.glove_pretrained_data) as f:
            for line in f:
                split = line.split(' ')
                self.embeddings[split[0]] = np.array([float(i) for i in split[1:]])
        self.vocab_size = len(self.embeddings)
        self.embedding_dimension = self.embedding_size()
        logging.info("Done loading embedding from file.")

    def get_embedding(self, word):
        if word in self.embeddings:
            return self.embeddings[word]
        else:
            return np.zeros(self.embedding_size())

    def embedding_size(self):
        return len(self.embeddings['a'])


if __name__ == '__main__':
    gflags.DEFINE_boolean('verbose', True, 'turn on debug output, local only.')
    FLAGS(sys.argv)
    e = GloveEmbeddings()
    print(e.embeddings['hello'])
    print(e.embedding_dimension)
    print(e.vocab_size)
