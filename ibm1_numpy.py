
#!/usr/bin/env python
import optparse
import sys
import time
import numpy as np
import nltk
nltk.download()
from nltk.stem.snowball import EnglishStemmer, FrenchStemmer

e_stemmer = EnglishStemmer()
f_stemmer = FrenchStemmer()


def prepare_index(data):
    """
  Function: preparing the index from sentence pairs
  Input: list of sentence pairs
  Output: indices of each langauge words
  """
    print("preparing index...")
    index_e = {}
    index_f = {}
    i = 0
    j = 0
    for (f, e) in data:
        for ew in e:
            if ew not in index_e.keys():
                index_e[ew] = i
                i += 1
        for fw in f:
            if fw not in index_f.keys():
                index_f[fw] = j
                j += 1
    return index_f, index_e


def preprocess_data(data):
    """
  Function: Converts each word in the dataset to lower case and stemming. Also appends a NULL token to french sentence
    as an english word might not align with any french word in the sentence.
  Input: list of sentence pairs
  Output: list sentence pairs
  """
    print("preprocessing data...")
    sentence_pairs = []
    for (f, e) in data:
        sentence_pairs.append(([f_stemmer.stem(fw.lower()) for fw in f], [e_stemmer.stem(ew.lower()) for ew in e]))
    for (f, e) in sentence_pairs:
        f.append("NULL")
    return sentence_pairs


def convert_to_index(data, index_e=None, index_f=None):
    """
  Function: Converts each word in the dataset to lower case.
  Input: list of sentence pairs, and indices of words of the 2 language
  Output: list sentence pairs but with numbers ie indices instead of words
  """
    print("indexing data...")
    sentence_pairs = []
    for (f, e) in data:
        sentence_pairs.append(([index_f[fw] for fw in f], [index_e[ew] for ew in e]))
    return sentence_pairs


def initialize_trans_probs(index_f, index_e):
    """
  Function: Initializes translation probabilities of each word pair uniformly.
  Input:  A set of word indices for language 1 and language 2 each.
  Output: A numpy array with initialized translation probabilities
  """
    print("initializing trans probs...")
    t_ef = np.full([len(index_f), len(index_e)], 1 / (len(index_f)))
    return t_ef


def run_em_algorithm(iterations, index_f, index_e, sentence_pairs, t_ef):
    """
  Function: The EM Algorithm to calculate translation probabilities of each word pair
    i.e. translation probability of word e given the word f.
  Input:  #iterations to run the algorithm, language 1 index, language 2 index, sentence pairs and translation
    probabilities.
  Output: None. It just updated the translation probabilities received in the input.
  """
    print("EM algo running...")
    for _ in range(iterations):
        # count_ef is a dictionary with key as a tuple of word pairs (e,f) i.e. e given f.
        count_ef = np.zeros(shape=(len(index_f), len(index_e)))
        # total_f is a dictionary with key as word f and value as its count.
        total_f = np.zeros(shape=len(index_f))

        for (f, e) in sentence_pairs:
            s_total_e = np.zeros(len(index_e))
            # CALCULATING THE NORMALIZATION
            # we calculate the marginal probability for each word e_w in sentence e
            for e_w in e:
                for f_w in f:
                    s_total_e[e_w] += t_ef[f_w][e_w]

            # COLLECTING COUNTS OF EACH WORD PAIR AND FRENCH WORD
            for e_w in e:
                for f_w in f:
                    c = t_ef[f_w][e_w] / s_total_e[e_w]
                    count_ef[f_w][e_w] += c
                    total_f[f_w] += c

        # ESTIMATING TRANSLATION PROBABILITIES
        try:
            print("Estimating probs for iteration " + str(_))
            for f_w in range(len(index_f)):
                for e_w in range(len(index_e)):
                    t_ef[f_w][e_w] = count_ef[f_w][e_w] / total_f[f_w]
        except Exception as e:
            print("Error occurred for {}", e)
            return
        print("Iteration " + str(_) + " Done")


def prepare_alignment_file(filename, sentence_pairs, t_ef):
    """
  Function: Prepares the alignment file of the dataset.
  Input:  output filename, sentence pairs, and translation probabilities calculated
  Output: None
  """
    print("Preparing alignment...")
    file = open(filename + ".a", "w+")
    for (f, e) in sentence_pairs:
        for (i, f_i) in enumerate(f):
            if i < len(f) - 1:
                # Assigning the max probable aligned e word to f word.
                max_probable = -1
                j_max_probable = -1
                for (j, e_j) in enumerate(e):
                    if t_ef[f_i][e_j] > max_probable:
                        max_probable = t_ef[f_i][e_j]
                        j_max_probable = j
                file.write("%i-%i " % (i, j_max_probable))
        file.write("\n")

def main():
    """
  We are calculating translation probabilities for translating language f to e and using EM Algorithm for IBM Model 1 for that.
  """
    start_time = time.time()

    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest="train", default="data/hansards",
                         help="Data filename prefix (default=data)")
    optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float",
                         help="Threshold for aligning with Dice's coefficient (default=0.5)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=100000000000, type="int",
                         help="Number of sentences to use for training and alignment")
    optparser.add_option("-i", "--iterations", dest="iterations", default=3, type="int",
                         help="iterations to run both models")
    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (opts.train, opts.french)
    e_data = "%s.%s" % (opts.train, opts.english)

    sys.stderr.write("IBM Model 1...")
    sys.stderr.write("\n")
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][
             :opts.num_sents]

    sentence_pairs = preprocess_data(data=bitext)
    index_f, index_e = prepare_index(data=sentence_pairs)
    sentence_pairs = convert_to_index(data=sentence_pairs, index_e=index_e, index_f=index_f)
    t_ef = initialize_trans_probs(index_f=index_f, index_e=index_e)
    em_start_time = time.time()
    run_em_algorithm(iterations=opts.iterations, index_f=index_f, index_e=index_e, sentence_pairs=sentence_pairs, t_ef=t_ef)
    em_time = (time.time() - em_start_time)
    print('Execution of EM Algo for IBM 1 time in seconds: ' + str(em_time))
    prepare_alignment_file(filename="alignment_ibm1", sentence_pairs=sentence_pairs, t_ef=t_ef)
    execution_time = (time.time() - start_time)
    print('Execution time in seconds: ' + str(execution_time))


if __name__ == "__main__":
    main()