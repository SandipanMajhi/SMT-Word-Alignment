from ibm1_numpy import *


def ibm_model_1_util(n, iter, prepare_alignment=False, filename="alignment_ibm_1"):
    """
  Function: 1) Runs the IBM Model 1 on the dataset using the parameters. 2) prepares indices of both language for faster
  calculations. 3) preprocesses the dataset.
  Input:  number of sentences, number of iterations, a boolean to decide whether to produce alignments or not, and if so,
    then filename
  Output: translation probabilities from IBM model 1, processed sentence pairs, indices of words of each language
  """
    start_time = time.time()
    f_data = "%s.%s" % ("data/hansards", "f")
    e_data = "%s.%s" % ("data/hansards", "e")
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:n]
    sentence_pairs = preprocess_data(data=bitext)
    index_f, index_e = prepare_index(data=sentence_pairs)
    sentence_pairs = convert_to_index(data=sentence_pairs, index_e=index_e, index_f=index_f)
    t_ef = initialize_trans_probs(index_f=index_f, index_e=index_e)
    run_em_algorithm(iterations=iter, index_f=index_f, index_e=index_e, sentence_pairs=sentence_pairs, t_ef=t_ef)
    execution_time = (time.time() - start_time)
    if prepare_alignment:
        prepare_alignment_file(filename=filename, sentence_pairs=sentence_pairs, t_ef=t_ef)
    print('Execution time in seconds: ' + str(execution_time))
    return t_ef, sentence_pairs, index_f, index_e


def initialize_align_probs(sentence_pairs):
    """
  Function: Initializes alignment probabilities of each position of french word given position of eng and length of the
  two sentences.
  Input:  sentence pairs containing indices
  Output: alignment probabilities, maximum length of a sentence of each language
  """
    print("initializing alignment probs...")
    french_sent_lengths = [len(f) for (f, e) in sentence_pairs]
    # 0 to max(french sentences) + 1 due to null
    max_french_length = max(french_sent_lengths)
    english_sent_lengths = [len(e) for (f, e) in sentence_pairs]
    # 0 to max(eng sentences)
    max_eng_length = max(english_sent_lengths)
    align_probs = np.zeros([max_french_length, max_eng_length, max_eng_length+1, max_french_length+1])

    for (l, m) in zip(french_sent_lengths, english_sent_lengths):
        init_prob = 1/l
        for i in range(l):
            for j in range(m):
                align_probs[i][j][m][l] = init_prob
    return align_probs, max_french_length, max_eng_length


def run_ibm_model2(iterations, t_ef, sentence_pairs, index_f, index_e, max_french_length, max_eng_length, align_probs):
    """
  Function: IBM 2 model algorithm.
  Input:  num of iterations, translation probabilities from IBM 1, sentence pairs containing indices, indices of both
  languages, maximum length of french and english sentences, alignment probabilities which were initialized
  Output: None
  """
    print("Running IBM Model 2...")
    for _ in range(iterations):
        print("Starting iteration " + str(_+1))
        # count_ef is a dictionary with key as a tuple of word pairs (e,f) i.e. e given f.
        count_ef = np.zeros(shape=(len(index_f), len(index_e)))
        # total_f is a dictionary with key as word f and value as its count.
        total_f = np.zeros(shape=len(index_f))

        c_ai_jlm = np.zeros([max_french_length, max_eng_length, max_eng_length+1, max_french_length+1])
        t_jlm = np.zeros([max_eng_length, max_eng_length+1, max_french_length+1])

        for (f, e) in sentence_pairs:
            s_total_e = np.zeros(len(index_e))
            l = len(f)
            m = len(e)
            # normalization
            for j, ew in enumerate(e):
                for i, fw in enumerate(f):
                    s_total_e[ew] += t_ef[fw][ew] * align_probs[i][j][m][l]
            # counts
            for j, ew in enumerate(e):
                for i, fw in enumerate(f):
                    count = t_ef[fw][ew] * align_probs[i][j][m][l] / s_total_e[ew]
                    count_ef[fw][ew] += count
                    total_f[fw] += count
                    c_ai_jlm[i][j][m][l] += count
                    t_jlm[j][m][l] += count

        for f_w in range(len(index_f)):
            for e_w in range(len(index_e)):
                t_ef[f_w][e_w] = count_ef[f_w][e_w] / total_f[f_w]

        for i in range(max_french_length):
            for j in range(max_eng_length):
                for m in range(1, max_eng_length+1):
                    for l in range(1, max_french_length+1):
                        if c_ai_jlm[i][j][m][l] != 0:
                            align_probs[i][j][m][l] = c_ai_jlm[i][j][m][l] / t_jlm[j][m][l]
                        else:
                            align_probs[i][j][m][l] = 0
        print("Done " + str(_+1) +"/"+str(iterations))


def prepare_alignment_file_ibm2(filename, sentence_pairs, t_ef, align_probs):
    """
  Function: Alignment generator for IBM 2 that uses both translation and alignment probabilities
  Input:  filename, sentence pairs consisting indices, translation and alignment probabilities
  Output: None
  """
    print("Preparing alignment for ibm 2...")
    file = open(filename + ".a", "w+")
    for (f, e) in sentence_pairs:
        l = len(f)
        m = len(e)
        for (i, f_i) in enumerate(f):
            if i < len(f) - 1:
                # Assigning the max probable aligned e word to f word.
                max_probable = -1
                j_max_probable = -1
                for (j, e_j) in enumerate(e):
                    value = t_ef[f_i][e_j] * align_probs[i][j][m][l]
                    if value > max_probable:
                        max_probable = value
                        j_max_probable = j
                file.write("%i-%i " % (i, j_max_probable))
        file.write("\n")


def main():
    """
    We are calculating translation probabilities for translating language f to e and using IBM Model 2
    that builds on the IBM model 1 for that.
  """
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

    sys.stderr.write("IBM Model 1...")
    sys.stderr.write("\n")
    em_start_time = time.time()
    t_ef, sentence_pairs, index_f, index_e = ibm_model_1_util(opts.num_sents, opts.iterations, True, filename="final/alignment_ibm_1_10iter")
    em_time = (time.time() - em_start_time)
    print('Execution of EM Algo for IBM-1 time in seconds: ' + str(em_time))
    sys.stderr.write("IBM Model 2...")
    start_time = time.time()
    align_probs, max_french_length, max_eng_length = initialize_align_probs(sentence_pairs)
    run_ibm_model2(opts.iterations, t_ef, sentence_pairs, index_f, index_e, max_french_length, max_eng_length, align_probs)
    prepare_alignment_file_ibm2(filename="final/alignment_ibm_2_10iter", sentence_pairs=sentence_pairs, t_ef=t_ef, align_probs=align_probs)
    execution_time = (time.time() - start_time)
    print('Execution time for IBM-2 in seconds: ' + str(execution_time))


if __name__ == "__main__":
    main()