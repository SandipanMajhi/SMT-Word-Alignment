# SMT-Word-Alignment

We modified the IBM model 1 by using English and French stemmed words to get consistent word alignment probabilities. We use the same word translation probabilites in Model 2 to have the alignments.

### Requirements :
```
   numpy
   nltk
```
### Results :
AER Comparison - 
```
Name           Iterations  Precision   Recall      AER
DICE (100000)              0.315315    0.482249    0.628486
IBM 1          5           0.499307    0.736686    0.424969
IBM1 + Stem    5           0.570042    0.801775    0.355996
IBM1 + Stem    10          0.596394    0.810651    0.335222
IBM2 + Stem    10          0.699029    0.875740    0.244570
```

Train Runtime Comparison - 

```
Name           Iterations     Time Taken
IBM Model 1    10             1504 seconds
IBM Model 2    10             7982 seconds

```
### Instructions :

There are 3 python programs here (`-h` for usage):

[comment]: <> (- `./align` aligns words.)

[comment]: <> (- `./check-alignments` checks that the entire dataset is aligned, and)

[comment]: <> (  that there are no out-of-bounds alignment points.)

[comment]: <> (- `./score-alignments` computes alignment error rate.)

- `./ibm1_numpy.py` produces alignments using IBM Model 1 optimized by indexing, numpy, stemming and preprocessing of data.

- `./ibm2.py` produces alignments using IBM Model 2. Note that ibm2.py uses functions coded in ibm1_numpy.py and both codes are run when ibm2.py is run.

- `./align` same as ibm2.py but name is different for submission.

The example commands to run ibm1 and ibm2 respectively:
```
   > python ibm1_numpy.py

   > python ibm2.py

   > python align
```
To control iterations we have added an option -i. Example for 10 iterations:
```
   > python ibm2.py -i 10
```
[comment]: <> (The `data` directory contains a fragment of the Canadian Hansards,)

[comment]: <> (aligned by Ulrich Germann:)

[comment]: <> (- `hansards.e` is the English side.)

[comment]: <> (- `hansards.f` is the French side.)

[comment]: <> (- `hansards.a` is the alignment of the first 37 sentences. The )

[comment]: <> (  notation i-j means the word as position i of the French is )

[comment]: <> (  aligned to the word at position j of the English. Notation )

[comment]: <> (  i?j means they are probably aligned. Positions are 0-indexed.)

[comment]: <> (The `final` directory contains alignments of IBM 1 and IBM 2 with 10 iterations )

[comment]: <> (- `alignment_ibm_1_10iter.a` is the IBM 1 output.)

[comment]: <> (- `alignment_ibm_2_10iter.a` is the IBM 2 output.)

[comment]: <> (The `scores` directory contains score outputs of some experiments )

[comment]: <> (- `score1_3iter_all.out` is the IBM 1 output with 3 iterations on entire dataset.)

[comment]: <> (- `score1_5iter_all.out` is the IBM 1 output with 5 iterations on entire dataset.)

[comment]: <> (- `score1_10iter_all.out` is the IBM 1 output with 10 iterations on entire dataset.)

[comment]: <> (- `score2_10iter_all.out` is the IBM 2 output with 10 iterations on entire dataset.)
