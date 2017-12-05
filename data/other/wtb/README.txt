This is the Chinese Weibo Treebank dataset used in the following paper:

William Yang Wang, Lingpeng Kong, Kathryn Mazaitis, and William W. Cohen, 
"Dependency Parsing for Weibo: An Efficient Probabilistic Logic Programming Approach", 
in Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), 
short paper, Doha, Qatar, Oct. 25-29, 2014, ACL.

----------------------------------------------

There are three files:

train.conll: the training set.
dev.conll: the development set.
test.conll: the test set.

In each of the files, we provide the CoNLL-style dependency annotation of the Weibo dataset.

The Chinese word segmentations were first produced using the Stanford Chinese Word Segmenter,
and then the two annotators manually examined the segmentations, and corrected 
the mis-segmented words. 

Note that the part-of-speech tags are not perfect: 
they are automatically produced using the Stanford PoS Tagger.

The inter-annotator agreement rate on a randomly selected subset of 373 tokens is 82.31 %.

----------------------------------------------
All the original documents are collected by Ling et al. (ACL 2013). 
The original sources retain the copyright of the data.

Note that there are absolutely no guarantees with this treebank,
and you are welcome to report the errors and bugs of the preliminary version
of this treebank.

You are allowed to use this dataset for research purposes only.
You may re-distribute the treebank, but you must retain this readme file in the re-distribution.

For more question about the dataset, please contact:
William Wang, yww@cs.cmu.edu

v1.0 08/27/2014