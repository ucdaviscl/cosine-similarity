# Wikipedia Data Sets

The included data set contains 137,362 aligned sentences extracted by pairing Simple English Wikipedia with English Wikipedia.  A complete description of the extraction process can be found in "Simple English Wikipedia: A New Simplification Task", William Coster and David Kauchak (2011).  In Proceedings of ACL (short paper).  The data set contains those sentences with a similarity above 0.50.  Higher precision alignments may be obtained by TF-IDF thresholding at higher levels.

Two files are included: wiki.normal and wiki.simple.  Each file contains 137,362 lines and corresponds to a sentence.  The nth line/sentence in wiki.normal corresponds to the nth line/sentence in wiki.simple.  Some minimal tokenization has been done to treat most punctuation characters as separate words/tokens.

For questions regarding the data set set, contact David Kauchak at Pomona College.

Reference: http://www.cs.pomona.edu/~dkauchak/simplification/