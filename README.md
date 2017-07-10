# ML-tools
A grab bag of random Machine Learning tools written in MATLAB. W.I.P.


#### Label Propagation:
- A MATLAB implementation of the semi-supervised technique, Label Propagation.
- You are advised to read [these slides](https://www.slideshare.net/dav009/label-propagation-semisupervised-learning-with-applications-to-nlp)(besides the paper): 
- Some speed optimizations have been made, see comments in code for details.

#### Self-Learning Least squares:
- This is a vanilla least squares classifier, that uses a one-vs-all approach for multiclass problems.
- The (semi-supervised) self-learning implemented is as follows: Fit on train set, predict labels of test set. From there on, Fit on entire dataset and predict on test set. Repeat until convergence(until labels don't change anymore between iterations).
