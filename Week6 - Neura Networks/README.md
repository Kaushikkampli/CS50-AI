First, I added 32 filter with 3*3 kernel for feature extraction
Then I tried adding a hidden layer since that seemed to be go to solution to improve accuracy.
I soon realised adding more hidden layers wasnt improving the accuracy as much as I had hoped for.
Adding more hidden layers also increased the computation speed.
Feature extraction seemed to play the most important role and hence I made use of two sequential convolution and pooling groups.
One layer with 64 filters and the other with 128
In the end the model was able to achieve a decent accuracy of 0.93 with reasonable speed.