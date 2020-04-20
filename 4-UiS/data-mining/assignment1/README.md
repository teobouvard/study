In this assignment the goal is to implement the dimensionality reduction technique Principal Component Analysis (PCA) to a very high dimensional data and apply visualization. Note that you are not allowed to use the built-in PCA API provided by the sklearn library. Instead you will be implementing from the scratch.

  - Please DO NOT change the function names in the notebook so the tests can be able to run
  - The tests are based on the variance ratio from reconstruction error computed by the `getVarianceRatio(Z, U, X, K)` function with a few K values. The ratios will be compared to the variance ratio obtained from the built-in PCA implementation in sklearn. The absolute errors should be lower than 5% to pass a test.
