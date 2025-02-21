# implementation of Federerated K-means and its comparison with local k-means

## Limitations

Even if Federated k-means converges, this poc suffers from secrutiy issues:
if `k=n_samples`, then it will be possible to extract all points