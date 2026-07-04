import numpy as np


class FuzzyKmeans:
    def __init__(self, k: int, mbship_param: float):

        assert mbship_param > 1, "cannot use fuzzykmeans for mbship_param = " + str(mbship_param)
        self._n_centroid = k
        self._mbship_param = mbship_param
        self._weights = []
        self._centroids = None
        self.U = None
    
    def partial_fit(self, X):
        import numpy as np
        if self._centroids is None:
            # initialize centroids
            idx = np.random.choice(X.shape[0], self._n_centroid, replace=False)
            self._centroids = X[idx]
        self._score = 0
        U = self.update_membership(X, self._centroids)
        self.U = U
        centroids, w, xb_score = self.compute_local_stats(X, U, self._centroids)
        self._centroids = centroids
        self._score = xb_score

        self._weights = w
        
    def predict(self, X):
  
        U = self.update_membership(X, self._centroids)
        labels = np.argmax(U, axis=1)
   
        return labels

    def score(self, X):
        """returns Xie Beni estimator
        XB=\frac{\sum_i\sum_j u_{ij}^m\|x_i-c_j\|^2} {N\min_{p\neq q}\|c_p-c_q\|^2}
        """
        # separation (min distance between centroids)

        um = self.U ** self._mbship_param
        xb_local = 0
        total_points = X.shape[0]
        min_dist = float('inf')
        C = self._centroids.shape[0] 
        for i in range(X.shape[0]):
            for j in range(C):
                dist_sq = np.linalg.norm(X[i] - self._centroids[j]) ** 2
                xb_local += um[i, j] * dist_sq


        min_dist = np.inf

        for i in range(C):
            for j in range(i+1, C):
                d = np.linalg.norm(self._centroids[i] - self._centroids[j])**2
                min_dist = min(min_dist, d)
        
        xb = xb_local / (len(X) * min_dist)
        return xb

    def update_membership(self, data, centroids):
        N = data.shape[0] # nb data
        C = centroids.shape[0]  # nb centroids
        U = np.zeros((N, C))

        
        for i in range(N):
            assigned = False
            for j in range(C):
                d_ij = np.linalg.norm(data[i] - centroids[j])
                if d_ij == 0:
                    U[i, :] = 0
                    U[i, j] = 1
                    assigned = True
                    break
            if assigned:
                continue
            for j in range(C):
                d_ij = np.linalg.norm(data[i] - centroids[j])
                sum_term = 0
                for k in range(C):
                    d_ik = np.linalg.norm(data[i] - centroids[k])
                    if d_ik == 0:
                        d_ik = 1e-6
                    sum_term += (d_ij / d_ik) ** (2 / (self._mbship_param - 1))
    
                U[i, j] = 1 / sum_term
        return U

    def compute_local_stats(self, data, U, centroids):
        C = centroids.shape[0]  # nb centroids
        um = U ** self._mbship_param
    
        # Local centroid
        c_local = (um.T @ data) / np.sum(um.T, axis=1, keepdims=True)
    
        # Weights for aggregation
        weights = np.sum(um, axis=0)
        
        # XB numerator contribution
        xb_local = 0
        for i in range(data.shape[0]):
            for j in range(C):
                dist_sq = np.linalg.norm(data[i] - c_local[j]) ** 2
                xb_local += um[i, j] * dist_sq
    
        return c_local, weights,  xb_local