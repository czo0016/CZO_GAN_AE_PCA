import numpy as np

class MyPCA():
    
    def __init__(self, num_reduced_dims):
        self.dims = num_reduced_dims


    def fit(self, X):

        #find covariance matrix
        cov =  np.cov(X,rowvar=False)

        #calculate eigenvales and eigenvectors
        eigenvals, eigenvecs = np.linalg.eig(cov)

        #sort the indices of the eignvalues (largest to smallest)
        eigen_indicies = np.flip(np.argsort(eigenvals),axis=0)

        #extract the n highest eigenvalue indices
        n_largest_indicies = eigen_indicies[:self.dims]

        #extract the n eignvectors corresponding to the n largest eigenvalues
        largest_eigenvecs = eigenvecs[:, n_largest_indicies]

        #normalize to create W vector
        self.W = np.array([x / np.linalg.norm(x) for x in largest_eigenvecs.T]).T


    def project(self, x):

        #multiply matrix by PCA matrix
        proj = x @ self.W
        
        return proj
