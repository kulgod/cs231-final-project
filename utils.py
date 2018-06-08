import h5py
import numpy as np

class Dataset(object):
    def __init__(self, X=[], Y=[], name="", batch_size=100, shuffle=False):
        """
        Construct a Dataset object to iterate over data X and labels y
        """

        if len(X) and len(Y):
            assert X.shape[0] == Y.shape[0], 'Got different numbers of data and labels'
            _, H, W, C = X.shape
            with h5py.File(name, 'w') as f:
                f.create_dataset('X', data=X, chunks=(batch_size, H, W, C))
                f.create_dataset('Y', data=Y, chunks=(batch_size, H, W, 1))

        self.f = h5py.File(name, 'r')
        self.X = self.f['X']
        self.Y = self.f['Y']

        self.batch_size, self.shuffle = batch_size, shuffle

    def __iter__(self):
        N, B = self.X.shape[0], self.batch_size
        idxs = np.arange(N)
        if self.shuffle:
            np.random.shuffle(idxs)
        return iter((self.X[i:i+B], self.Y[i:i+B]) for i in range(0, N, B))

    def __del__(self):
        self.f.close()
