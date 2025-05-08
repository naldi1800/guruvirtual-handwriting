import numpy as np

class RBFNN:
    def __init__(self, n_centers=10, sigma=None):
        self.n_centers = n_centers  # Jumlah neuron di hidden layer
        self.sigma = sigma          # Parameter spread
        self.centers = None        # Pusat RBF
        self.weights = None        # Bobot output

    def _kmeans(self, X, max_iter=100):
        # Inisialisasi pusat secara acak dari data
        np.random.seed(0)
        idx = np.random.choice(len(X), self.n_centers, replace=False)
        centers = X[idx]
        
        for _ in range(max_iter):
            # Hitung jarak ke semua pusat
            distances = np.sqrt(((X[:, np.newaxis] - centers)**2).sum(axis=2))
            
            # Assign ke cluster terdekat
            labels = np.argmin(distances, axis=1)
            
            # Update pusat
            new_centers = np.array([X[labels == i].mean(axis=0) for i in range(self.n_centers)])
            
            # Hentikan jika konvergen
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        
        return centers

    def _calculate_sigma(self, centers):
        # Hitung sigma sebagai rata-rata jarak antar pusat
        dist_sum = 0
        count = 0
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                dist_sum += np.linalg.norm(centers[i] - centers[j])
                count += 1
        return dist_sum / count if count > 0 else 1.0

    def _gaussian_rbf(self, X, center):
        # Fungsi aktivasi Gaussian
        return np.exp(-np.linalg.norm(X - center, axis=1)**2 / (2 * self.sigma**2))

    def fit(self, X, y):
        # 1. Tentukan pusat dengan K-Means
        self.centers = self._kmeans(X)
        
        # 2. Hitung sigma jika tidak ditentukan
        if self.sigma is None:
            self.sigma = self._calculate_sigma(self.centers)
        
        # 3. Hitung matriks aktivasi
        phi = np.zeros((len(X), self.n_centers))
        for i in range(self.n_centers):
            phi[:, i] = self._gaussian_rbf(X, self.centers[i])
        
        # 4. Tambahkan bias dan hitung bobot
        phi = np.hstack([phi, np.ones((len(X), 1))])
        self.weights = np.linalg.pinv(phi.T @ phi) @ phi.T @ y

    def predict(self, X):
        phi = np.zeros((len(X), self.n_centers))
        for i in range(self.n_centers):
            phi[:, i] = self._gaussian_rbf(X, self.centers[i])
        
        phi = np.hstack([phi, np.ones((len(X), 1))])
        return phi @ self.weights