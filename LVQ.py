import numpy as np

class LVQ:
    def __init__(self, learning_rate=0.01, n_epochs=100, n_prototypes_per_class=1):
        """
        Inisialisasi model LVQ.

        Parameters:
        - learning_rate: Laju pembelajaran (alpha) untuk update prototipe.
        - n_epochs: Jumlah iterasi pelatihan.
        - n_prototypes_per_class: Jumlah prototipe per kelas.
        """
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_prototypes_per_class = n_prototypes_per_class
        self.prototypes = None
        self.labels = None

    def _initialize_prototypes(self, X, y):
        """
        Inisialisasi prototipe secara acak berdasarkan data.

        Parameters:
        - X: Data input (n_samples x n_features).
        - y: Label target (n_samples).
        """
        classes = np.unique(y)
        self.prototypes = []
        self.labels = []

        for c in classes:
            indices = np.where(y == c)[0]
            selected = np.random.choice(indices, self.n_prototypes_per_class, replace=False)
            self.prototypes.append(X[selected])
            self.labels.extend([c] * self.n_prototypes_per_class)

        self.prototypes = np.vstack(self.prototypes)
        self.labels = np.array(self.labels)

    def _update_prototype(self, x, y):
        """
        Update prototipe berdasarkan jarak ke data input.

        Parameters:
        - x: Data input (satu sampel).
        - y: Label target (satu sampel).
        """
        # Hitung jarak Euclidean ke semua prototipe
        distances = np.linalg.norm(self.prototypes - x, axis=1)
        closest_index = np.argmin(distances)

        # Update prototipe jika label cocok atau tidak cocok
        if self.labels[closest_index] == y:
            self.prototypes[closest_index] += self.learning_rate * (x - self.prototypes[closest_index])
        else:
            self.prototypes[closest_index] -= self.learning_rate * (x - self.prototypes[closest_index])

    def fit(self, X, y):
        """
        Melatih model LVQ.

        Parameters:
        - X: Data input (n_samples x n_features).
        - y: Label target (n_samples).
        """
        self._initialize_prototypes(X, y)

        for epoch in range(self.n_epochs):
            for i in range(X.shape[0]):
                self._update_prototype(X[i], y[i])

    def predict(self, X):
        """
        Membuat prediksi berdasarkan prototipe.

        Parameters:
        - X: Data input (n_samples x n_features).

        Returns:
        - Prediksi label untuk setiap sampel.
        """
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.prototypes - x, axis=1)
            closest_index = np.argmin(distances)
            predictions.append(self.labels[closest_index])
        return np.array(predictions)
