import numpy as np

class ExtremeLearningMachine:
    def __init__(self, input_size, hidden_size, activation_function='sigmoid'):
        """
        Constructor untuk Extreme Learning Machine.
        
        Parameter:
        - input_size: Jumlah fitur input.
        - hidden_size: Jumlah neuron di hidden layer.
        - activation_function: Fungsi aktivasi ('sigmoid', 'relu', 'tanh').
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation_function = activation_function
        self.input_weights = None
        self.biases = None
        self.output_weights = None
    
    def _activation(self, x):
        """Pilih fungsi aktivasi berdasarkan parameter."""
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        else:
            raise ValueError("Fungsi Aktivasi tidak ditemukan.")
    
    def fit(self, X, y):
        """
        Melatih model ELM.
        
        Parameter:
        - X: Matriks data input (n_samples x input_size).
        - y: Label target (n_samples x output_size).
        """
        # Inisialisasi bobot input dan bias secara acak
        self.input_weights = np.random.randn(self.input_size, self.hidden_size)
        self.biases = np.random.randn(self.hidden_size)
        
        # Hitung H (output hidden layer)
        H = self._activation(np.dot(X, self.input_weights) + self.biases)
        
        # Hitung output weights (solusi Moore-Penrose)
        self.output_weights = np.dot(np.linalg.pinv(H), y)
    
    def predict(self, X):
        """
        Membuat prediksi menggunakan model ELM.
        
        Parameter:
        - X: Matriks data input (n_samples x input_size).
        
        Output:
        - Prediksi (n_samples x output_size).
        """
        # Hitung H (output hidden layer)
        H = self._activation(np.dot(X, self.input_weights) + self.biases)
        
        # Hitung prediksi
        return np.dot(H, self.output_weights)
