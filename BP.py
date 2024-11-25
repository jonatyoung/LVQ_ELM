import numpy as np

class BackpropagationNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        """
        Inisialisasi jaringan saraf tiruan.

        Parameters:
        - input_size: Jumlah neuron di layer input.
        - hidden_size: Jumlah neuron di hidden layer.
        - output_size: Jumlah neuron di layer output.
        - learning_rate: Laju pembelajaran untuk update bobot.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inisialisasi bobot dan bias secara acak
        self.weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5
        self.bias_hidden = np.random.rand(hidden_size) - 0.5
        self.weights_hidden_output = np.random.rand(hidden_size, output_size) - 0.5
        self.bias_output = np.random.rand(output_size) - 0.5

    def sigmoid(x):
        # Fungsi aktivasi sigmoid.
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(x):
        """Turunan dari fungsi sigmoid."""
        return x * (1 - x)

    def forward(self, X):
        """
        Melakukan forward propagation.
        
        Parameters:
        - X: Data input (n_samples x input_size).

        Returns:
        - output: Output jaringan saraf.
        """
        # Hitung nilai hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Hitung nilai output layer
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.final_input)

        return self.final_output

    def backward(self, X, y, output):
        """
        Melakukan backward propagation untuk menghitung gradien dan memperbarui bobot.

        Parameters:
        - X: Data input (n_samples x input_size).
        - y: Target output (n_samples x output_size).
        - output: Output jaringan saraf dari forward propagation.
        """
        # Error di output layer
        error_output = y - output
        delta_output = error_output * self.sigmoid_derivative(output)

        # Error di hidden layer
        error_hidden = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden * self.sigmoid_derivative(self.hidden_output)

        # Update bobot dan bias
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_output.T, delta_output)
        self.bias_output += self.learning_rate * np.sum(delta_output, axis=0)
        self.weights_input_hidden += self.learning_rate * np.dot(X.T, delta_hidden)
        self.bias_hidden += self.learning_rate * np.sum(delta_hidden, axis=0)

    def fit(self, X, y, epochs=1000):
        """
        Melatih jaringan saraf.

        Parameters:
        - X: Data input (n_samples x input_size).
        - y: Target output (n_samples x output_size).
        - epochs: Jumlah iterasi pelatihan.
        """
        for epoch in range(epochs):
            # Forward propagation
            output = self.forward(X)

            # Backward propagation
            self.backward(X, y, output)

            # (Opsional) Cetak error setiap beberapa iterasi
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X):
        """
        Membuat prediksi menggunakan jaringan saraf.

        Parameters:
        - X: Data input (n_samples x input_size).

        Returns:
        - Prediksi jaringan saraf.
        """
        return self.forward(X)
