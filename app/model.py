import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_percentage_error
import matplotlib.pyplot as plt
from app.util import getScaler,getXandY
from sklearn.model_selection import train_test_split


class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        
        # Inisialisasi bobot dan bias secara acak
        np.random.seed(42)
        self.weights_input_hidden1 = np.random.rand(self.input_size, self.hidden_size1)
        self.bias_hidden1 = np.random.rand(self.hidden_size1)

        self.weights_hidden1_hidden2 = np.random.rand(self.hidden_size1, self.hidden_size2)
        self.bias_hidden2 = np.random.rand(self.hidden_size2)

        self.weights_hidden2_output = np.random.rand(self.hidden_size2, self.output_size)
        self.bias_output = np.random.rand(self.output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, input_data):
        self.hidden_layer1_input = np.dot(input_data, self.weights_input_hidden1) + self.bias_hidden1
        self.hidden_layer1_output = self.sigmoid(self.hidden_layer1_input)

        self.hidden_layer2_input = np.dot(self.hidden_layer1_output, self.weights_hidden1_hidden2) + self.bias_hidden2
        self.hidden_layer2_output = self.sigmoid(self.hidden_layer2_input)

        self.output_layer_input = np.dot(self.hidden_layer2_output, self.weights_hidden2_output) + self.bias_output
        self.predicted_output = self.sigmoid(self.output_layer_input)
    
    def backward(self, input_data, output_data, learning_rate):
        error = output_data - self.predicted_output

        d_predicted_output = error * self.sigmoid_derivative(self.predicted_output)

        error_hidden_layer2 = d_predicted_output.dot(self.weights_hidden2_output.T)
        d_hidden_layer2 = error_hidden_layer2 * self.sigmoid_derivative(self.hidden_layer2_output)

        error_hidden_layer1 = d_hidden_layer2.dot(self.weights_hidden1_hidden2.T)
        d_hidden_layer1 = error_hidden_layer1 * self.sigmoid_derivative(self.hidden_layer1_output)

        self.weights_hidden2_output += self.hidden_layer2_output.T.dot(d_predicted_output) * learning_rate
        self.bias_output += np.sum(d_predicted_output) * learning_rate

        self.weights_hidden1_hidden2 += self.hidden_layer1_output.T.dot(d_hidden_layer2) * learning_rate
        self.bias_hidden2 += np.sum(d_hidden_layer2) * learning_rate

        self.weights_input_hidden1 += input_data.T.dot(d_hidden_layer1) * learning_rate
        self.bias_hidden1 += np.sum(d_hidden_layer1) * learning_rate
    
    def train(self, input_data, output_data, epochs, learning_rate):
        loss = []
        for epoch in range(epochs):
            self.forward(input_data)
            self.backward(input_data, output_data, learning_rate)
            mse = mean_squared_error(y_pred=self.predicted_output,y_true=output_data)
            loss.append(mse)

        return loss
            
    
    def predict(self, input_data):
        self.forward(input_data)
        return self.predicted_output

# Contoh data training (XOR)
# input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# output_data = np.array([[0], [1], [1], [0]])
# X,y = getXandY('Jambi')

# scalerX = getScaler(X)
# scalerY = getScaler(y)

# X_scaler = scalerX.transform(X)
# y_scaler = scalerY.transform(y)

# X_train,X_test,y_train,y_test =  train_test_split(X_scaler,y_scaler,random_state=42,test_size=0.2)

# # Inisialisasi model
# input_size = 4
# hidden_size1 = 4
# hidden_size2 = 4
# output_size = 1
# learning_rate = 0.1
# epochs = 10000

# model = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)
# # Pelatihan model
# hist = model.train(X_train, y_train, epochs, learning_rate)

# Prediksi setelah pelatihan

# pred = model.predict(X_test)
# pred_ = scalerY.inverse_transform(pred)
# y_test_ = scalerY.inverse_transform(y_test)
# print("Predicted Output:")
# print(pred_)
# print("actual : ")
# print(scalerY.inverse_transform(y_test))
# print()
# print("MAPE : ",mean_absolute_percentage_error(y_pred=pred_,y_true=y_test_))

# plt.plot(hist)
# plt.show()

