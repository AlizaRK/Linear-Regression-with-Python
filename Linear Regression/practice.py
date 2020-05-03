import numpy as np 
import matplotlib.pyplot as plt

def generate_examples(num=1000):
    W = [1.0 , -3.0]
    b = 1.0

    W = np.reshape(W, (2,1))
    # W = np.shape(W, (1 , 2))
    X = np.random.randn(num, 2)
    # X = np.random.randn(2, num)
    y = b + np.dot(X, W)

    y = np.reshape(y , (num, 1))

    return X, y
X , y = generate_examples()
# print(X.shape, y.shape)
# print(X[0], y[0])

# Task 3 
class Model:
    def __init__(self, features):
        # Features are independent variables, in this case area and distance (2). 
        self.features = features
        self.W = np.random.randn(features, 1) # (2,1)
        self.b = np.random.randn()
model = Model(2)
# print(model.W)
# print(model.b)

# Task 4
class Model(Model):
    def forward_pass(self, X):
        y_hat = self.b + np.dot(X, self.W)
        return y_hat
y_hat = Model(2).forward_pass(X)
# print(y_hat.shape) # Number of y_hat values
# print(y_hat, y_hat[0])

# Task 5
class Model(Model):
    def compute_loss(self, y_hat, y_true):
        return np.sum(np.square((y_hat - y_true)))/(2 * y_hat.shape[0]) # m = number of examples
model = Model(2)
y_hat = model.forward_pass(X)
loss = model.compute_loss(y_hat, y)
print(loss)

# Task 6
class Model(Model):
    def backward_pass(self, X, y_true, y_hat):
        m = y_true.shape[0]
        db = (1/m)*np.sum(y_hat - y_true)
        dW = (1/m)*np.sum(np.dot(np.transpose(y_hat - y_true),X), axis=0) # for the sake of clarity we are using np.sum
        return db, dW
model = Model(2)
X, y = generate_examples()
y_hat = model.forward_pass(X)
db, dW = model.backward_pass(X, y, y_hat)
print(dW, db)

# Task 7
class Model(Model):
    def update_params(self, dW, db, lr): # To prevent the function from diverging we would use lr (learning rate), a factor to converge it. 
        self.W = self.W - (lr * np.reshape(dW, (self.features, 1)))
        self.b = self.b - db


# Task 8
class Model(Model):
    def train(self, x_train, y_train, iterations, lr):
        losses = []
        for i in range(0, iterations):
            y_hat = self.forward_pass(x_train)
            loss = self.compute_loss(y_hat, y_train)
            dW, db = self.backward_pass(x_train, y_train, y_hat)
            self.update_params(dW, db, lr)
            losses.append(loss)
            if i%int(iterations/10)==0:
                print('Iter: {}, Loss: {: .4f}'. format(i, loss))
        return losses
model = Model(2)
x_train, y_train = generate_examples()
losses = model.train(x_train, y_train, 1000, 3e-3)
