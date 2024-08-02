#Recognizing HandWritten Digits
from sklearn import datasets
digits = datasets.load_digits()
dir(digits)

print(digits.images[0])

import matplotlib.pyplot s plt
def plot_multi(i)
    nplots = 16
    fig = plt.figure(figsize=(15, 15))
    for j in range(nplots):
        plt.subplot(4, 4, j+1)
        plt.imshow(digits.images[i+j], cmap='binary')
        plt.title(digits.target[i+j])
        plt.axis('off')
    plt.show()
    plot_multi(0)

y = digits.target
x = digits.images.reshape((len(digits.images), -1))
x.shape
x[0]
x_train = x[:1000]
y_train = y[:1000]
x_test = x[1000:]
y_test = y[1000:]   

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(15,),
                    activation='logistic',
                    alpha=1e-4, solver='sgd',
                    tol=1e-4, random_state=1,
                    learning_rate_init=.1,
                    verbose=True)

mlp.fit(x_train, y_train)

fig, axes = plt.subplots(1, 1)
axes.plot(mlp.loss_curve_, 'o-')
axes.set_xlabel("number of iteration")
axes.set_ylabel("loss")
plt.show()
predictions = mlp.predict(x_test)
predictions[:50]
x_test[:50]

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
