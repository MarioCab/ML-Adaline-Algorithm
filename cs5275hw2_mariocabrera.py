# ## Implementing an adaptive linear neuron in Python

from sklearn.datasets import load_breast_cancer
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


import timeit
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

scaler = StandardScaler()
df = load_breast_cancer(as_frame=True).frame
sns.heatmap(df.corr(), annot=False, cmap="viridis")
df.describe()
plt.show(block=True)

SEED = 5275  ## Our replacement for the random_state value

## Adaline class


class AdalineGD:
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    b_ : Scalar
      Bias unit after fitting.
    losses_ : list
      Mean squared eror loss function values in each epoch.

    """

    def __init__(self, eta=0.01, n_iter=50, random_state=SEED):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float_(0.0)
        self.losses_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_) + self.b_

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


## Logic to plot decision regions


def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ("o", "s", "^", "v", "<")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Class {cl}",
            edgecolor="black",
        )


######## Plotting against learning rate 0.1 (ada1) and 0.0001 (ada2) ########

y = df["target"].values  # select target value
training_features = [24, 29]  # select features to use in training
X = df.iloc[0:569, training_features].values  # select training data

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

#### Training for ada1
start_time = timeit.default_timer()
ada1 = AdalineGD(n_iter=15, eta=0.1).fit(X, y)
end_time = timeit.default_timer()
ada1_time = end_time - start_time
print(f"\nTraining time for ada1 (seconds): {ada1_time}")  # Print ada1 training time
ax[0].plot(range(1, len(ada1.losses_) + 1), np.log10(ada1.losses_), marker="o")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("log(Mean squared error)")
ax[0].set_title("Adaline - Learning rate 0.1")

#### Training for ada2
start_time = timeit.default_timer()
ada2 = AdalineGD(n_iter=15, eta=0.0001).fit(X, y)
end_time = timeit.default_timer()
ada2_time = end_time - start_time
print(f"\nTraining time for ada2 (seconds): {ada1_time}")  # Print ada2 training time
ax[1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker="o")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Mean squared error")
ax[1].set_title("Adaline - Learning rate 0.0001")


#### Get loss values for ada1
print("\nLoss values for ada1 (eta 0.1):")
for i, loss in enumerate(ada1.losses_, start=1):
    print(f"Epoch {i}: {loss}")
print("\n")

#### Get loss values for ada2
print("Loss values for ada2 (eta 0.0001):")
for i, loss in enumerate(ada2.losses_, start=1):
    print(f"Epoch {i}: {loss}")
print("\n")

#### Crete and map a confusion matrix for ada1
y_pred = ada1.predict(X)
confmat = confusion_matrix(y, y_pred)
print("Confusion Matrix for ada1:")
print(confmat)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(j, i, str(confmat[i, j]), va="center", ha="center")
ax.xaxis.set_ticks_position("bottom")
plt.title("Ada1 (0.1)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

#### Crete and map a confusion matrix for ada2
y_pred = ada2.predict(X)
confmat = confusion_matrix(y, y_pred)
print("Confusion Matrix for ada2:")
print(confmat)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(j, i, str(confmat[i, j]), va="center", ha="center")
ax.xaxis.set_ticks_position("bottom")
plt.title("Ada2 (0.0001)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")


plt.show()  # Display all of the graphs
