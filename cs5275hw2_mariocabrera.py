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


def run_ada(eta, iter, feature1, feature2):
    """Uses the given parameters to run a new Adaline algorithm.

    Parameters
    ----------
    eta : float
      Learning rate (between 0.0 and 1.0)
    iter : integer
      Epochs to run the algorithm with
    feature1 : integer
      Training feature 1.
    feature2 : integer
      Training feature 2.

    """
    y = df["target"].values  # select target value
    training_features = [feature1, feature2]  # select features to use in training
    X = df.iloc[0:569, training_features].values  # select training data

    # Creates a 1 row 1 column figure to map the learning rate
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

    start_time = timeit.default_timer()  # Starts timer
    ada = AdalineGD(n_iter=15, eta=eta).fit(X, y)
    end_time = timeit.default_timer()  # Ends timer
    ada_time = end_time - start_time  # Shows time took to train
    print(
        f"\nTraining time for ada with ({eta}) learning rate (in seconds): {ada_time}"
    )
    ax.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker="o")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean squared error")
    ax.set_title(f"Adaline - Learning rate {eta}")

    # Print the loss values
    print(f"\nLoss values for ada with ({eta}) learning rate:")
    for i, loss in enumerate(ada.losses_, start=1):
        print(f"Epoch {i}: {loss}")
    print("\n")

    # Calculate confusion matrix
    y_pred = ada.predict(X)
    confmat = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix for ada with ({eta}) learning rate:")
    print(confmat)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(j, i, str(confmat[i, j]), va="center", ha="center")
    ax.xaxis.set_ticks_position("bottom")
    plt.title(f"Ada with ({eta}) learning rate")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.show()  # Display all of the graphs


######## TEST ADA1 vs ADA2 ########

#### Training for Ada1 with a learning rate of 0.1 on features 24 and 29
ada1 = run_ada(0.1, 15, 24, 29)
"""
  Using a learning rate of 0.1 at 15 epochs we see that the loss function begins to decrease rapidly in the beginning, and then start to converge at around 8 epochs.
  At the end of the 15 epochs we see that the loss function value is sitting at around .25
  Based on the rate that convergence occurred, and that the loss value got to .25 at only 15 epochs it is safe to say that the learning rate here seems to be a good rate to match our given features (or so I thought until analyzing the rest of the data).
  As far as the confusion matrix is concerned, we didn't get any true positive or false positive guesses. This seems to indicate that there were no positive outcomes that the algorithm was guessing at all, just negative outcomes.
    """


#### Training for Ada2 with a learning rate of 0.0001 on features 24 and 29
ada2 = run_ada(0.0001, 15, 24, 29)
"""
  Using a learning rate of 0.0001 at 15 epochs we see that the loss function value decreases linearly at a slow rate and never converges.
  This would suggest that the learning rate is too low for the algorithm and would need to run for
  much longer than 15 epochs for us to see any sort of convergence.
  At the end of the 15 epochs we see that the loss function value is sitting at around .6255 without coming close to converging, so this wouldn't make many accurate guesses without being adjusted.
  As far as the confusion matrix is concerned, we didn't get any false negative or true negative guesses. This seems to indicate that there were no negative outcomes that the algorithm was guessing at all, just positive outcomes.
    """

#### Comparing Ada1 and Ada2
"""
These results were interesting and demonstrated a great example of how the learning rate can affect our perception of the data. At a 0.1 learning rate with the given features, we see a loss value of .25
compared to the .6255 of the .0001 learning rate model. Even if the models were to run for an additional 15 epochs (30 total) the 0.0001 learning rate algorithm most likely still would be vastly inferior to the 0.1 model
even taking into consideration that the 0.1 model was already leveling out its convergence at only 15 epochs.

The most interesting data that we found from these comparisons however is the confusion matrix. Even though these models were using the same dataset with the same two features, the 0.1 learning rate model showed a 100%
bias towards positive outcomes, and the 0.0001 model did the opposite, with a 100% bias towards negative outcomes.

What this tells me is that despite my initial beliefs that 0.1 is a vastly superior learning rate, the honest truth is that NEITHER of them seem to be good. 
This is evident in our models bias. The lack of negative outcomes from the 0.1 model suggest that the rate was too fast, causing the weight to be over adjusted.
Similarly, the 0.0001 model showing only negative outcomes suggests that the rate was too slow, causing the weight to not be adjusted fast enough.
    """
