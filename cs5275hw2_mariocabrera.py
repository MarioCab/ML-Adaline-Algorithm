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
    ada = AdalineGD(n_iter=iter, eta=eta).fit(X, y)
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


def run_ada_std(eta, iter, feature1, feature2):
    """Uses the given parameters to run a new Adaline algorithm based on standardized features.

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

    X = scaler.fit_transform(X)  # Standardize the selected features

    # Creates a 1 row 1 column figure to map the learning rate
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 4))

    start_time = timeit.default_timer()  # Starts timer
    ada = AdalineGD(n_iter=iter, eta=eta).fit(X, y)
    end_time = timeit.default_timer()  # Ends timer
    ada_time = end_time - start_time  # Shows time took to train
    print(
        f"\nTraining time for STANDARDIZED ada with ({eta}) learning rate (in seconds): {ada_time}"
    )
    ax.plot(range(1, len(ada.losses_) + 1), ada.losses_, marker="o")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean squared error")
    ax.set_title(f"STANDARDIZED Adaline - Learning rate {eta}")

    # Print the loss values
    print(f"\nLoss values for STANDARDIZED ada with ({eta}) learning rate:")
    for i, loss in enumerate(ada.losses_, start=1):
        print(f"Epoch {i}: {loss}")
    print("\n")

    # Calculate confusion matrix
    y_pred = ada.predict(X)
    confmat = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix for STANDARDIZED ada with ({eta}) learning rate:")
    print(confmat)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            ax.text(j, i, str(confmat[i, j]), va="center", ha="center")
    ax.xaxis.set_ticks_position("bottom")
    plt.title(f"STANDARDIZED Ada with ({eta}) learning rate")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.show()  # Display all of the graphs


######## TESTS ########

#### Training for Ada1 with a learning rate of 0.1 on features 24 and 29
ada1 = run_ada(0.1, 15, 24, 29)
"""
  Using a learning rate of 0.1 at 15 epochs we see that the loss function begins to decrease rapidly in the beginning, and then start to converge at around 8 epochs.
  At the end of the 15 epochs we see that the loss function value is sitting at around .23
  Based on the rate that convergence occurred, and that the loss value got to .23 at only 15 epochs it is safe to say that the learning rate here seems to be a good rate to match our given features (or so I thought until analyzing the rest of the data).
  As far as the confusion matrix is concerned, we didn't get any true positive or false positive guesses. This seems to indicate that there were no positive outcomes that the algorithm was guessing at all, just negative outcomes.
  
  Training time for ada with (0.1) learning rate (in seconds): 0.0006319000094663352

Loss values for ada with (0.1) learning rate:
Epoch 1: 0.6276539029634107
Epoch 2: 0.48313504102384225
Epoch 3: 0.3917720851065082
Epoch 4: 0.3340120070512799
Epoch 5: 0.2974942237497543
Epoch 6: 0.27440490639082293
Epoch 7: 0.25980448860004385
Epoch 8: 0.25057038904651485
Epoch 9: 0.24472864616568848
Epoch 10: 0.24103140638843126
Epoch 11: 0.23868983037855854
Epoch 12: 0.23720524784826555
Epoch 13: 0.23626242089722826
Epoch 14: 0.23566206968806827
Epoch 15: 0.23527821808155353


Confusion Matrix for ada with (0.1) learning rate:
[[  0 212]
[  0 357]]
    
    """


#### Training for Ada2 with a learning rate of 0.0001 on features 24 and 29
ada2 = run_ada(0.0001, 15, 24, 29)
"""
  Using a learning rate of 0.0001 at 15 epochs we see that the loss function value decreases linearly at a slow rate and never converges.
  This would suggest that the learning rate is too low for the algorithm and would need to run for
  much longer than 15 epochs for us to see any sort of convergence.
  At the end of the 15 epochs we see that the loss function value is sitting at around .6255 without coming close to converging, so this wouldn't make many accurate guesses without being adjusted.
  As far as the confusion matrix is concerned, we didn't get any false negative or true negative guesses. This seems to indicate that there were no negative outcomes that the algorithm was guessing at all, just positive outcomes.
    
Training time for ada with (0.0001) learning rate (in seconds): 0.0006596000166609883

Loss values for ada with (0.0001) learning rate:
Epoch 1: 0.6276539029634107
Epoch 2: 0.6274929044530949
Epoch 3: 0.6273319719138367
Epoch 4: 0.6271711053186014
Epoch 5: 0.6270103046403658
Epoch 6: 0.6268495698521176
Epoch 7: 0.6266889009268555
Epoch 8: 0.6265282978375897
Epoch 9: 0.6263677605573409
Epoch 10: 0.626207289059141
Epoch 11: 0.6260468833160331
Epoch 12: 0.6258865433010716
Epoch 13: 0.6257262689873212
Epoch 14: 0.6255660603478583
Epoch 15: 0.6254059173557702


Confusion Matrix for ada with (0.0001) learning rate:
[[212   0]
[357   0]]
    
    """

#### Comparing Ada1 and Ada2
"""
These results were interesting and demonstrated a great example of how the learning rate can affect our perception of the data. At a 0.1 learning rate with the given features, we see a loss value of .23
compared to the .6255 of the .0001 learning rate model. Even if the models were to run for an additional 15 epochs (30 total) the 0.0001 learning rate algorithm most likely still would be vastly inferior to the 0.1 model
even taking into consideration that the 0.1 model was already leveling out its convergence at only 15 epochs.

The most interesting data that we found from these comparisons however is the confusion matrix. Even though these models were using the same dataset with the same two features, the 0.1 learning rate model showed a 100%
bias towards positive outcomes, and the 0.0001 model did the opposite, with a 100% bias towards negative outcomes.

What this tells me is that despite my initial beliefs that 0.1 is a vastly superior learning rate, the honest truth is that NEITHER of them seem to be good. 
This is evident in our models bias. The lack of negative outcomes from the 0.1 model suggest that the rate was too fast, causing the weight to be over adjusted.
Similarly, the 0.0001 model showing only negative outcomes suggests that the rate was too slow, causing the weight to not be adjusted fast enough.
    """


#### Training for Ada3 with a learning rate of 0.025 and an epoch of 30 on features 24 and 29
ada3 = run_ada(0.025, 30, 24, 29)
"""
The thought process for choosing the hyperparameters for ada3 were relatively simple. Firstly, I thought 15 iterations was too little to see a convergence without having too drastically a fast learning rate.
Secondly, I chose a learning rate of 0.025 because based on the values of ada1 and ada2 I know that the most optimal value should be somewhere closer to .1 than .001. I wanted ada3 to test this on the lower end, so I set
the value to .025. While the loss value of ada3 beats ada2 by its 2nd epoch, unfortunately, even at ada3's 30th epoch, it still falls slightly short of overtaking ada1's 15th epoch by around a 2% margin. At first I began to
think that the learning rate was still too low, however after analyzing the confusion martix I became more convinced that ada3 actually outperformed ada1. This is because we finally see that the model isn't 100% bias on either
positive or negative outcomes, unlike what I saw in ada1 and ada2. There is still a large bias towards positive outcomes here, however it isn't 100% which is proof that I am getting closer to the general ballpark of what
might be considered a "good" learning rate.

Training time for ada with (0.025) learning rate (in seconds): 0.0009338000090792775

Loss values for ada with (0.025) learning rate:
Epoch 1: 0.6276539029634107
Epoch 2: 0.5884311603808925
Epoch 3: 0.5531238884759965
Epoch 4: 0.5213411899149565
Epoch 5: 0.4927311922337934
Epoch 6: 0.46697715182693256
Epoch 7: 0.44379394689047325
Epoch 8: 0.4229249204891888
Epoch 9: 0.4041390387929795
Epoch 10: 0.38722833301813303
Epoch 11: 0.37200559674998546
Epoch 12: 0.35830231315122324
Epoch 13: 0.3459667891054046
Epoch 14: 0.3348624756365181
Epoch 15: 0.32486645600788416
Epoch 16: 0.31586808476028877
Epoch 17: 0.3077677626204728
Epoch 18: 0.30047583371548614
Epoch 19: 0.2939115928826164
Epoch 20: 0.28800239208360173
Epoch 21: 0.2826828360291455
Epoch 22: 0.2778940581075056
Epoch 23: 0.2735830686000759
Epoch 24: 0.2697021679672551
Epoch 25: 0.2662084187083728
Epoch 26: 0.2630631699479858
Epoch 27: 0.2602316294846572
Epoch 28: 0.2576824785638449
Epoch 29: 0.2553875251095789
Epoch 30: 0.25332139157543004

Confusion Matrix for ada with (0.025) learning rate:
[[182  30]
[349   8]]
    
    """

#### Training for Ada4 with a learning rate of 0.075 and an epoch of 30 on features 24 and 29
ada4 = run_ada(0.0252, 30, 24, 29)
"""
For ada4 I wanted to expand on what I saw from ada3, but just increase the learning rate ever-so-slightly to see how drastic of a difference it would make. Even with just
an increase of two ten-thousandths I was able to see some tangible differences while keeping the same epochs as ada3. While the loss value was about a thousandth of a percentage faster,
the biggest factor I was looking to change in this model was the confusion matrix, and I am not disappointed, as the bias on both the positive end is slowly being accounted for. My reasoning on wanting to level the confusion matrix out is to get to fit more with the actual dataset
which is roughly a 60:40 split on positive to negative values. By that logic, the algorithm should be expecting a fair bit of value types. Getting rid of the major biases would help bring those outer deltas closer to a
converging point, which would help me determine if my learning rate is ideal or not.
    
Training time for ada with (0.0252) learning rate (in seconds): 0.0009004000166896731

Loss values for ada with (0.0252) learning rate:
Epoch 1: 0.6276539029634107
Epoch 2: 0.5881256924970708
Epoch 3: 0.5525741776727562
Epoch 4: 0.5205992569031941
Epoch 5: 0.4918410834515521
Epoch 6: 0.4659760153275758
Epoch 7: 0.44271297267105086
Epoch 8: 0.4217901616246476
Epoch 9: 0.4029721278230757
Epoch 10: 0.3860471063353326
Epoch 11: 0.3708246382334336
Epoch 12: 0.35713342696191297
Epoch 13: 0.34481941038136427
Epoch 14: 0.33374402678671383
Epoch 15: 0.3237826553841258
Epoch 16: 0.31482321367397936
Epoch 17: 0.30676489595335366
Epoch 18: 0.29951703873976193
Epoch 19: 0.29299810034639034
Epoch 20: 0.2871347431238779
Epoch 21: 0.28186100803919556
Epoch 22: 0.277117572301443
Epoch 23: 0.2728510816790833
Epoch 24: 0.2690135499937908
Epoch 25: 0.2655618190321684
Epoch 26: 0.2624570727965972
Epoch 27: 0.25966440062807217
Epoch 28: 0.25715240428393793
Epoch 29: 0.25489284454815075
Epoch 30: 0.2528603233966395


Confusion Matrix for ada with (0.0252) learning rate:
[[128  84]
[324  33]]
    
    """

#### Run ada1 again with the same parameters || STANDARDIZED
ada1_Std = run_ada_std(0.1, 15, 24, 29)
"""
Training time for STANDARDIZED ada with (0.1) learning rate (in seconds): 0.0005957999965175986

Loss values for STANDARDIZED ada with (0.1) learning rate:
Epoch 1: 0.6263276377601824
Epoch 2: 0.4632935510478784
Epoch 3: 0.36262818192214763
Epoch 4: 0.2998396552932518
Epoch 5: 0.26036579527424997
Epoch 6: 0.23539499092778793
Epoch 7: 0.21951953227017063
Epoch 8: 0.20938381484017088
Epoch 9: 0.20288784722453113
Epoch 10: 0.19870889905304634
Epoch 11: 0.19600971329318337
Epoch 12: 0.19425832123895923
Epoch 13: 0.19311571614847917
Epoch 14: 0.192365325458327
Epoch 15: 0.1918684889260187


Confusion Matrix for STANDARDIZED ada with (0.1) learning rate:
[[101 111]
[ 50 307]]

    """

#### Run ada2 again with the same parameters || STANDARDIZED
ada2_Std = run_ada_std(0.0001, 15, 24, 29)
"""
Training time for STANDARDIZED ada with (0.1) learning rate (in seconds): 0.0005957999965175986

Loss values for STANDARDIZED ada with (0.1) learning rate:
Epoch 1: 0.6263276377601824
Epoch 2: 0.4632935510478784
Epoch 3: 0.36262818192214763
Epoch 4: 0.2998396552932518
Epoch 5: 0.26036579527424997
Epoch 6: 0.23539499092778793
Epoch 7: 0.21951953227017063
Epoch 8: 0.20938381484017088
Epoch 9: 0.20288784722453113
Epoch 10: 0.19870889905304634
Epoch 11: 0.19600971329318337
Epoch 12: 0.19425832123895923
Epoch 13: 0.19311571614847917
Epoch 14: 0.192365325458327
Epoch 15: 0.1918684889260187


Confusion Matrix for STANDARDIZED ada with (0.1) learning rate:
[[101 111]
[ 50 307]]

    """

#### Run ada3 again with the same parameters || STANDARDIZED
ada3_Std = run_ada_std(0.025, 30, 24, 29)
"""
Training time for STANDARDIZED ada with (0.1) learning rate (in seconds): 0.0005957999965175986

Loss values for STANDARDIZED ada with (0.1) learning rate:
Epoch 1: 0.6263276377601824
Epoch 2: 0.4632935510478784
Epoch 3: 0.36262818192214763
Epoch 4: 0.2998396552932518
Epoch 5: 0.26036579527424997
Epoch 6: 0.23539499092778793
Epoch 7: 0.21951953227017063
Epoch 8: 0.20938381484017088
Epoch 9: 0.20288784722453113
Epoch 10: 0.19870889905304634
Epoch 11: 0.19600971329318337
Epoch 12: 0.19425832123895923
Epoch 13: 0.19311571614847917
Epoch 14: 0.192365325458327
Epoch 15: 0.1918684889260187


Confusion Matrix for STANDARDIZED ada with (0.1) learning rate:
[[101 111]
[ 50 307]]

    """

#### Run ada4 again with the same parameters || STANDARDIZED
ada4_Std = run_ada_std(0.0252, 30, 24, 29)
"""
Training time for STANDARDIZED ada with (0.1) learning rate (in seconds): 0.0005957999965175986

Loss values for STANDARDIZED ada with (0.1) learning rate:
Epoch 1: 0.6263276377601824
Epoch 2: 0.4632935510478784
Epoch 3: 0.36262818192214763
Epoch 4: 0.2998396552932518
Epoch 5: 0.26036579527424997
Epoch 6: 0.23539499092778793
Epoch 7: 0.21951953227017063
Epoch 8: 0.20938381484017088
Epoch 9: 0.20288784722453113
Epoch 10: 0.19870889905304634
Epoch 11: 0.19600971329318337
Epoch 12: 0.19425832123895923
Epoch 13: 0.19311571614847917
Epoch 14: 0.192365325458327
Epoch 15: 0.1918684889260187


Confusion Matrix for STANDARDIZED ada with (0.1) learning rate:
[[101 111]
[ 50 307]]

    """
