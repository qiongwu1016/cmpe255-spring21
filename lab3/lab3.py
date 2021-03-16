import numpy as np
import os

np.random.seed(42)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "/Users/qiongwu/git/cmpe255-spring21/lab3"
IMAGE_DIR = ""

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", IMAGE_DIR, fig_id + ".png")
    print("Saving figure (index: ", fig_id, ')')
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)
    

def random_digit(X, ind):

    some_digit = X[ind]
    some_digit_image = some_digit.reshape(28, 28)
    plt.imshow(some_digit_image, cmap = mpl.cm.binary,
            interpolation="nearest")
    plt.axis("off")

    save_fig("%d"%ind)
    plt.show()

   
def load_and_sort():
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, cache=True)
        mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
        sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
    except ImportError:
        from sklearn.datasets import fetch_mldata
        mnist = fetch_mldata('MNIST original')
    return mnist["data"], mnist["target"]


def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:60000])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[60000:])]))[:, 1]
    mnist.data[:60000] = mnist.data[reorder_train]
    mnist.target[:60000] = mnist.target[reorder_train]
    mnist.data[60000:] = mnist.data[reorder_test + 60000]
    mnist.target[60000:] = mnist.target[reorder_test + 60000]


def train_predict(some_digit, X, y):
    import numpy as np
    shuffle_index = np.random.permutation(60000)
    X_train, y_train = X[shuffle_index], y[shuffle_index]
    X_test, y_test = X[60000:], y[60000:]

    # Example: Binary number 4 Classifier
    #y_train_4 = (y_train == 4)
    #y_test_4 = (y_test == 4)

    y_train_5 = (y_train == 5)
    y_test_5 = (y_test == 5)


    from sklearn.linear_model import SGDClassifier
    # TODO
    # print prediction result of the given input some_digit
    sgd = SGDClassifier(random_state = 42)
    sgd.fit(X_train, y_train_5)
    prediction = sgd.predict(X_test)
    n_correct = sum(prediction == y_test_5)
    print("accuracy of test data predictions: ", n_correct / len(prediction))
    return sgd, prediction

def calculate_cross_val_score(sgd, X, y):
    # TODO
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(sgd, X, y, cv=3, scoring='accuracy')
    return scores


if __name__ == "__main__":
    from sklearn.linear_model import SGDClassifier
    print('Loading data from sklearn.dataset ...')
    X, y = load_and_sort()
    print('Train SGDClassifier moedel with training dataset ...')
    sgd, prediction = train_predict(5, X, y)
    print('Model training completed')
    
    #calculate_cross_val_score
    sgd1 = SGDClassifier(random_state = 42)
    scores = calculate_cross_val_score(sgd1, X, y)
    print('cross-val scores: ', scores)

    # test out a prediction from the trained SGD model
    from random import seed
    from random import randint
    seed(10)
    print('Test out a prediction from the trained SGD model ...')
    ind = randint(0, 70000)
    pred = sgd.predict(X[ind].reshape(1, -1))
    print('single digit prediction: ', pred, ', actual digit: ', y[ind])
    random_digit(X, ind)
    print('Test out a prediction from the trained SGD model ...')
    ind = randint(0, 70000)
    pred = sgd.predict(X[ind].reshape(1, -1))
    print('single digit prediction: ', pred, ', actual digit: ', y[ind])
    random_digit(X, ind)



