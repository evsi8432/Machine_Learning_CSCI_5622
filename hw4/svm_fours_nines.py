import argparse
import numpy as np

# from svm import weight_vector, find_support, find_slack
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        train_set, valid_set, test_set = pickle.load(f)

        self.x_train = train_set[0][np.where(np.logical_or(train_set[1] == 4, train_set[1] == 9))[0], :]
        self.y_train = train_set[1][np.where(np.logical_or(train_set[1] == 4, train_set[1] == 9))[0]]

        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff, :]
        self.y_train = self.y_train[shuff]

        self.x_valid = valid_set[0][np.where(np.logical_or(valid_set[1] == 4, valid_set[1] == 9))[0], :]
        self.y_valid = valid_set[1][np.where(np.logical_or(valid_set[1] == 4, valid_set[1] == 9))[0]]

        self.x_test = test_set[0][np.where(np.logical_or(test_set[1] == 4, test_set[1] == 9))[0], :]
        self.y_test = test_set[1][np.where(np.logical_or(test_set[1] == 4, test_set[1] == 9))[0]]

        f.close()


def mnist_digit_show(flatimage, outname=None):
    import matplotlib.pyplot as plt

    image = np.reshape(flatimage, (-1, 28))

    plt.matshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    if outname:
        plt.savefig(outname)
    else:
        plt.show()


if __name__ == "__main__":

    limit = 1000

    parser = argparse.ArgumentParser(description='SVM classifier options')
    parser.add_argument('--limit', type=int, default=1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = FoursAndNines("../data/mnist.pkl.gz")
    data.x_train = data.x_train[0:limit]
    data.y_train = data.y_train[0:limit]

    svc_lin = SVC(kernel='linear')
    svc_rbf = SVC(kernel='rbf')
    svc_pol = SVC(kernel='poly')

    clf_lin = GridSearchCV(svc_lin, {'C': [0.0001, 0.001, 0.01, 0.1, 1]})
    clf_rbf = GridSearchCV(svc_rbf, {'C': [0.1, 1, 10, 100, 1000], 'gamma': [0.1, 0.01, 0.001, 0.0001]})
    clf_pol = GridSearchCV(svc_pol, {'C': [0.1, 1, 10, 100, 1000, 10000], 'degree': [1, 2, 3, 4, 5]})

    clf_lin.fit(data.x_train, data.y_train)
    clf_rbf.fit(data.x_train, data.y_train)
    clf_pol.fit(data.x_train, data.y_train)

    print('train accuracy:')
    print(clf_lin.score(data.x_train, data.y_train))
    print(clf_rbf.score(data.x_train, data.y_train))
    print(clf_pol.score(data.x_train, data.y_train))

    print('optimal params:')
    print(clf_lin.best_params_)
    print(clf_rbf.best_params_)
    print(clf_pol.best_params_)

    print('test accuracy:')
    lin_final = clf_lin.best_estimator_
    lin_final.fit(data.x_train, data.y_train)
    print(lin_final.score(data.x_test, data.y_test))

    rbf_final = clf_rbf.best_estimator_
    rbf_final.fit(data.x_train, data.y_train)
    print(rbf_final.score(data.x_test, data.y_test))

    pol_final = clf_pol.best_estimator_
    pol_final.fit(data.x_train, data.y_train)
    print(pol_final.score(data.x_test, data.y_test))

    support_inds = pol_final.support_

    # -----------------------------------
    # Plotting Examples
    # -----------------------------------

    # Display in on screen
    predicts = []
    print(len(support_inds))
    for i in support_inds:
        if len(predicts) == 2:
            break
        print(i)
        pred = pol_final.predict(data.x_train[i].reshape(1, -1))[0]
        print(pred)
        if pred not in predicts:
            mnist_digit_show(data.x_train[i, :])
            predicts.append(pred)
            # Plot image to file
            # mnist_digit_show(data.x_train[1,:], "mnistfig.png")
