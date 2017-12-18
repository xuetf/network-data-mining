# encoding=utf-8
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import validation_curve
import numpy as np
from constant import *
from sklearn.utils import shuffle

def plot_word_cloud(best_words):
    best_words = dict(best_words)
    # Generate a word cloud image
    cloud = WordCloud(font_path="C:\Windows\Fonts\simhei.ttf")
    wordcloud = cloud.generate_from_frequencies(best_words)
    # Display the generated image:
    # the matplotlib way:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

# plot_word_cloud({u'你好':100, u'我们':200,u'啊啊':30})

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        train_sizes=np.linspace(.1, 1.0, 5), baseline=None, scoring='f1'):
    """
    Generate a simple plot of the test and traning learning curve.
    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.
    title : string
        Title for the chart.
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.
    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects
    """

    X, y = shuffle(X, y) # important for logistic because of the para
    plt.figure()
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv = cv, train_sizes=train_sizes, scoring=scoring) # 垃圾短信neg=1的f1 score
    print train_sizes
    print '-------------'
    print train_scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="b",
             label="Cross-validation score")
    if baseline:
        plt.axhline(y=baseline, color='red', linewidth=5, label='Desired Performance')  # baseline
    plt.xlabel("Training examples")
    plt.ylabel("%s Score"  %scoring)
    plt.legend(loc="best")
    plt.grid("on")
    if ylim:
        plt.ylim(ylim)
    plt.title(title)
    plt.show()

def plot_validation_curve(estimator,title, X, y,
                          param_name, param_range, param_plot_range,
                          x_ticks=np.arange(1.00, 2.10, 0.1), y_ticks=np.arange(0.96, 0.97, 0.001), cv=None, scoring='f1'):
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print train_scores_mean
    print test_scores_mean
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("%s Score" %scoring)


    plt.semilogx(param_plot_range, train_scores_mean, label="Training score",
                 color="r")
    plt.fill_between(param_plot_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="r")
    plt.semilogx(param_plot_range, test_scores_mean, label="Cross-validation score",
                 color="b")
    plt.fill_between(param_plot_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="b")
    plt.yticks(y_ticks)
    plt.xticks(x_ticks,rotation=90)
    plt.legend(loc="best")
    plt.show()


