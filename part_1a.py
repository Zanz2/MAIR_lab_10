import numpy as np
import pandas as pd
import seaborn as sn
import pickle as pkl
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from os import listdir, path


# preparing the data, splitting the labels from the sentences, using a vectorizer to be able to process data
# then count the amount of times each label occurs
class Dataset:
    def __init__(self, full_data, label_to_id, vectorizer):
        self.full_data = full_data
        self.sentences = [line[1] for line in self.full_data]
        self.vectorized = vectorizer.transform(self.sentences)
        self.labels = [line[0] for line in self.full_data]
        self.ids = [label_to_id[label] for label in self.labels]
        self.occurrences = self.__count_occurrences(label_to_id)

    def __count_occurrences(self, label_to_id):
        occurrences = {label: 0 for label in label_to_id}
        for label in self.labels:
            occurrences[label] += 1
        return occurrences


# split data into training set, test set, full set, developing set
# then parse the data and adjust it by removing newline characters and convert upper- to lowercase letters
class DataElements:
    def __init__(self, filename):
        self.filename = filename
        self.original_data = self.__parse_data()
        self.n_traindev = int(len(self.original_data) * 0.85)  # Here is the number that will be in the Training split
        self.n_train = int(self.n_traindev * 0.85)

        self.vectorizer = TfidfVectorizer()  # This will change our sentences into vectors of words, will calculate occurances and also normalize them
        self.vectorizer.fit([sentence[1] for sentence in self.original_data[:self.n_train]])  # Makes a sparse word matrix out of the training corpus (vectorizes it)
        self.unique_labels = list(set(sentence[0] for sentence in self.original_data))
        self.label_to_id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id_to_label = {i: label for i, label in enumerate(self.unique_labels)}
        self.cached_clfs = {}

        self.fullset = Dataset(self.original_data, self.label_to_id, self.vectorizer)
        self.trainset = Dataset(self.original_data[:self.n_train], self.label_to_id, self.vectorizer)
        self.devset = Dataset(self.original_data[self.n_train:self.n_traindev], self.label_to_id, self.vectorizer)
        self.testset = Dataset(self.original_data[self.n_traindev:], self.label_to_id, self.vectorizer)

    def __parse_data(self):
        original_data = []
        with open(self.filename) as f:
            content = f.readlines()
            for line in content:  # I open the file, read its contents line by line, remove the trailing newline character, and convert them to lowercase
                line = line.rstrip("\n").lower()
                line = line.split(" ", 1)  # this just seperates te first dialog_act from the remaining sentence
                original_data.append(line)
        return original_data

    # if the classifier was already trained and added in cached_clfs then it returns that stored instance, otherwise it first trains the classifier on
    # the supplied train_x, train_y data, stores the trained classifier, and returns it
    def get_fitted_classifier(self, classifier, clf, train_x, train_y):
        if classifier not in self.cached_clfs:
            clf.fit(train_x, train_y)
            self.cached_clfs[classifier] = clf
        return self.cached_clfs[classifier]
    
    # calculate the average amount of words of the given sentences
    def average_sentence_length(self):
        total_words = sum(len(sentence.split(" ")) for sentence in self.fullset.sentences)
        return total_words / len(self.fullset.sentences)
    
    # check which words are not in the trainset, but do appear in the devset and testset 
    def out_of_vocabulary(self):
        trainset_voc, out_of_voc_devset, out_of_voc_testset = [], [], []
        for utterance in self.trainset.sentences:
            words = utterance.split(" ")
            for word in words:
                if word not in trainset_voc:
                    trainset_voc.append(word)
        for utterance in self.devset.sentences:
            words = utterance.split(" ")
            for word in words:
                if word not in trainset_voc and word not in out_of_voc_devset:
                    out_of_voc_devset.append(word)
        for utterance in self.testset.sentences:
            words = utterance.split(" ")
            for word in words:
                if word not in trainset_voc and word not in out_of_voc_testset:
                    out_of_voc_testset.append(word)
        print(f"In devset and not in trainset: {out_of_voc_devset}")
        print(f"In testset and not in trainset: {out_of_voc_testset}")

    def print_statistics(self):
        print(self.fullset.occurrences)
        print("Average sentence length: ", self.average_sentence_length())
        self.out_of_vocabulary()


# we define a function to calculate the accuracy, this is done by returning the sum in which the actual labels match the predicted labels
def calculate_accuracy(true_labels, predicted_labels):
    length = len(true_labels)
    assert(len(predicted_labels) == length)
    return sum(true_labels[i] == predicted_labels[i] for i in range(length)) / length


# we count the amount of true positives (the label is assigned and correct), true negatives (the label is rightfully not assigned), false negatives
# (the label is not assigned when it should have been) and false positives (the label is assigned when it should not have been)
def count_prediction_accuracies(true_labels, predicted_labels):
    length = len(true_labels)
    assert (len(predicted_labels) == length)
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
    for i in range(len(true_labels)):
        if true_labels[i]:
            if predicted_labels[i]:
                true_pos += 1
            else:
                false_neg += 1
        else:
            if predicted_labels[i]:
                false_pos += 1
            else:
                true_neg += 1
    return {"true_pos": true_pos, "true_neg": true_neg, "false_pos": false_pos, "false_neg": false_neg}


# Evaluation metric: we define a function to calculate precision, this is done by dividing the true positives by the
# total amount of positives (the percentage of correctly assigned positives).
def calculate_precision(true_labels, predicted_labels):
    counts = count_prediction_accuracies(true_labels, predicted_labels)
    if counts["true_pos"] == 0 and counts["false_pos"] == 0:
        return 0.
    return counts["true_pos"] / (counts["true_pos"] + counts["false_pos"])


# Evaluation metric: we define a function to calculate recall, which is done by dividing the true positives by the total
# amount of correct predictions.
def calculate_recall(true_labels, predicted_labels):
    counts = count_prediction_accuracies(true_labels, predicted_labels)
    if counts["true_pos"] == 0 and counts["false_neg"] == 0:
        return 0.
    return counts["true_pos"] / (counts["true_pos"] + counts["false_neg"])


# Evaluation metric: function to calculate f1-score, which is a generalised version of precision and recall.
def calculate_f1score(true_labels, predicted_labels):
    precision = calculate_precision(true_labels, predicted_labels)
    recall = calculate_recall(true_labels, predicted_labels)
    if precision == 0 and recall == 0:
        return 0.
    return 2 * precision * recall / (precision + recall)


# a function to call upon the previously defined metric functions, for ease of use and consistency
def calculate_evaluationmetric(metric, true_labels, predicted_labels):
    if metric == "precision":
        return calculate_precision(true_labels, predicted_labels)
    elif metric == "recall":
        return calculate_recall(true_labels, predicted_labels)
    elif metric == "f1score":
        return calculate_f1score(true_labels, predicted_labels)
    elif metric == "accuracy":
        return calculate_accuracy(true_labels, predicted_labels)
    else:
        raise NotImplementedError()


# a function which calculates the f1score for all labels together
def calculate_multiclassf1score(true_labels, predicted_labels, occurrences, weighted=False):
    length = len(true_labels)
    assert(len(predicted_labels) == length)
    f1scores = {}
    for label in occurrences:
        binary_true = [tl == label for tl in true_labels]
        binary_pred = [pl == label for pl in predicted_labels]
        f1scores[label] = calculate_f1score(binary_true, binary_pred)
    if weighted:
        return sum(f1scores[label] * occurrences[label] for label in f1scores) / sum(occurrences.values())
    else:
        return sum(f1scores.values()) / len(f1scores)


# show the user the outcome of the metrics
def print_evaluation_metrics(true_labels, predicted_labels, occurrences, name):
    accuracy = calculate_accuracy(true_labels, predicted_labels)
    meanf1score = calculate_multiclassf1score(true_labels, predicted_labels, occurrences, weighted=False)
    weightedf1score = calculate_multiclassf1score(true_labels, predicted_labels, occurrences, weighted=True)
    print(f"{name} evaluation metrics:")
    print(f"    Prediction Accuracy: {accuracy}")
    print(f"          Mean F1-score: {meanf1score}")
    print(f"      Weighted F1-score: {weightedf1score}")


# plot a confusion matrix with the true labels on the y-axis and the predicted labels on the x-axis
def plot_confusion_matrix(true_labels, predicted_labels, unique_labels, name):
    assert(len(true_labels) == len(predicted_labels))
    counts = np.zeros((len(unique_labels), len(unique_labels)))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    for i in range(len(true_labels)):
        true_index = label_to_index[true_labels[i]]
        pred_index = label_to_index[predicted_labels[i]]
        counts[true_index, pred_index] += 1
    dataframe = pd.DataFrame(counts, index=unique_labels, columns=unique_labels)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sn.heatmap(dataframe, annot=True, fmt="g", ax=ax)
    ax.set_title(name)
    return fig


# Baseline 1: we define a majority classiefier, which finds the most commonly occurring label - the majority class - and assigns it to every sentence
def majority_classifier(data, dataset):
    majority_class = max(data.trainset.occurrences.items(), key=operator.itemgetter(1))[0]  # Here it returns the dialogue act that occurs the most times, in this case "inform"
    predictions = [majority_class for _ in range(len(dataset))]
    return predictions


# Baseline 2: we define a rule based classifier, in which we connect utterances to labels, such as 'how about' to the 'reqalts' label
def rule_based(_, dataset):
    # This is a dictionary with values as the dialogue act and keys as the text to be looked for
    # (example: if sentance contains 'is there' we classify it as reqalts dialogue act)
    prediction_dict = {
        "bye": "bye",
        "goodbye": "bye",
        "thank": "thankyou",
        "how about": "reqalts",
        "is there": "reqalts",
        "what": "request",
        "is it": "confirm",
        "i": "inform",
        "no": "negate",
        "yes": "affirm",
        "hello": "hello",
        "im": "inform",
        "any": "inform",
        "phone": "request",
        "address": "request",
        "post": "request",
        "food": "inform",
        "west": "inform",
        "east": "inform",
        "centre": "inform",
        "north": "inform",
        "south": "inform"
    }
    predictions = []
    for sentence in dataset:
        p = "null"
        for key, prediction in prediction_dict.items():
            if key in sentence:
                p = prediction
                break
        predictions.append(p)
    return predictions


# Alternative classifier 1: we define a decision tree, through the scikit predefined functions, learns simple decision
# rules inferred from data features we set the mas depth at 30 to avoid overfitting.
def decision_tree(data, dataset):  # https://scikit-learn.org/stable/modules/tree.html
    clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=30)
    # I set criterion as entropy and split as best, so hopefully it will split on inform class
    # https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    # cached_clf is the trained classifier
    cached_clf = data.get_fitted_classifier("decisiontree", clf, data.trainset.vectorized, data.trainset.labels)
    # tree.plot_tree(clf, fontsize=5)  # This will plot the graph if you uncomment it
    # plt.show()
    return [r for r in cached_clf.predict(dataset)]


# Alternative classifier 2: define a feedforward neural network , through the scikit predefined function, trains on
# dataset and then tested on validation set we opt for the solver because of the improvement in speed
def ff_nn(data, dataset):  # feed forward neural network https://scikit-learn.org/stable/modules/neural_networks_supervised.html
    clf = MLPClassifier(solver='adam', alpha=0.001, random_state=1, early_stopping=False)  # will stop early if small validation subset isnt improving while training
    # cached_clf is the trained classifier (if it still needs to be trained, then it takes a minute or so, depending on your pc)
    cached_clf = data.get_fitted_classifier("neuralnet", clf, data.trainset.vectorized, data.trainset.labels)
    return [r for r in cached_clf.predict(dataset)]  # Accuracy is 0.9866 on validation sets


# Alternative classifier 3: stochastic gradient descent is a linear optimisation technique, we pick 20 iterations, it
# performs relatively well like that except for some minority classes
def sto_gr_des(data, dataset):  # stochastic gradient descent https://scikit-learn.org/stable/modules/sgd.html
    clf = SGDClassifier(loss="modified_huber", penalty="l2", max_iter=20, early_stopping=False)  # requires a mix_iter (maximum of iterations) of at least 7
    # loss could be different loss-functions that measures models fits. I chose modified_huber (smoothed hinge-loss) since it leads to the highest accuracy (could be changed with regards to other eval-methods)
    # penalty penalizes model complexity
    cached_clf = data.get_fitted_classifier("sgradientdescent", clf, data.trainset.vectorized, data.trainset.labels)
    return [r for r in cached_clf.predict(dataset)]  # accuracy of ~97%


# Evaluation: we define a dictionary in which we save the metrics (precision, recall and f1score) of our models
# once stored, we plot a graph to visualise performance across models
def comparison_evaluation(data):
    predictions = {
        "majority": majority_classifier(data, data.devset.sentences),
        "rulebased": rule_based(data, data.devset.sentences),
        "decisiontree": decision_tree(data, data.devset.vectorized),
        "neuralnet": ff_nn(data, data.devset.vectorized),
        "sgradientdescent": sto_gr_des(data, data.devset.vectorized)}
    labels = [lb for lb in data.label_to_id]
    true_labels = [label for label in data.devset.labels]
    metrics = ["precision", "recall", "f1score"]
    evaluations = {}
    for metric in metrics:
        evaluations[metric] = {}
        for label in labels:
            binary_true = [tl == label for tl in true_labels]
            evaluations[metric][label] = {}
            for classifier in predictions:
                binary_pred = [pl == label for pl in predictions[classifier]]
                evaluations[metric][label][classifier] = calculate_evaluationmetric(metric, binary_true, binary_pred)
    fig, axes = plt.subplots(len(evaluations), 1, sharex="all", sharey="all")
    barwidth = 1 / (len(predictions) + 1)
    numbered = [i for i in range(len(labels))]
    for i, metric in enumerate(evaluations):
        axes[i].set_title(metric)
        for j, classifier in enumerate(predictions):
            x_offset = -0.5 * len(predictions) * barwidth + j * barwidth
            x_values = [n + x_offset for n in numbered]
            y_values = [evaluations[metric][lb][classifier] for lb in labels]
            axes[i].bar(x_values, y_values, barwidth, label=classifier)
        axes[i].set_xticks(numbered)
        axes[i].set_xticklabels(labels)
    axes[0].legend(loc=4)
    plt.show()
    fig.set_size_inches(18.5, 10.5)
    fig.savefig('metric_plot', dpi=150)
    # also plot the confusion matrix for all classifiers
    for classifier in predictions:
        fig = plot_confusion_matrix(data.devset.labels, predictions[classifier], data.unique_labels, classifier)
        fig.savefig(f"confusion_matrix_{classifier}", dpi=150)


# we define a function allowing the user to interact with our models
# this will process user input according to the defined classifier
def interact(data, classifier, vectorize=True):
    while True:
        test_text = input(f"Please input a sentence (to be classified by {classifier.__name__}): ").lower()
        if vectorize:
            predicted_label = classifier(data, data.vectorizer.transform([test_text]))
        else:
            predicted_label = classifier(data, [test_text])
        print(f"The sentence (\"{test_text}\") is classified as: {predicted_label}")
        test_text = input("Enter 0 to exit, anything else to continue")
        if str(test_text) == "0":
            break


# The function below calls different models with the supplied training and test data, it predicts the classes of sentances
# in the supplied dataset
def analyse_validation(data, classifier, vectorize=True):
    if vectorize:
        predictions = classifier(data, data.devset.vectorized)
    else:
        predictions = classifier(data, data.devset.sentences)
    print_evaluation_metrics(data.devset.labels, predictions, data.devset.occurrences, str(classifier.__name__))


# Takes a sentance as an argument, returns its predicted label as result, (no direct user input, used for classes)
def predict_sentence(data, supplied_text, classifier, vectorize=True):
    supplied_text = supplied_text.lower()
    if vectorize:
        predicted_label = classifier(data, data.vectorizer.transform([supplied_text]))
    else:
        predicted_label = classifier(data, [supplied_text])
    if predicted_label == ["null"]:
        # if it cannot classify, use the same sentence again but try to predict it with rule based classifier (edge cases)
        predicted_label = predict_sentence(data, supplied_text, rule_based, vectorize=False)
    return predicted_label


class NeuralNetTuner:
    # This class contains a couple of methods that have helped us in finding the right parameters for our classifier.
    @staticmethod
    def fit_nn(data, clf, save=None):
        # Train an MLP classifier and save it in the 'trained_classiffiers' directory.
        clf.fit(data.trainset.vectorized, data.trainset.labels)
        predictions = [r for r in clf.predict(data.devset.vectorized)]
        print_evaluation_metrics(data.devset.labels, predictions, data.devset.occurrences, str(save))
        if save is not None:
            with open(f"trained_classifiers\\{save}", "wb") as save_file:
                pkl.dump(clf, save_file)

    @classmethod
    def fit_nn_hyperparameter_variations(cls, data, selection_round, threshold, max_iter, tol):
        # Train a range of different classifiers: with varying learning rates and different structures for the hidden layers.
        learning_rate_inits = [0.01, 0.001, 0.0001]
        learning_rates = ["constant", "adaptive"]
        hidden_layer_sizes = [(100,), (50, 50), (20, 20, 20), (80, 40, 20, 10)]
        for learning_rate_init in learning_rate_inits:
            for learning_rate in learning_rates:
                for hidden_layer_size in hidden_layer_sizes:
                    lsize = "x".join(str(i) for i in hidden_layer_size)
                    save_name = f"{learning_rate_init}_{learning_rate}_{lsize}"
                    if selection_round == 1:
                        clf = MLPClassifier(solver="adam", learning_rate_init=learning_rate_init, alpha=0.001,
                                            max_iter=200, learning_rate=learning_rate,
                                            hidden_layer_sizes=hidden_layer_size)
                        cls.fit_nn(data, clf, save=f"r1_{save_name}")
                    else:
                        old_file = f"trained_classifiers\\r{selection_round - 1}_{save_name}"
                        if path.exists(old_file):
                            with open(old_file, "rb") as saved_file:
                                clf = pkl.load(saved_file)
                            preds = [r for r in clf.predict(data.devset.vectorized)]
                            meanf1 = calculate_multiclassf1score(data.devset.labels, preds, data.devset.occurrences,
                                                                 weighted=False)
                            if meanf1 > threshold:
                                clf.set_params(max_iter=max_iter, tol=tol)
                                cls.fit_nn(data, clf, save=f"r{selection_round}_{save_name}")

    @staticmethod
    def compare_nn_versions(data):
        classifiers = []
        dataset = data.devset
        for file in listdir("trained_classifiers\\"):
            with open(f"trained_classifiers\\{file}", "rb") as saved_file:
                clf = pkl.load(saved_file)
            pred = [r for r in clf.predict(dataset.vectorized)]
            accuracy = calculate_accuracy(dataset.labels, pred)
            meanf1score = calculate_multiclassf1score(dataset.labels, pred, dataset.occurrences, weighted=False)
            weightedf1score = calculate_multiclassf1score(dataset.labels, pred, dataset.occurrences, weighted=True)
            classifiers.append((file, accuracy, meanf1score, weightedf1score))
        classifiers.sort(key=lambda c: -c[1] - c[2] - c[3])
        for i, m in enumerate(("ACCURACY", "MEANF1SCORE", "WEIGHTEDF1SCORE")):
            classifiers.sort(key=lambda c: -c[i + 1])
            print(f"\nSORTED BY {m}:")
            print("\n".join(f"{c[0]:50}: {c[1]:.4f}{' ' * 10}{c[2]:.4f}{' ' * 10}{c[3]:.4f}" for c in classifiers))


# load dialog_acts, show options of interaction and display to user, process user request
def main(analyse=False):
    data_elements = DataElements("dialog_acts.dat")
    if analyse:  # Analysis mode.
        data_elements.print_statistics()
        # NeuralNetTuner.fit_nn_hyperparameter_variations(data_elements, 2, 0.89, 10000, 0.000001)
        NeuralNetTuner.compare_nn_versions(data_elements)
    else:
        while True:
            print("Enter")
            print("0 for exit")
            print("1 for Majority classifier on test data")
            print("2 for manual prediction on test data")
            print("3 for Decision tree on test data")
            print("4 for Feed forward neural network on test data")
            print("5 for Stochastic gradient descent on test data")
            print("1i for Majority classifier on user input")
            print("2i for manual prediction on user input")
            print("3i for Decision tree on user input")
            print("4i for Feed forward neural network on user input")
            print("5i for Stochastic gradient descent on user input")
            print("c for Comparison Evaluation")
            print("d to talk to with our recommender chatbot")
            test_text = input()
            command = str(test_text)
            if command == "0":
                break
            elif command == "1":
                analyse_validation(data_elements, majority_classifier, vectorize=False)
            elif command == "2":
                analyse_validation(data_elements, rule_based, vectorize=False)
            elif command == "3":
                analyse_validation(data_elements, decision_tree)
            elif command == "4":
                analyse_validation(data_elements, ff_nn)
            elif command == "5":
                analyse_validation(data_elements, sto_gr_des)
            elif command == "1i":
                interact(data_elements, majority_classifier, vectorize=False)
            elif command == "2i":
                interact(data_elements, rule_based, vectorize=False)
            elif command == "3i":
                interact(data_elements, decision_tree)
            elif command == "4i":
                interact(data_elements, ff_nn)
            elif command == "5i":
                interact(data_elements, sto_gr_des)
            elif command == "c":
                comparison_evaluation(data_elements)
                break  # break out of loop to execute the plot.
            else:
                break


if __name__ == "__main__":
    main()
