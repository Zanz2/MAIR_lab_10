import numpy as np
import pandas as pd
import sys
import operator
import random
import matplotlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier

class DataElements:
	def __init__(self, filename):
		self.filename = filename
		self.original_data = self.__parse_data()
		self.training_part_length = int(len(self.original_data) * 0.85)  # Here is the number that will be in the Training split
		self.dialog_acts_counter = self.__count_occurrences()
		self.corpus = [sentence[1] for sentence in self.original_data]
		self.correct_labels = [sentence[0] for sentence in self.original_data]
		self.label_to_id = {key: i for i, (key, value) in enumerate(self.dialog_acts_counter.items())}
		self.id_to_label = {self.label_to_id[label]: label for label in self.label_to_id}
		self.classes_counter = {self.label_to_id[label]: self.dialog_acts_counter[label] for label in self.dialog_acts_counter}
		self.correct_classes = [self.label_to_id[label] for label in self.correct_labels]

		self.training_corpus = self.corpus[:self.training_part_length]
		self.training_classes = self.correct_classes[:self.training_part_length]  # Classes are just ids, the id represents a class label from the mapping
		self.training_labels = self.correct_labels[:self.training_part_length]

		self.test_corpus = self.corpus[self.training_part_length:]
		self.test_classes = self.correct_classes[self.training_part_length:]
		self.test_labels = self.correct_labels[self.training_part_length:]

		self.vectorizer = TfidfVectorizer()  # This will change our sentences into vectors of words, will calculate occurances and also normalize them
		self.vectorizer.fit(self.corpus)  # Makes a sparse word matrix out of the entire word corpus (vectorizes it)
		self.vectorized_training_data = self.vectorizer.transform(self.training_corpus)  # transforms the models so they use that matrix as a representation
		self.vectorized_test_data = self.vectorizer.transform(self.test_corpus)

	def __parse_data(self):
		original_data = []
		with open(self.filename) as f:
			content = f.readlines()
			for line in content:  # I open the file, read its contents line by line, remove the trailing newline character, and convert them to lowercase
				line = line.rstrip("\n").lower()
				line = line.split(" ", 1)  # this just seperates te first dialog_act from the remaining sentence
				original_data.append(line)
		return original_data

	def __count_occurrences(self):
		dialog_acts_counter = {}
		for element in self.original_data[:self.training_part_length]:  # Here I find all the unique dialogue act categories, and count how many occurences there are for each category
			if element[0] not in dialog_acts_counter:
				dialog_acts_counter[element[0]] = 0
			dialog_acts_counter[element[0]] += 1
		return dialog_acts_counter


def calculate_accuracy(true_labels, predicted_labels):
	length = len(true_labels)
	assert(len(predicted_labels) == length)
	return sum(true_labels[i] == predicted_labels[i] for i in range(length)) / length


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


def calculate_precision(true_labels, predicted_labels):
	counts = count_prediction_accuracies(true_labels, predicted_labels)
	return 1. if counts["true_pos"] + counts["false_pos"] == 0 else counts["true_pos"] / (counts["true_pos"] + counts["false_pos"])


def calculate_recall(true_labels, predicted_labels):
	counts = count_prediction_accuracies(true_labels, predicted_labels)
	return 1. if counts["true_pos"] + counts["false_neg"] == 0 else counts["true_pos"] / (counts["true_pos"] + counts["false_neg"])


def calculate_f1score(true_labels, predicted_labels):
	precision = calculate_precision(true_labels, predicted_labels)
	recall = calculate_recall(true_labels, predicted_labels)
	return 2 * precision * recall / (precision + recall)


def calculate_multiclassf1score(true_labels, predicted_labels, dialog_acts_counter, weighted=False):
	length = len(true_labels)
	assert(len(predicted_labels) == length)
	f1scores = {}
	for label in dialog_acts_counter:
		f1scores[label] = calculate_f1score([tl == label for tl in true_labels], [pl == label for pl in predicted_labels])
	if weighted:
		return sum(f1scores[label] * dialog_acts_counter[label] for label in dialog_acts_counter) / sum(dialog_acts_counter.values())
	else:
		return sum(f1scores.values()) / len(f1scores)


def print_evaluation_metrics(true_labels, predicted_labels, dialog_acts_counter, name):
	print(f"{name} evaluation metrics")
	print(f"    Prediction Accuracy: {calculate_accuracy(true_labels, predicted_labels)}")
	print(f"          Mean F1-score: {calculate_multiclassf1score(true_labels, predicted_labels, dialog_acts_counter, weighted=False)}")
	print(f"      Weighted F1-score: {calculate_multiclassf1score(true_labels, predicted_labels, dialog_acts_counter, weighted=True)}")


def majority_classifier(data, dataset):
	majority_class = max(data.dialog_acts_counter.items(), key=operator.itemgetter(1))[0]  # Here it returns the dialogue act that occurs the most times, in this case "inform"
	# if not user_input:
	# dataset = data.original_data[data.training_part_length:]
	predictions = [majority_class for _ in range(len(dataset))]
	# print_evaluation_metrics([s[0] for s in dataset], predictions, data.dialog_acts_counter, "Majority Classifier")
	return predictions


def rule_based(data, dataset):
	# This is a dictionary with values as the dialogue act and keys as the text to be looked for
	# (example: if sentance contains 'is there' we classify it as reqalts dialogue act)
	prediction_dict = {"bye": "bye", "goodbye": "bye", "thank you": "thankyou", "how about": "reqalts", "is there": "reqalts", "what": "request", "is it": "confirm", "i": "inform", "no": "negate", "yes": "affirm", "hello": "hello", "im": "inform"}
	# if not user_input:
	# dataset = data.original_data[data.training_part_length:]
	predictions = []
	for sentence in dataset:
		p = ""
		for key, prediction in prediction_dict.items():
			if key in sentence:
				p = prediction
				break
		predictions.append(p)
	return predictions


def decision_tree(data, dataset):  # https://scikit-learn.org/stable/modules/tree.html
	clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=30)  # the max depth can be set imperically, but if we set it too big there will be overfitting
	# I set criterion as entropy and split as best, so hopefully it will split on inform class
	# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

	clf.fit(data.vectorized_training_data, data.training_labels)  # We train our tree
	# tree.plot_tree(clf, fontsize=5)  # This will plot the graph if you uncomment it
	# plt.show()
	results = [r for r in clf.predict(dataset)]
	return results


def ff_nn(data, dataset):  # feed forward neural network https://scikit-learn.org/stable/modules/neural_networks_supervised.html
	clf = MLPClassifier(solver='adam', alpha=0.001, random_state=1, early_stopping=False)  # will stop early if small validation subset isnt improving while training
	clf.fit(data.vectorized_training_data, data.training_labels)  # takes around a minute or so, depending on your pc

	results = [r for r in clf.predict(dataset)]  # Accuracy is 0.9866 on validation sets
	return results


def sto_gr_des(data, dataset):  # stochastic gradient descent https://scikit-learn.org/stable/modules/sgd.html
	clf = SGDClassifier(loss="modified_huber", penalty="l2", max_iter=20, early_stopping=False)  # requires a mix_iter (maximum of iterations) of at least 7
	# loss could be different loss-functions that measures models fits. I chose modified_huber (smoothed hinge-loss) since it leads to the highest accuracy (could be changed with regards to other eval-methods)
	# penalty penalizes model complexity
	clf.fit(data.vectorized_training_data, data.training_labels)
	# used the same set-up as decision trees & feed forward neural network
	results = [r for r in clf.predict(dataset)]  # accuracy of ~97%
	return results

def comparison_evaluation(data):
	predictions = {
		"majority": majority_classifier(data, data.test_corpus),
		"rulebased": rule_based(data, data.test_corpus),
		"decisiontree": decision_tree(data, data.vectorized_test_data),
		"neuralnet": ff_nn(data, data.vectorized_test_data),
		"sgradientdescent": sto_gr_des(data, data.vectorized_test_data)}
	labels = [lb for lb in data.dialog_acts_counter]
	true_labels = [label for label in data.test_labels]
	metrics = {
		"precision": lambda t, p: calculate_precision(t, p),
		"recall": lambda t, p: calculate_recall(t, p),
		"f1score": lambda t, p: calculate_f1score(t, p)}
	evaluations = {}
	for metric in metrics:
		evaluations[metric] = {}
		for label in labels:
			binary_true = [tl == label for tl in true_labels]
			evaluations[metric][label] = {}
			for classifier in predictions:
				binary_pred = [pl == label for pl in predictions[classifier]]
				evaluations[metric][label][classifier] = metrics[metric](binary_true, binary_pred)
	fig, axes = plt.subplots(len(evaluations), 1, sharex="all", sharey="all")
	barwidth = 1 / (len(predictions) + 1)
	numbered = [i for i in range(len(labels))]
	for i, metric in enumerate(evaluations):
		axes[i].set_title(metric)
		for j, classifier in enumerate(predictions):
			x_offset = -0.5 * len(predictions) * barwidth + j * barwidth
			axes[i].bar([n + x_offset for n in numbered], [evaluations[metric][lb][classifier] for lb in labels], barwidth, label=classifier)
		axes[i].set_xticks(numbered)
		axes[i].set_xticklabels(labels)
	axes[0].legend(loc=4)
	plt.show()
	fig.set_size_inches(18.5, 10.5)
	fig.savefig('metric_plot', dpi=150)


def interact(data, classifier, vectorize = True):
	while True:
		test_text = input("Please input a sentence: ").lower()
		if vectorize:
			predicted_label = classifier(data, data.vectorizer.transform([test_text]))
		else:
			predicted_label = classifier(data, [test_text])
		print(f"The sentence (\"{test_text}\") is classified as: {predicted_label}")
		test_text = input("Enter 0 to exit, anything else to continue")
		if str(test_text) == "0":
			break


def analyse_test(data, classifier, vectorize = True):
	if vectorize:
		predictions = classifier(data, data.vectorized_test_data)
	else:
		predictions = classifier(data, data.test_corpus)
	print_evaluation_metrics(data.test_labels, predictions, data.dialog_acts_counter, str(classifier.__name__))


def main():
	data_elements = DataElements("dialog_acts.dat")
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
		test_text = input()
		command = str(test_text)
		if command == "0":
			break
		elif command == "1":
			analyse_test(data_elements, majority_classifier, vectorize=False)
		elif command == "2":
			analyse_test(data_elements, rule_based, vectorize=False)
		elif command == "3":
			analyse_test(data_elements, decision_tree)
		elif command == "4":
			analyse_test(data_elements, ff_nn)
		elif command == "5":
			analyse_test(data_elements, sto_gr_des)
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

	# https://uu.blackboard.com/webapps/assignment/uploadAssignment?content_id=_3578249_1&course_id=_128324_1&group_id=&mode=view


if __name__ == "__main__":
	main()
