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


def calculate_accuracy(true_labels, predicted_labels):
	length = len(true_labels)
	assert(len(predicted_labels) == length)
	return sum(true_labels[i] == predicted_labels[i] for i in range(length)) / length


def majority_classifier(dialog_acts_counter, dataset=None):
	majority_class = max(dialog_acts_counter.items(), key=operator.itemgetter(1))[0]  # Here it returns the dialogue act that occurs the most times, in this case "inform"
	if dataset:
		predictions = [majority_class for _ in range(len(dataset))]
		print(f"Prediction accuracy: {calculate_accuracy([s[0] for s in dataset], predictions)}")
	else:
		while True:
			test_text = input("Please input a sentence: ")
			print(f"we classify this sentence (\"{test_text}\") as: {majority_class}")
			test_text = input("Enter 0 to exit, anything else to continue")
			if str(test_text) == "0":
				break


def rule_based(dataset=None):
	# This is a dictionary with values as the dialogue act and keys as the text to be looked for
	# (example: if sentance contains 'is there' we classify it as reqalts dialogue act)
	prediction_dict = {"bye": "bye", "goodbye": "bye", "thank you": "thankyou", "how about": "reqalts", "is there": "reqalts", "what": "request", "is it": "confirm", "i": "inform", "no": "negate", "yes": "affirm", "hello": "hello", "im": "inform"}
	if dataset:
		predictions = []
		for full_sentence in dataset:
			p = ""
			for key, prediction in prediction_dict.items():
				if key in full_sentence[1]:
					p = prediction
					break
			predictions.append(p)
		print(f"Prediction accuracy: {calculate_accuracy([s[0] for s in dataset], predictions)}")
	else:
		while True:
			test_text = input("Please input a sentence: ")
			sentence = str(test_text)
			found = False
			for key, prediction in prediction_dict.items():
				if key in sentence:
					print(f"we classify this sentence as: {prediction}")
					found = True
					break

			if not found:
				print("We could not classify this sentence")
			test_text = input("Enter 0 to exit, anything else to continue")
			if str(test_text) == "0":
				break


def get_class_label(correct_classes_mapping, target_class_id):
	for class_label, class_id in correct_classes_mapping.items():
		if class_id == target_class_id:
			return class_label


def decision_tree(vectorizer, correct_classes_mapping, dataset, assigned_classes, test_dataset=None, test_classes=None):  # https://scikit-learn.org/stable/modules/tree.html
	clf = tree.DecisionTreeClassifier(criterion="entropy", splitter="best", max_depth=10)  # the max depth can be set imperically, but if we set it too big there will be overfitting
	# I set criterion as entropy and split as best, so hopefully it will split on inform class
	# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

	# !IMPORTANT! the model might be overfitting to the data, so we can discuss this
	# It does get 86% accuracy on validation sets

	clf.fit(dataset, assigned_classes)  # We train our tree
	if test_dataset is not None and test_classes is not None:
		# tree.plot_tree(clf, fontsize=5)  # This will plot the graph if you uncomment it
		# plt.show()
		results = clf.predict(test_dataset)
		print(f"Decision tree accuracy on test data: {calculate_accuracy(results, test_classes)}")
	else:
		while True:
			test_text = input("Please input a sentence: ")
			sentence = vectorizer.transform([str(test_text)])
			results = clf.predict(sentence)
			print(f"The sentence (\"{test_text}\") is classified as: {get_class_label(correct_classes_mapping, results[0])}")
			test_text = input("Enter 0 to exit, anything else to continue")
			if str(test_text) == "0":
				break


def ff_nn(vectorizer, dataset, assigned_classes, test_dataset=None, test_classes=None):  # feed forward neural network https://scikit-learn.org/stable/modules/neural_networks_supervised.html
	clf = MLPClassifier(solver='adam', alpha=1e-5, random_state=1, early_stopping=False)  # will stop early if small validation subset isnt improving while training
	clf.fit(dataset, assigned_classes)  # takes around a minute or so, depending on your pc
	# if its taking too long on your pc, add this to the function parameters above: hidden_layer_sizes=(5, 2)

	if test_dataset is not None and test_classes is not None:
		results = clf.predict(test_dataset)  # Accuracy is 0.9866 on validation sets
		print(f"Feed forward neural network accuracy on test data: {calculate_accuracy(results, test_classes)}")
	else:
		while True:
			test_text = input("Please input a sentence: ")
			sentence = vectorizer.transform([str(test_text)])
			print(f"The sentence (\"{test_text}\") is classified as: {clf.predict(sentence)}")
			test_text = input("Enter 0 to exit, anything else to continue")
			if str(test_text) == "0":
				break


def sto_gr_des(vectorizer, dataset, assigned_classes, test_dataset=None, test_classes=None):  # stochastic gradient descent https://scikit-learn.org/stable/modules/sgd.html
	clf = SGDClassifier(loss="modified_huber", penalty="l2", max_iter=20, early_stopping=False)  # requires a mix_iter (maximum of iterations) of at least 7
	# loss could be different loss-functions that measures models fits. I chose modified_huber (smoothed hinge-loss) since it leads to the highest accuracy (could be changed with regards to other eval-methods)
	# penalty penalizes model complexity
	clf.fit(dataset, assigned_classes)
	# used the same set-up as decision trees & feed forward neural network

	if test_dataset is not None and test_classes is not None:
		results = clf.predict(test_dataset)  # accuracy of ~97%
		print(f"Stochastic gradient descent accuracy on test data: {calculate_accuracy(results, test_classes)}")
	else:
		while True:
			test_text = input("Please input a sentence: ")
			sentence = vectorizer.transform([str(test_text)])
			print(f"The sentence is classified as: {clf.predict(sentence)}")
			test_text = input("Enter 0 to exit, anything else to continue")
			if str(test_text) == "0":
				break
	# still misses command


def main():
	filename = "dialog_acts.dat"
	# Note that this dataset contains a simplification compared to the original data. In case an utterance was labeled with two different dialog acts,
	# only the first dialog act is used as a label. When performing error analysis (see below) this is a possible aspect to take into account.

	line_array = []
	with open(filename) as f:
		content = f.readlines()
		for line in content:  # I open the file, read its contents line by line, remove the trailing newline character, and convert them to lowercase
			line = line.rstrip("\n").lower()
			line = line.split(" ", 1)  # this just seperates te first dialog_act from the remaining sentence
			line_array.append(line)

	arr_length = (len(line_array))  # This is the number of sentences in the entire file
	training_part_length = int(arr_length*0.85)  # Here is the number that will be in the Training split

	# random.shuffle(line_array) # !IMPORTANT! Would randomize the contents of test and training array, we might need commands like this later for cross validation
	dialog_acts_counter = {}
	for element in line_array[:training_part_length]:  # Here I find all the unique dialogue act categories, and count how many occurences there are for each category
		if element[0] not in dialog_acts_counter:
			dialog_acts_counter[element[0]] = 0
		dialog_acts_counter[element[0]] += 1

	corpus = []
	correct_classes_mapping = {}
	correct_classes = []
	unique_counter = 0
	for full_sentence in line_array:
		corpus.append(full_sentence[1])  # This will make an array with just sentences, no classes
		if full_sentence[0] in correct_classes_mapping:  # This changes our predicted classes to unique integers, because thats what the documentation says to do
			correct_classes.append(correct_classes_mapping[full_sentence[0]])
		else:
			correct_classes_mapping[full_sentence[0]] = unique_counter
			correct_classes.append(correct_classes_mapping[full_sentence[0]])
			unique_counter += 1

	correct_labels = np.array(line_array)
	correct_labels = correct_labels[:, :1].flatten()  # Extracts only the first column from the array (class labels)

	training_corpus = corpus[:training_part_length]
	training_classes = correct_classes[:training_part_length]  # Classes are just ids, the id represents a class label from the mapping
	training_labels = correct_labels[:training_part_length]

	test_corpus = corpus[training_part_length:]
	test_classes = correct_classes[training_part_length:]
	test_labels = correct_labels[training_part_length:]

	vectorizer = TfidfVectorizer()  # This will change our sentences into vectors of words, will calculate occurances and also normalize them
	# more: https://scikit-learn.org/stable/modules/feature_extraction.html)
	# vectorizer_bigram = TfidfVectorizer(ngram_range=(1, 2),token_pattern=r'\b\w+\b', min_df=1)
	# print(len(corpus))
	# print(len(correct_classes)) # This should match, every sentence in corpus should have a predicted class

	# https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/
	vectorizer.fit(corpus)  # Makes a sparse word matrix out of the entire word corpus (vectorizes it)

	vectorized_training_data = vectorizer.transform(training_corpus)  # transforms the models so they use that matrix as a representation
	vectorized_test_data = vectorizer.transform(test_corpus)

	# bigram_vectorized_training_data = vectorizer_bigram.fit_transform(training_corpus) # if needed later
	# the first one splits each word, which is called 1 gram, 2 gram (bigram i guess) splitting would be where it would split by 2 words
	# Example ( here i am -> 1 gram ["here", "i", "am"], 2 gram ["here i", "i am"]
	# (these words are changed into unique ids, but just as an example)

	# I played around with the prints below to see how it works and what they contain
	# print(vectorizer.get_feature_names())
	# print(vectorizer.vocabulary_.get('thisworddoesntexist')) # if word is not in vocabulary None is returned

	# print(len(training_array)) # 21675 samples with:
	# print(len(vectorizer.vocabulary_)) # 711 features
	# print(vectorized_training_data.toarray()) # the dimensions of this are 21676 x 711

	# We now have our bag of words

	while True:
		print("Enter")
		print("0 for exit")
		print("1 for Majority classifier on test data")
		print("2 for manual prediction on test data")
		print("3 for Decision tree on test data")
		print("4 for Feed forward neural network on test data")
		print("1i for Majority classifier on user input")
		print("2i for manual prediction on user input")
		print("3i for Decision tree on user input")
		print("4i for Feed forward neural network on user input")

		test_text = input()
		command = str(test_text)
		if command == "0":
			break
		elif command == "1":
			majority_classifier(dialog_acts_counter, line_array[training_part_length:])
		elif command == "2":
			rule_based(line_array[training_part_length:])
		elif command == "3":
			decision_tree(vectorizer, correct_classes_mapping, vectorized_training_data, training_classes, vectorized_test_data, test_classes)
		elif command == "4":
			ff_nn(vectorizer, vectorized_training_data, training_labels, vectorized_test_data, test_labels)
		elif command == "1i":
			majority_classifier(dialog_acts_counter)
		elif command == "2i":
			rule_based()
		elif command == "3i":
			decision_tree(vectorizer, correct_classes_mapping, vectorized_training_data, training_classes)
		elif command == "4i":
			ff_nn(vectorizer, vectorized_training_data, training_labels)
		else:
			break

	# print("Total entries predicted as bye: "+str(counter_total)+", of those: "+str(correct_count)+ " are correct")
	# https://uu.blackboard.com/webapps/assignment/uploadAssignment?content_id=_3578249_1&course_id=_128324_1&group_id=&mode=view
	# Ill get the above into a state where you can input your own sentences and the above 2 systems classify them


if __name__ == "__main__":
	main()
