import numpy as np
import pandas as pd
from sklearn import *
import sys
import operator
import random


filename = "dialog_acts.dat"
print(filename)
#Note that this dataset contains a simplification compared to the original data. In case an utterance was labeled with two different dialog acts,
# only the first dialog act is used as a label. When performing error analysis (see below) this is a possible aspect to take into account.

line_array = []
with open(filename) as f:
    content = f.readlines()
    for line in content: # I open the file, read its contents line by line, remove the trailing newline character, and convert them to lowercase
        line = line.rstrip("\n").lower()
        line = line.split(" ", 1) # this just seperates te first dialog_act from the remaining sentance
        line_array.append(line)

arr_length = (len(line_array)) # This is the number of sentances in the entire file
test_part_length = int(arr_length*0.15)+1 # Here is the number that will be in the Testing split (+1 because it rounded down)
training_part_length= int(arr_length*0.85) # Here is the number that will be in the Training split

#random.shuffle(line_array) # Would randomize the contents of test and training array
training_array = line_array[0:training_part_length]
test_array = line_array[training_part_length:(training_part_length+test_part_length)]

def majority_classifier(dataset = None):
	dialog_acts = []
	dialogue_acts_counter = {}

	for element in line_array:  # Here I find all the unique dialogue act categories, and count how many occurences there are for each category
		if (element[0]) not in dialog_acts:
			dialog_acts.append(element[0])
		if (element[0] in dialogue_acts_counter):
			dialogue_acts_counter[element[0]] += 1
		else:
			dialogue_acts_counter[element[0]] = 0

	correct_count = 0
	counter_total = 0
	majority_class = max(dialogue_acts_counter.items(), key=operator.itemgetter(1))[0] # Here it returns the dialogue act that occurs the most times, in this case "inform"
	if(dataset):
		for full_sentence in dataset:
			if(full_sentence[0] == majority_class):
				correct_count += 1
				counter_total += 1
			else:
				counter_total += 1
		print("Prediction accuracy: "+str(correct_count/counter_total))
	else:
		while(True):
			test_text = input("Please input a sentence: ")
			sentence = str(test_text)
			print("we classify this sentance as: "+majority_class)
			test_text = input("Enter 0 to exit, anything else to continue")
			if str(test_text) == "0":
				break

def rule_based(dataset = None):
	# This is a dictionary with values as the dialogue act and keys as the text to be looked for
	#  (example: if sentance contains 'is there' we classify it as reqalts dialogue act)
	prediction_dict = {"bye": "bye","goodbye": "bye", "thank you": "thankyou", "how about": "reqalts", "is there": "reqalts"}
	correct_count = 0
	counter_total = 0
	if (dataset):
		for full_sentence in dataset:
			for key, prediction in prediction_dict.items():
				if key in full_sentence[1]:
					if prediction == full_sentence[0]:
						correct_count += 1
						break
					else:
						break
			counter_total += 1
		print("Prediction accuracy: " + str(correct_count / counter_total))
	else:
		while (True):
			test_text = input("Please input a sentence: ")
			sentence = str(test_text)
			found = False
			for key, prediction in prediction_dict.items():
				if key in sentence:
					print("we classify this sentance as: " + prediction)
					found = True
					break

			if not found: print("We could not classify this sentance")
			test_text = input("Enter 0 to exit, anything else to continue")
			if str(test_text) == "0":
				break

while(True):
	print("Enter")
	print("0 for exit")
	print("1 for Majority classifier on test data")
	print("2 for manual prediction on test data")
	print("1i for Majority classifier on user input")
	print("2i for manual prediction on user input")

	test_text = input()
	command = str(test_text)
	if command == "0":
		break
	elif command == "1":
		majority_classifier(test_array)
	elif command == "2":
		rule_based(test_array)
	elif command == "1i":
		majority_classifier()
	elif command == "2i":
		rule_based()
	else:
		break


#print("Total entries predicted as bye: "+str(counter_total)+", of those: "+str(correct_count)+ " are correct")

#https://uu.blackboard.com/webapps/assignment/uploadAssignment?content_id=_3578249_1&course_id=_128324_1&group_id=&mode=view

# Ill get the above into a state where you can input your own sentences and the above 2 systems classify them





