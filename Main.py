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
        line = line.split(" ",1) # this just seperates te first dialog_act from the remaining sentance
        line_array.append(line)

arr_length = (len(line_array)) # This is the number of sentances in the entire file
test_part_length = int(arr_length*0.15)+1 # Here is the number that will be in the Testing split (+1 because it rounded down)
training_part_length= int(arr_length*0.85) # Here is the number that will be in the Training split

dialog_acts = []
dialogue_acts_counter = {}

for element in line_array: # Here I find all the unique dialogue act categories, and count how many occurences there are for each category
	if(element[0]) not in dialog_acts:
		dialog_acts.append(element[0])
	if(element[0] in dialogue_acts_counter):
		dialogue_acts_counter[element[0]] += 1
	else:
		dialogue_acts_counter[element[0]] = 0



print(max(dialogue_acts_counter.items(), key=operator.itemgetter(1))[0])
# Here it returns the dialogue act that occurs the most times, in this case "inform"
print("Baseline 1:")
print("The majority classifier is: "+max(dialogue_acts_counter.items(), key=operator.itemgetter(1))[0])

print("Baseline 2:")
correct_count = 0
counter_total = 0
for element in line_array:
	if "goodbye" in element[1] and "bye" == element[0]:
		correct_count += 1
		counter_total += 1
	elif "goodbye" in element[1]:
		counter_total += 1

print("Total entries predicted as bye: "+str(counter_total)+", of those: "+str(correct_count)+ " are correct")

# Ill get the above into a state where you can input your own sentances and the above 2 systems classify them




random.shuffle(line_array)
training_array = line_array[0:training_part_length]
test_array = line_array[training_part_length:(training_part_length+test_part_length)]
