import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from survey_analysis import *

#Create a dictionary containing POS tags as keys and their frequency as values
# question_dictionary = {
# "level1_likert_scale": total count of this answer,
# "level2_likert_scale": total count of this answer,
# "level3_likert_scale": total count of this answer,
# "level4_likert_scale": total count of this answer,
# "level5_likert_scale": total count of this answer
#}
#- for graph 4 groups are
#A first task
#B second task
#A second task
#B first task
# each bar is average for that question
#x_bucket = [first_question_average,second_question_average_third_question_average,...]
survey_output = Survey("participant_survey.csv")

graph_dictionary = {
    "a": {
        "first": [0.,0.,0.,0.,0.,0.,0.,0.],
        "second": [0.,0.,0.,0.,0.,0.,0.,0.]
    },
    "b": {
        "first": [0.,0.,0.,0.,0.,0.,0.,0.],
        "second": [0.,0.,0.,0.,0.,0.,0.,0.]
    }
}
group_count = {
    "a": 0,
    "b": 0
}
# 8 questions first and second
number_of_participants = len(survey_output.entries)
for question_index in range(8):
    for participant_index in range(number_of_participants):
        first_task_question_response = survey_output.entries[participant_index].first.get_data_array()[question_index]
        second_task_question_response = survey_output.entries[participant_index].second.get_data_array()[question_index]
        group = survey_output.entries[participant_index].group
        graph_dictionary[group]["first"][question_index] += first_task_question_response
        graph_dictionary[group]["second"][question_index] += second_task_question_response

for entry in survey_output.entries:
    group_count[entry.group] += 1

for group in ["a", "b"]:
    for question_index in range(8):
        graph_dictionary[group]["first"][question_index] = round(graph_dictionary[group]["first"][question_index] / group_count[group],2)
        graph_dictionary[group]["second"][question_index] = round(graph_dictionary[group]["second"][question_index] / group_count[group],2)

#A first task
#B second task
#A second task
#B first task
result_array = {}
x_task_label = []
for question_index in range(8):
    result_array[question_index] = []
    result_array[question_index].append(graph_dictionary["a"]["first"][question_index])
    result_array[question_index].append(graph_dictionary["b"]["second"][question_index])
    result_array[question_index].append(graph_dictionary["a"]["second"][question_index])
    result_array[question_index].append(graph_dictionary["b"]["first"][question_index])
    result_array[question_index].append(0)


    x_task_label.append("Q{} A Task 1".format(question_index+1))
    x_task_label.append("Q{} B Task 2".format(question_index+1))
    x_task_label.append("Q{} A Task 2".format(question_index+1))
    x_task_label.append("Q{} B Task 1".format(question_index+1))
    x_task_label.append("-{}-".format(question_index + 1))

sums = [0,0,0,0]
for question_index in range(8):
    sums[0] += graph_dictionary["a"]["first"][question_index]
    sums[1] += graph_dictionary["b"]["second"][question_index]
    sums[2] += graph_dictionary["a"]["second"][question_index]
    sums[3] += graph_dictionary["b"]["first"][question_index]

new_max = len(result_array)
result_array[new_max] = []
result_array[new_max].append(sums[0]/(new_max))
result_array[new_max].append(sums[1]/(new_max))
result_array[new_max].append(sums[2]/(new_max))
result_array[new_max].append(sums[3]/(new_max))

x_task_label.append("Group A Task 1 combined average")
x_task_label.append("Group B Task 2 combined average")
x_task_label.append("Group A Task 2 combined average")
x_task_label.append("Group B Task 1 combined average")
#print(result_array)
#x_question_label = ["question {}".format(number + 1) for number in result_array]

non_flat_array = [result for key, result in result_array.items()]
x_values = [item for sublist in non_flat_array for item in sublist]


print(len(x_values))
print(len(x_task_label))
orange_patch = mpatches.Patch(color='orange', label='Group A, 1st task')
purple_patch = mpatches.Patch(color='purple', label='Group B, 2nd task')
blue_patch = mpatches.Patch(color='blue', label='Group A, 2nd task')
red_patch = mpatches.Patch(color='red', label='Group B, 1.st task')
#green_patch = mpatches.Patch(color='green', label='Overall average user satisfaction')
null_patch = mpatches.Patch(color='black', label="")

plt.legend(handles=[orange_patch,purple_patch,blue_patch,red_patch],loc=1,fontsize=8)
x_task_label_minified = [
    "Question 1",
    "Question 2",
    "Question 3",
    "Question 4",
    "Question 5",
    "Question 6",
    "Question 7",
    "Question 8",
    "Combined \naverages\n over groups\nand tasks",
]
x_task_label_numbers = np.arange(0,44,1)
print(len(x_task_label_minified))
print(len(x_task_label))

#plt.locator_params(axis = 'x', nbins = 9)
plt.bar(x_task_label_numbers,x_values,color=["orange","purple","blue","red","black"])
plt.ylim(0, 5)
x_tick_list = list(np.arange(2,41,5))
x_tick_list.append(43)
plt.xticks(x_tick_list,x_task_label_minified)
#plt.xticks(np.arange(1, len(x_values), 1))
plt.tick_params(axis='x', which='major', labelsize=8, rotation="auto")
plt.figtext(0.3, 0.90, "Average question score per group", wrap=True, horizontalalignment='center', fontsize=10)
plt.gcf().autofmt_xdate(rotation="vertical")
#plt.ylabel(labels)
plt.grid(axis='x', alpha=0.2)
plt.savefig("surveyplot.png")
plt.show()
