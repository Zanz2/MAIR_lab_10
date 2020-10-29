import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from matplotlib.pyplot import figure
from survey_analysis import *

def rnd(num):
    return round(num,3)

def get_graph_dictionary(survey_results_file):
    survey_output = Survey(survey_results_file)

    output_dictionary = {
        "a": {
            "first": [0.,0.,0.,0.,0.,0.,0.,0.],
            "second": [0.,0.,0.,0.,0.,0.,0.,0.]
        },
        "b": {
            "first": [0.,0.,0.,0.,0.,0.,0.,0.],
            "second": [0.,0.,0.,0.,0.,0.,0.,0.]
        },
        "non_implicit_responses": [[], [], [], [], [], [], [], []],
        "implicit_responses": [[], [], [], [], [], [], [], []]
    }
    group_count = {
        "a": 0,
        "b": 0
    }
    # 8 questions first and second
    number_of_participants = len(survey_output.entries)
    for question_index in range(8): # get total cumulative score in the 4 arrays above
        for participant_index in range(number_of_participants):
            first_task_question_response = survey_output.entries[participant_index].first.get_data_array()[question_index]
            second_task_question_response = survey_output.entries[participant_index].second.get_data_array()[question_index]
            group = survey_output.entries[participant_index].group
            output_dictionary[group]["first"][question_index] += first_task_question_response
            output_dictionary[group]["second"][question_index] += second_task_question_response
            if group == "a":
                output_dictionary["non_implicit_responses"][question_index].append(first_task_question_response)
                output_dictionary["implicit_responses"][question_index].append(second_task_question_response)
            else:
                output_dictionary["non_implicit_responses"][question_index].append(second_task_question_response)
                output_dictionary["implicit_responses"][question_index].append(first_task_question_response)

    for entry in survey_output.entries:
        group_count[entry.group] += 1

    for group in ["a", "b"]: # divide the 4 arrays by the number of responses (get mean)
        for question_index in range(8):
            output_dictionary[group]["first"][question_index] = round(output_dictionary[group]["first"][question_index] / group_count[group], 2)
            output_dictionary[group]["second"][question_index] = round(output_dictionary[group]["second"][question_index] / group_count[group], 2)
    return output_dictionary



def plot_bar_averages(graph_dictionary, save_file = False):
    #A first task w/o
    #B second task w/o
    #A second task w
    #B first task w
    result_array = {}
    for question_index in range(8): # for visualizing the means over groups
        result_array[question_index] = []
        result_array[question_index].append(graph_dictionary["a"]["first"][question_index])
        result_array[question_index].append(graph_dictionary["b"]["second"][question_index])
        result_array[question_index].append(graph_dictionary["a"]["second"][question_index])
        result_array[question_index].append(graph_dictionary["b"]["first"][question_index])
        result_array[question_index].append(0)

    sums = [0,0,0,0]
    for question_index in range(8): #final 4 bars show combined averages
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

    non_flat_array = [result for key, result in result_array.items()]
    x_values = [item for sublist in non_flat_array for item in sublist]

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
    x_task_label_numbers = np.arange(0, 44, 1)

    plt.bar(x_task_label_numbers,x_values,color=["orange","purple","blue","red","black"])
    plt.ylim(0, 5)
    x_tick_list = list(np.arange(2, 41, 5))
    x_tick_list.append(43)
    plt.xticks(x_tick_list, x_task_label_minified)
    plt.tick_params(axis='x', which='major', labelsize=8, rotation="auto")
    plt.figtext(0.3, 0.90, "Average question score per group", wrap=True, horizontalalignment='center', fontsize=10)
    plt.gcf().autofmt_xdate(rotation="vertical")
    plt.grid(axis='x', alpha=0.2)
    if save_file: plt.savefig("images/survey_group_averages.png", dpi=300)
    plt.show()

def plot_error_bars(graph_dictionary, save_file=False):
    a_1_mean = np.mean(graph_dictionary["a"]["first"])
    b_2_mean = np.mean(graph_dictionary["b"]["second"])
    a_2_mean = np.mean(graph_dictionary["a"]["second"])
    b_1_mean = np.mean(graph_dictionary["b"]["first"])

    a_1_std = np.std(graph_dictionary["a"]["first"])
    b_2_std = np.std(graph_dictionary["b"]["second"])
    a_2_std = np.std(graph_dictionary["a"]["second"])
    b_1_std = np.std(graph_dictionary["b"]["first"])

    labels = ["Group A\n(w/o implicit conf.)\nMean: {}\nSigma: {}".format(rnd(a_1_mean),rnd(a_1_std)),
              "Group B\n(w/o implicit conf.)\nMean: {}\nSigma: {}".format(rnd(b_2_mean),rnd(b_2_std)),
              "Group A\n(with implicit conf.)\nMean: {}\nSigma: {}".format(rnd(a_2_mean),rnd(a_2_std)),
              "Group B\n(with implicit conf.)\nMean: {}\nSigma: {}".format(rnd(b_1_mean),rnd(b_1_std))
              ]
    x_pos = np.arange(len(labels))
    CTEs = [a_1_mean,b_2_mean,a_2_mean,b_1_mean]
    error = [a_1_std,b_2_std,a_2_std,b_1_std]

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs,
           yerr=error,
           align='center',
           alpha=0.5,
           ecolor='black',
           capsize=10)
    ax.set_ylabel('')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels,rotation="0")
    ax.set_title('Average response scores on all questions')
    ax.yaxis.grid(True)
    ax.set_ylim(0, 5)

    # Save the figure and show
    plt.tight_layout()
    if save_file: plt.savefig('images/bar_plot_with_error_bars.png', dpi=300)
    plt.show()

def plot_question_score_responses(grap_dictionary, save_file = False, condensed = False):
    if not condensed:
        fig, ax = plt.subplots(2,8,figsize=(18,8))
        for i in range(16):
            main_index = 0
            if i<8:
                data = graph_dictionary["non_implicit_responses"][i]
            else:
                data = graph_dictionary["implicit_responses"][i % 8]
                main_index = 1

            ax[main_index][i % 8].set_xticks(np.arange(1,6))
            ax[main_index][i % 8].set_xticklabels(np.arange(1,6),fontsize=18)
            ax[main_index][i%8].set_xlim(1,5)
            #ax[main_index][i % 8].set_yticklabels(np.arange(0, 15), fontsize=18)
            ax[main_index][i % 8].tick_params(axis='y', labelsize=18)
            ax[main_index][i % 8].set_ylim(0, 15)
            if i < 8:
                ax[main_index][i % 8].set_title('Question {}'.format(i%8+1),{"fontsize":21})
            else:
                ax[main_index][i % 8].set_xlabel("Likert score",fontsize=21)
            ax[main_index][i%8].hist(data,bins=5,edgecolor="black")

        ax[0][0].set_ylabel("# of responses", fontsize=21)
        ax[1][0].set_ylabel("# of responses", fontsize=21)
        fig.suptitle("Number of scores (1-5) per question. No implicit confirmation above,\n with implicit confirmation below ",fontsize=21)
        plt.tight_layout()
        if save_file: plt.savefig('images/hist_question_score_responses.png', dpi=100)

    else:
        fig, ax = plt.subplots(2)
        without_implicit = ""
        with_implicit = ""

        data1_nonflat = graph_dictionary["non_implicit_responses"]
        data2_nonflat = graph_dictionary["implicit_responses"]
        data1 = [item for sublist in data1_nonflat for item in sublist]
        data2 = [item for sublist in data2_nonflat for item in sublist]

        ax[0].set_xticks(np.arange(1,6))
        ax[0].set_xticklabels(np.arange(1, 6), fontsize=12)
        ax[0].set_xlim(1, 5)
        ax[0].set_ylim(0, 0.5)

        ax[1].set_xticks(np.arange(1, 6))
        ax[1].set_xticklabels(np.arange(1, 6), fontsize=12)
        ax[1].set_xlim(1, 5)
        ax[1].set_ylim(0, 0.5)

        n, bins, patches = ax[0].hist(data1,bins=5,edgecolor="black",density=1)
        mean = np.mean(data1)
        std = np.std(data1)
        median = np.median(data1)
        vari = np.var(data1)
        without_implicit += "Median: {}, Mean: {}, Standard deviation: {}, Variance: {}".format(median,rnd(mean),rnd(std),rnd(vari))
        y = ((1 / (np.sqrt(2 * np.pi) * std)) *
             np.exp(-0.5 * (1 / std * (bins - mean)) ** 2))
        ax[0].plot(bins,y, "--")
        ax[0].set_ylabel("Probability of response")
        ax[0].set_xlabel("Likert score\n"+without_implicit)

        n, bins, patches = ax[1].hist(data2, bins=5, edgecolor="black", density=1)
        mean = np.mean(data2)
        std = np.std(data2)
        median = np.median(data2)
        vari = np.var(data2)
        with_implicit += "Median: {}, Mean: {}, Standard deviation: {}, Variance: {}".format(median,rnd(mean),rnd(std),rnd(vari))
        y2 = ((1 / (np.sqrt(2 * np.pi) * std)) *
             np.exp(-0.5 * (1 / std * (bins - mean)) ** 2))
        ax[1].plot(bins, y2, "--")
        ax[1].set_ylabel("Probability of response")
        ax[1].set_xlabel("Likert score\n"+with_implicit)

        fig.suptitle("Number of scores (1-5) across all questions. No implicit confirmation above,\n with implicit confirmation below ")
        fig.tight_layout()
        if save_file: plt.savefig('images/hist_question_score_responses_condensed.png', dpi=300)

    plt.show()


save_stuff = True
graph_dictionary = get_graph_dictionary("participant_survey.csv")
plot_bar_averages(graph_dictionary, save_file=save_stuff)
plot_error_bars(graph_dictionary, save_file=save_stuff)
plot_question_score_responses(graph_dictionary, save_file=save_stuff, condensed=False)
plot_question_score_responses(graph_dictionary, save_file=save_stuff, condensed=True)