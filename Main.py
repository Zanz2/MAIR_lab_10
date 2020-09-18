from part_1b import *


# load dialog_acts, show options of interaction and display to user, process user request
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
		elif command == "d":
			dialogue_state = DialogueState()
			dialogue_state.current_message()
			user_text = input()
			dialogue(data_elements, dialogue_state, user_text)
		else:
			break
			

if __name__ == "__main__":
	main()
