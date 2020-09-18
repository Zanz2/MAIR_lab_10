import Levenshtein  # this will give you an error if you dont have it installed
from part_1a import *


class State:
	HELLO = 0
	ASK = 1
	ASK_PREF_1 = 2
	ASK_PREF_2 = 3
	ASK_PREF_3 = 4
	VALID = 5
	SUGGEST = 6


class DialogueState:  # has dialogue state
	def __init__(self):
		self.current_state = State.HELLO
		self.user_utterances_array = []

	def current_message(self):
		if self.current_state == State.HELLO:
			print("Hi, welcome to the group 10 dialogue system.")
			print("You can ask for restaurants by area , price range or food type . How may I help you?")
		elif self.current_state == State.ASK:
			print("I handled your inform")
		else:
			print("I did not understand what you said, can you rephrase that?")

	def handle_label(self, classified_label, user_utterance):  # has to modify itself according to the sentence contents
		if classified_label == "hello":
			self.current_state = State.HELLO
		elif classified_label == "inform":
			self.current_state = State.ASK
			# [type_of_food, price_range, location]
			keywords = get_keywords(user_utterance)
			# first use check_word() to check if all the supplied keywords are correct
			# then ask them if they are sure and if they want to specify any keywords
			# that might be missing (maybe he only specified price range, ask about other 2)
			# if they are not provided treat them as any (wildcard options)
			# then suggest a restaurant
		else:
			self.current_state = State.INVALID

	def check_word(self, word):
		todo = True
		# use levenshtein here, maybe find a correct word, or ask user to repeat
		# if he made error


def dialogue(data, dialogue_state, user_utterance):
	while True:
		predicted_label = predict_sentence(data, user_utterance, sto_gr_des)[0]
		# returns an array of one, so we select first entry
		if predicted_label == "hello":
			dialogue_state.handle_label(predicted_label, user_utterance)
			break
		elif predicted_label == "inform":
			dialogue_state.handle_label(predicted_label, user_utterance)
			break
		else:
			break
	dialogue_state.current_message()
	return dialogue_state
