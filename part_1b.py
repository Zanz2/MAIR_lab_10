import Levenshtein  # this will give you an error if you dont have it installed
from part_1a import *


# define dialog data as a class and parse
class DialogData:
    def __init__(self, filename):
        self.filename = filename
        self.dialogs = self.__parse_data()

    # split data into list of lists, by reading until the dotted-line divider and then appending 
    # import all lines into 'dialogs' and return
    def __parse_data(self):
        original_data = []
        with open(self.filename) as f:
            content = f.readlines()
            for line in content:
                # read the data into a list with an element for each line of text (and lowercase it).
                line = line.rstrip("\n").lower()
                original_data.append(line)
        splitted_data = [[]]
        for line in original_data:
            if line[:5] == "-----":
                splitted_data.append([])
            else:
                splitted_data[-1].append(line)
        if len(splitted_data[-1]) == 0:
            splitted_data = splitted_data[:-1]
        dialogs = []
        for dialog_lines in splitted_data:
            dialogs.append(Dialog(dialog_lines))
        return dialogs


# seperate class to categorise into task number, turn index, session id
# actual lines (following user:, speech act: , and system:) are added to turn_data
class Dialog:
    def __init__(self, dialog_lines):
        self.turns = []
        turn_data = []
        for line in dialog_lines:
            if line[:10] == "session id":
                self.session_id = line[12:]
            elif line[:4] == "task":
                self.task_no = int(line[6:10])
                self.task = line[12:]
            elif line[:10] == "turn index":
                turn_data = []
            elif line[:6] == "system" or line[:4] == "user" or line[:10] == "speech act":
                turn_data.append(line)
                if line[:10] == "speech act":
                    self.turns.append(DialogTurn(turn_data))
            else:
                raise NotImplementedError()
    
    # define session id, task no, task and turns
    def __str__(self):
        return f"{{'session_id': '{self.session_id}', 'task_no': {self.task_no}, 'task': '{self.task}', " \
               f"'turns': [{', '.join(str(dt) for dt in self.turns)}]}}"


# a class with all the turns of speech, so the 3 lines which comprise a turn (system ask, user response, classification)
class DialogTurn:
    def __init__(self, turn_lines):
        assert(len(turn_lines) == 3)
        assert(turn_lines[0][:6] == "system")
        assert(turn_lines[1][:4] == "user")
        assert(turn_lines[2][:10] == "speech act")
        self.system = turn_lines[0][8:]
        self.user = turn_lines[1][6:]
        self.speech_acts = []
        for act in turn_lines[2][12:].split("|"):
            self.speech_acts.append(SpeechAct(act))

    # define f string with system, user and speech act
    def __str__(self):
        speech_acts = f"[{', '.join(str(sa) for sa in self.speech_acts)}]"
        return f"{{'system': '{self.system}', 'user': '{self.user}', 'speech_acts: {speech_acts}"


# we define a class of speech acts (for example inform and request) in which we split the label from the parameters
# then for inform, we split the key (such as area) from the value (such as south) and add them to the dictionary
# for request we generate numbered keys and add those to the values
class SpeechAct:
    def __init__(self, raw_text):
        self.act = raw_text.split("(")[0]
        self.parameters = {}
        raw_parameters = raw_text.split("(")[1].replace(")", "")
        for key_value in raw_parameters.split(","):
            if key_value != "":
                if "=" in key_value:
                    key, value = key_value.split("=")
                    self.parameters[key] = value
                else:
                    key = f"unnamed{len(self.parameters)}"
                    self.parameters[key] = key_value
    
    # assign print strings for readability
    def __str__(self):
        parameters = f"{{{', '.join(k + ': ' + v for k, v in self.parameters.items())}}}"
        return f"{{'act': '{self.act}', 'parameters': {parameters}}}"


# here we load the restaurant info and create a list with the restaurant data (including name and pricerange etc)
class RestaurantInfo:
    def __init__(self, filename):
        self.filename = filename
        self.restaurants = self.__parse_data()

    def __parse_data(self):
        restaurants = []
        with open(self.filename) as f:
            content = f.readlines()
            for line in content:
                # read the data into a list with an element for each line of text (and lowercase it).
                line = line.rstrip("\n").lower()
                columns = [c.strip('"') for c in line.split(",")]
                name, price, area, food, phone, address, postcode = columns
                restaurants.append(Restaurant(name, price, area, food, phone, address, postcode))
        return restaurants


# here we define a class for each restaurant containing the relevant information, naming them accordingly
class Restaurant:
    def __init__(self, name, price, area, food, phone, address, postcode):
        self.name = name
        self.price = price
        self.area = area
        self.food = food
        self.phone = phone
        self.address = address
        self.postcode = postcode

    def __str__(self):
        return f"{{'name': '{self.name}', 'price': '{self.price}', 'area': '{self.area}', 'food': '{self.food}', " \
               f"'phone': '{self.phone}', 'address': '{self.address}', 'postcode': '{self.postcode}'}}"


# define system states
class State:
    HELLO = 0
    ASK = 1
    ASK_PREF_1 = 2
    ASK_PREF_2 = 3
    ASK_PREF_3 = 4
    VALID = 5
    SUGGEST = 6


# this class defines what the system is to do for any given state, then handles the user input by looking for keywords
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


# define dialogue by referring back to the functions to handle user input and then classified state
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


# load data
data = DialogData("all_dialogs.txt")
rests = RestaurantInfo("restaurant_info.csv")
