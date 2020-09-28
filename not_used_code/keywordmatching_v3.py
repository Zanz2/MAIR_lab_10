import re
import Levenshtein as Levenshtein
from part_1a import *
import random as rnd
import re


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
            for i, line in enumerate(content):
                # read the data into a list with an element for each line of text (and lowercase it).
                if i > 0:
                    line = line.rstrip("\n").lower()
                    columns = [c.strip('"') for c in line.split(",")]
                    name, pricerange, area, food, phone, addr, postcode = columns
                    restaurants.append(Restaurant(name, pricerange, area, food, phone, addr, postcode))
        return restaurants


# here we define a class for each restaurant containing the relevant information, naming them accordingly
class Restaurant:
    def __init__(self, name, pricerange, area, food, phone, addr, postcode):
        self.name = name
        self.pricerange = pricerange
        self.area = area
        self.food = food
        self.phone = phone
        self.addr = addr
        self.postcode = postcode

    def __str__(self):
        return f"{{'name': '{self.name}', 'pricerange': '{self.pricerange}', 'area': '{self.area}', 'food': " \
               f"'{self.food}', 'phone': '{self.phone}', 'addr': '{self.addr}', 'postcode': '{self.postcode}'}}"
    
    def get(self, attribute):
        if attribute == "name":
            return self.name
        elif attribute == "pricerange":
            return self.pricerange
        elif attribute == "area":
            return self.area
        elif attribute == "food":
            return self.food
        elif attribute == "phone":
            return self.phone
        elif attribute == "addr":
            return self.addr
        elif attribute == "postcode":
            return self.postcode
        else:
            print(f"Unknown attribute for Restaurant: {attribute}")
            raise NotImplementedError()



# keyword matching + recognizing patterns, still misses levensthein distance
class PatternAndMatch:
    def __init__(self, restaurant_info):
        self.restaurant_info = restaurant_info
        self.preference_values = {
            "food": [],
            "area": [],
            "pricerange": [],
            "phone": ["phone number", "phone", "phonenumber"],
            "addr": ["address"],
            "postcode": ["postcode"]
        }
        
        
        for restaurant in self.restaurant_info.restaurants:
            if restaurant.food not in self.preference_values["food"]:
                self.preference_values["food"].append(restaurant.food)
            if restaurant.area not in self.preference_values["area"]:
                self.preference_values["area"].append(restaurant.area)
            if restaurant.pricerange not in self.preference_values["pricerange"]:
                self.preference_values["pricerange"].append(restaurant.pricerange)

    """ 
          if restaurant.pricerange not in self.preference_info["phone"]:
                self.preference_info["phone"].append(restaurant.phone)
            if restaurant.pricerange not in self.preference_info["addr"]:
                self.preference_info["addr"].append(restaurant.addr)
            if restaurant.pricerange not in self.preference_info["postcode"]:
                self.preference_info["postcode"].append(restaurant.postcode) """

    def check_levenshtein(self, word, type = False):
        # allowed type : "food_type" "price_range" or "location"
        # type is food location or price range, each of these have their own minimum word length
        # based on the keyword matching words, for example if we misspell west as est, we still spellcheck est
        # if we would misspell it as st then we do not consider that
        # for general use the type is just falls, for correcting specific food types then
        # the type is supplied and each type has its own minimum length (see ifs below)
        blacklist = ["west", "would", "want", "world", "a", "part", "can", "what", "that", "the"]
        # words that get confused and changed easily frequently belong in the blacklist
        w_len = len(word)
        if word in blacklist or (type is False and w_len < 3):  # general words are only allowed if they are length 3 and up
            return False
        if (
                type == "food" and w_len < 2) or (
                type == "pricerange" and w_len < 3) or (
                type == "area" and w_len < 2) or (
                type == "phone" and w_len < 3) or (
                type == "addr" and w_len < 4) or (
                type == "postcode" and w_len < 2
        ):
            return False
        match_dict = {
            "correct_word": False,
            "type": False,
            "index": -1,
            "lv_distance": 4  # max allowed distance, if its 4 at the end we return false
        }
        loop_array = ["food", "pricerange", "area","phone", "addr", "postcode"]
        for type_index, value_type in enumerate(loop_array):
            if type is not False and value_type != type:
                continue
            for element in self.preference_values[value_type]:
                lv_distance = Levenshtein.distance(element, word)
                if lv_distance < match_dict["lv_distance"]:
                    match_dict["lv_distance"] = lv_distance
                    match_dict["type"] = value_type
                    match_dict["index"] = type_index
                    match_dict["correct_word"] = element
                    if lv_distance < 3: print(word + " changed into " + element)  # debug that prints the word changes
        if match_dict["lv_distance"] < 3:
            return match_dict["correct_word"]
        return False

    def patterns(self, user_utterance):
        user_text = user_utterance
        words = user_utterance.split(" ")
        pref_dict = {"food": None, "area": None, "pricerange": None}
        """
        # it first looks for patterns in the utterance and fills in the value for food or area if one of these patterns is discovered
        if re.findall(r' serves (.*?) food', user_text):
            pref_dict["food"] = (re.findall(r' serves (.*?) food', user_text))[0]
        if re.findall(r' serving (.*?) food', user_text):
            pref_dict["food"] = (re.findall(r' serving (.*?) food', user_text))[0]
        if re.findall(r' with (.*?) food', user_text):
            pref_dict["food"] = (re.findall(r' with (.*?) food', user_text))[0]
        if re.findall(r' in the (.*?) ', user_text):
            pref_dict["area"] = (re.findall(r' in the (.*?) ', user_text))[0]
            """
        # then it checks per word if it falls in one of the categories
        # if one of the categories already has a value attributed due to a pattern, then this category will not be considered
        for word in words:
            if pref_dict["food"] is None:
                if word in self.preference_values["food"]:
                    pref_dict["food"] = word
                elif self.check_levenshtein(word) in self.preference_values["food"]:
                    pref_dict["food"] = self.check_levenshtein(word)
            if pref_dict["area"] is None:
                if word in self.preference_values["area"]:
                    pref_dict["area"] = word
                elif self.check_levenshtein(word) in self.preference_values["area"]:
                    pref_dict["area"] = self.check_levenshtein(word)
            if pref_dict["pricerange"] is None:
                if word in self.preference_values["pricerange"]:
                    pref_dict["pricerange"] = word
                elif self.check_levenshtein(word) in self.preference_values["pricerange"]:
                    pref_dict["pricerange"] = self.check_levenshtein(word)
        print(pref_dict)
        return pref_dict

    def patterns2(self, user_utterance):
        user_text = user_utterance
        words = user_utterance.split(" ")
        pref_info = {"phone": False, "addr": False, "postcode": False}
        for word in words:
            if word == "phone" or word == "phonenumber":
                pref_info["phone"] = True
            if self.check_levenshtein(word) == "phone" or self.check_levenshtein(word) == "phonenumber":
                pref_info["phone"] = True
            if word == "address":
                pref_info["addr"] = True
            if self.check_levenshtein(word) == "address":
                pref_info["addr"] = True
            if word == "postcode" or word == "post": 
                pref_info["postcode"] = True
            if self.check_levenshtein(word) == "postcode" or self.check_levenshtein(word) == "post":
                pref_info["postcode"] = True
        print(pref_info)
        return

restaurant_info = RestaurantInfo("restaurant_info.csv")
a = PatternAndMatch(restaurant_info)
a.check_levenshtein("phonenumber")
a.patterns("I want hinese food that is moderately priced and is in the orth part of town and what is the phonenumber")
a.patterns2("postco")
