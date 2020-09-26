import Levenshtein  # this will give you an error if you dont have it installed
from part_1a import *
import random as rnd
import re


# we define a class of speech acts (for example inform and request) in which we split the label from the parameters
# then for inform, we split the key (such as area) from the value (such as south) and add them to the dictionary
# for request we generate numbered keys and add those to the values
class SpeechAct:
    def __init__(self, act, parameters=None):
        self.act = act
        self.parameters = {}
        if type(parameters) == list:
            self.parameters = {f"unnamed{i}": p for i, p in enumerate(parameters)}
        elif type(parameters) == dict:
            self.parameters = parameters
    
    # assign print strings for readability during testing
    def __str__(self):
        parameters = f"{{{', '.join(k + ': ' + v for k, v in self.parameters.items())}}}"
        return f"{{'act': '{self.act}', 'parameters': {parameters}}}"


# here we load the restaurant info and create a list with the restaurant data (including food, area, pricerange etc.)
class RestaurantInfo:
    def __init__(self, filename):
        self.filename = filename
        self.restaurants = self.__parse_data()

    def __parse_data(self):
        restaurants = []
        csv_data = pd.read_csv(self.filename, header=0, na_filter=False)
        for row in csv_data.values:
            restaurantname, pricerange, area, food, phone, addr, postcode = row
            restaurants.append(Restaurant(restaurantname, pricerange, area, food, phone, addr, postcode))
        return restaurants


# here we define a class for each restaurant containing the relevant information, naming them accordingly
class Restaurant:
    def __init__(self, restaurantname, pricerange, area, food, phone, addr, postcode):
        self.items = {
            "restaurantname": restaurantname,
            "pricerange": pricerange,
            "area": area,
            "food": food,
            "phone": phone,
            "addr": addr,
            "postcode": postcode}

    def __str__(self):
        return str(self.items)


class KeywordMatch:
    # keyword matching class
    def __init__(self, restaurant_info):
        self.restaurant_info = restaurant_info
        self.preference_values = {
            "food": [],
            "area": [],
            "pricerange": [],
            "phone": ["phone number", "phone", "phonenumber"],
            "addr": ["address"],
            "postcode": ["postcode", "post"]
        }
        
        # check for the restaurants whether possible preferences have been added, if not add
        for restaurant in self.restaurant_info.restaurants:
            for preference in ["food", "area", "pricerange"]:
                if restaurant.items[preference] not in self.preference_values[preference]:
                    self.preference_values[preference].append(restaurant.items[preference])

    def check_levenshtein(self, word, type = False):
        # allowed type : "food_type", "price_range", "location", "phone", "address" and "postcode"
        # different types have their own minimum word length
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
                    #if lv_distance < 3: print(word + " changed into " + element)  # debug that prints the word changes
        if match_dict["lv_distance"] < 3:
            return match_dict["correct_word"]
        return False

    def keyword_match_pref(self, user_utterance):
        # this method will check whether the user mentions a word that matches with a word from the csv file
        # it also checks words that are misspelled with help of the check_levenshtein method
        # it returns a dictionary with preferences for foodtype, area and pricerange
        
        words = user_utterance.split(" ")
        pref_dict = {"food": None, "area": None, "pricerange": None}

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
        return pref_dict

    def keyword_match_info(self, user_utterance):
        # this method checks whether one of the words is mentioned which indicates a request for information about a certain restaurant
        # it will be a dictionary with types mapped to a Boolean (True/ False)
        # when at least one entry is True, it will be able for the dialog manager to give the user the desired information about the restaurant
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
        return pref_info


class SystemUtterance:
    TEMPLATES = {
        "STATEMENT": {
            "area": "it is in the {0} part of town",
            "pricerange": "the prices are {0}",
            "food": "they serve {0} food"},
        "DESCRIPTION": {
            "area": "in the {0} part of town",
            "pricerange": "in the {0} pricerange",
            "food": "which serve {0} food"},
        "CONFIRMATION": {
            "area": "{0} part",
            "pricerange": "{0} prices",
            "food": "{0} food"},
        "QUESTION": {
            "area": "What part of town do you have in mind?",
            "pricerange": "Would you like something in the cheap, moderate, or expensive price range?",
            "food": "What kind of food would you like?"}}
    
    @classmethod
    def generate_combination(cls, preferences, utterance_type):
        assert(utterance_type in ("STATEMENT", "DESCRIPTION", "CONFIRMATION"))
        sub_sentences = []
        for cat, pref in preferences.items():
            if cat in ("pricerange", "area", "food"):
                if pref not in (None, "dontcare"):
                    sub_sentences.append(cls.TEMPLATES[utterance_type][cat].format(pref))
        if len(sub_sentences) <= 1:
            return "".join(sub_sentences)
        return ", ".join(sub_sentences[:-2] + [f"{sub_sentences[-2]} and {sub_sentences[-1]}"])
    
    @classmethod
    def ask_information(cls, category):
        return cls.TEMPLATES["QUESTION"][category]


class DialogHistory:
    def __init__(self, restaurant_info):
        self.restaurant_info = restaurant_info
        self.preferences = {"pricerange": None, "area": None, "food": None}
        self.declined = []
        self.requests = []
        self.terminate = False
        self.speech_acts = []
        self.last_suggestion = None
        self.last_inquiry = None
    
    def decline(self, restaurant):
        self.declined.append(restaurant)
    
    def set_request(self, request):
        assert(request in ("pricerange", "area", "food", "addr", "phone", "postcode"))
        if request not in self.requests:
            self.requests.append(request)

    def get_requests(self):
        requests = self.requests
        self.requests = []
        return requests

    def preferences_filled(self):
        return len([preference for preference in self.preferences.values() if preference is None]) == 0
    
    def restaurants(self):
        selection = [r for r in self.restaurant_info.restaurants]
        for category, preference in self.preferences.items():
            if preference not in (None, "dontcare"):
                selection = [r for r in selection if r.items[category] == preference]
        selection = [r for r in selection if r not in self.declined]
        return selection


class DialogState:
    # Here we define all the the different states of our system, corresponding to the flowchart as seen in the
    # report. Hierarchically it is structed like this:
    # BaseState:
    #     state_type == SYSTEM: (Here the system generates a sentence -> must override generate_sentence())
    #         Welcome
    #         ReportUnavailability
    #         ....
    #     state_type == USER: (User input is being handled -> must override process_user_act())
    #         ExpressPreference
    #         ....
    #     state_type == EVAL: (These states are where conditions are being checked)
    #         AllPreferencesKnown
    #         ....
    # In every state we can calculate the next state (until the flowchart is exhausted).
    class BaseState:
        def __init__(self, state_type, state, history):
            assert(state_type in ("SYSTEM", "USER", "EVAL"))  # only 3 types of possible states (see diagram).
            self.state_type = state_type
            self.state = state
            self.history = history
    
        def process_user_act(self, _):
            if self.state_type == "USER":
                raise NotImplementedError()  # In case of state_type = USER, this method must be overridden.
        
        def generate_sentence(self):
            if self.state_type == "SYSTEM":
                raise NotImplementedError()  # In case of state_type = SYSTEM, this method must be overridden.
            
        def determine_next_state(self):
            raise NotImplementedError()  # This method must always be overridden.
    
    # Next we define the state_type == SYSTEM states.
    class Welcome(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "Welcome", history)
        
        def generate_sentence(self):
            return "Hi, welcome to the group 10 dialog system. You can ask for restaurants by area, pricerange " \
                   "or food type. How may I help you?"
        
        def determine_next_state(self):
            return DialogState.ExpressPreference(self.history)
                
    class ReportUnavailability(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "ReportUnavailability", history)
        
        def generate_sentence(self):
            sentence = SystemUtterance.generate_combination(self.history.preferences, "DESCRIPTION")
            return f"I'm sorry, there are no restaurants that are {sentence}. Please change one of your preferences."
        
        def determine_next_state(self):
            return DialogState.ExpressPreference(self.history)
    
    class AskPreference(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "AskPreference", history)
            # choose a random preference that is not yet known to ask the user.
            open_preferences = [cat for cat, pref in self.history.preferences.items() if pref is None]
            self.history.last_inquiry = rnd.choice(open_preferences)
        
        def generate_sentence(self):
            confirm = ""
            if len(self.history.speech_acts[-1].parameters) > 0:
                confirm = SystemUtterance.generate_combination(self.history.speech_acts[-1].parameters, "CONFIRMATION")
                confirm = f"Ok, {confirm}. "
            return f"{confirm}{SystemUtterance.ask_information(self.history.last_inquiry)}"
        
        def determine_next_state(self):
            return DialogState.ExpressPreference(self.history)
    
    class SuggestOption(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "SuggestOption", history)
            # choose a random option from the restaurants satisfying the user's conditions.
            self.history.last_suggestion = rnd.choice(self.history.restaurants())
        
        def generate_sentence(self):
            sentence = SystemUtterance.generate_combination(self.history.last_suggestion.items, "STATEMENT")
            return f"{self.history.last_suggestion.items['restaurantname']} is a nice restaurant: {sentence}."
        
        def determine_next_state(self):
            return DialogState.ConfirmNegateOrInquire(self.history)
    
    class ProvideDetails(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "ProvideDetails", history)
        
        def generate_sentence(self):
            specs = []
            suggestion = self.history.last_suggestion
            requests = self.history.get_requests()
            if len(requests) == 0:  # this is temporary, until we have a matcher for 'request' acts.
                requests = ["addr", "postcode", "phone"]
            for request in requests:
                specs.append(f"the {request} is: {suggestion.items[request]}")
            specs[-1] = "and " + specs[-1]
            return f"{suggestion.items['restaurantname']} is a nice restaurant, {', '.join(specs)}."
        
        def determine_next_state(self):
            return DialogState.ConfirmNegateOrInquire(self.history)
    
    class Clarify(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "Clarify", history)

        def generate_sentence(self):
            return "Sorry, I didn't get that. Could you clarify that?"

        def determine_next_state(self):
            return DialogState.ConfirmNegateOrInquire(self.history)

    # Next we define the state_type == USER states.
    class ExpressPreference(BaseState):
        def __init__(self, history):
            super().__init__("USER", "ExpressPreference", history)

        def process_user_act(self, speech_act):
            self.history.speech_acts.append(speech_act)
            # update new information from the user
            if speech_act.act == "inform":
                for category, preference in speech_act.parameters.items():
                    if category in ("pricerange", "food", "area"):
                        self.history.preferences[category] = preference
                    elif category == "":
                        self.history.preferences[self.history.last_inquiry] = preference

        def determine_next_state(self):
            return DialogState.AllPreferencesKnown(self.history)
    
    class ConfirmNegateOrInquire(BaseState):
        def __init__(self, history):
            super().__init__("USER", "ConfirmNegateOrInquire", history)

        def process_user_act(self, speech_act):
            self.history.speech_acts.append(speech_act)
            # update new information from the user, or save the user requests to answer in the next state.
            if speech_act.act == "reqalts":
                self.history.decline(self.history.last_suggestion)
                for category, preference in speech_act.parameters.items():
                    if category in ("pricerange", "food", "area"):
                        self.history.preferences[category] = preference
            elif speech_act.act == "request":
                for request in speech_act.parameters.values():
                    self.history.set_request(request)
            elif speech_act.act in ("thankyou", "bye"):
                self.history.terminate = True

        def determine_next_state(self):
            return DialogState.DetailsAsked(self.history)
    
    # Next we define the state_type == EVAL states.
    class AllPreferencesKnown(BaseState):
        def __init__(self, history):
            super().__init__("EVAL", "AllPreferencesKnown", history)
        
        def determine_next_state(self):
            if self.history.preferences_filled():
                return DialogState.SuggestionAvailable(self.history)
            else:
                return DialogState.AskPreference(self.history)

    class SuggestionAvailable(BaseState):
        def __init__(self, history):
            super().__init__("EVAL", "SuggestionAvailable", history)
    
        def determine_next_state(self):
            if len(self.history.restaurants()) > 0:
                return DialogState.SuggestOption(self.history)
            else:
                return DialogState.ReportUnavailability(self.history)

    class DetailsAsked(BaseState):
        def __init__(self, history):
            super().__init__("EVAL", "DetailsAsked", history)
    
        def determine_next_state(self):
            if self.history.speech_acts[-1].act == "request":
                return DialogState.ProvideDetails(self.history)
            else:
                return DialogState.AlternativeAsked(self.history)

    class AlternativeAsked(BaseState):
        def __init__(self, history):
            super().__init__("EVAL", "AlternativeAsked", history)
    
        def determine_next_state(self):
            if self.history.speech_acts[-1].act == "reqalts":
                return DialogState.SuggestionAvailable(self.history)
            else:
                return DialogState.ThanksBye(self.history)

    class ThanksBye(BaseState):
        def __init__(self, history):
            super().__init__("EVAL", "ThanksBye", history)
    
        def determine_next_state(self):
            if self.history.speech_acts[-1].act in ("thankyou", "bye"):
                return None
            else:
                return DialogState.Clarify(self.history)


class Transitioner:
    def __init__(self, data_elements, restaurant_info):
        self.data_elements = data_elements
        self.matcher = KeywordMatch(restaurant_info)

    def transition(self, current_state, utterance):
        system_sentence = None
        if current_state.state_type == "SYSTEM":
            # Here we generate a system response.
            system_sentence = current_state.generate_sentence()
        elif current_state.state_type == "USER":
            # Here we process user responses.
            speech_act = self.utterance_to_speech_act(utterance)
            current_state.process_user_act(speech_act)
        # No matter the type of state, we determine the next state.
        next_state = current_state.determine_next_state()
        return next_state, system_sentence

    def utterance_to_speech_act(self, utterance):
        # Classify the user input with the SGD classifier from part 1a.
        vector = self.data_elements.vectorizer.transform([utterance])
        predicted = sto_gr_des(self.data_elements, vector)[0]
        if predicted == "inform":
            # Include the found parameters for the 'inform' act.
            preferences = self.matcher.keyword_match_pref(utterance)
            return SpeechAct(predicted, {cat: pref for cat, pref in preferences.items() if pref is not None})
        # For all other acts, don't include any parameters (for requests/reqalts this is yet to be developed).
        return SpeechAct(predicted)


# load dialog_acts and restaurant_info, and begin chat with user.
def main():
    data_elements = DataElements("dialog_acts.dat")
    restaurant_info = RestaurantInfo("restaurant_info.csv")
    transitioner = Transitioner(data_elements, restaurant_info)
    history = DialogHistory(restaurant_info)
    state = DialogState.Welcome(history)
    while state is not None:
        utterance = None
        if state.state_type == "USER":
            utterance = input("").lower()
            print(f"USER: {utterance}")
        state, sentence = transitioner.transition(state, utterance)
        if sentence is not None:
            print(f"SYSTEM: {sentence}")
    print("SYSTEM: Ok, good bye! Come again!")


if __name__ == "__main__":
    main()
