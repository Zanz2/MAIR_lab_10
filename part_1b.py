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


# keyword matching + recognizing patterns, still misses levensthein distance
class PatternAndMatch:
    def __init__(self, restaurant_info):
        self.restaurant_info = restaurant_info
        self.preference_values = {"food": [], "area": [], "pricerange": []}
        
        # Todo: @Hugo: here are the three points from Marijn notes on Teams that you can implement in this class:
        # - this will also allow to refactor the triple if in for restaurant in self.restaurant_info.restaurants:
        # - maybe make the regexes a bit smarter (combine serves/serving etc.)
        # - or remove the pattern matching here altogether (unless you like it)
        for restaurant in self.restaurant_info.restaurants:
            if restaurant.items["food"] not in self.preference_values["food"]:
                self.preference_values["food"].append(restaurant.items["food"])
            if restaurant.items["area"] not in self.preference_values["area"]:
                self.preference_values["area"].append(restaurant.items["area"])
            if restaurant.items["pricerange"] not in self.preference_values["pricerange"]:
                self.preference_values["pricerange"].append(restaurant.items["pricerange"])

    def patterns(self, user_utterance):
        user_text = user_utterance
        words = user_utterance.split(" ")
        pref_dict = {"food": None, "area": None, "pricerange": None}
        # it first looks for patterns in the utterance and fills in the value for food or area if one of these patterns is discovered
        if re.findall(r' serves (.*?) food', user_text):
            pref_dict["food"] = (re.findall(r' serves (.*?) food', user_text))[0]
        if re.findall(r' serving (.*?) food', user_text):
            pref_dict["food"] = (re.findall(r' serving (.*?) food', user_text))[0]
        if re.findall(r' with (.*?) food', user_text):
            pref_dict["food"] = (re.findall(r' with (.*?) food', user_text))[0]
        if re.findall(r' in the (.*?) ', user_text):
            pref_dict["area"] = (re.findall(r' in the (.*?) ', user_text))[0]
        # then it checks per word if it falls in one of the categories
        # if one of the categories already has a value attributed due to a pattern, then this category will not be considered
        for word in words:
            if pref_dict["food"] is None:
                if word in self.preference_values["food"]:
                    pref_dict["food"] = word
            if pref_dict["area"] is None:
                if word in self.preference_values["area"]:
                    pref_dict["area"] = word
            if word in self.preference_values["pricerange"]:
                pref_dict["pricerange"] = word
        return pref_dict


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
        "QUESTION": {
            "area": "What part of town do you have in mind?",
            "pricerange": "Would you like something in the cheap, moderate, or expensive price range?",
            "food": "What kind of food would you like?"}}
    
    @classmethod
    def generate_combination(cls, preferences, utterance_type):
        assert(utterance_type in ("STATEMENT", "DESCRIPTION"))
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
        self.last_preference = None
    
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
            return "Hi, welcome to the group 10 dialogue system. You can ask for restaurants by area , pricerange " \
                   "or food type . How may I help you?"
        
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
            self.history.last_preference = rnd.choice(open_preferences)
        
        def generate_sentence(self):
            return SystemUtterance.ask_information(self.history.last_preference)
        
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
            return DialogState.Reply(self.history)
    
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
            return DialogState.Reply(self.history)
    
    class Clarify(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "Clarify", history)

        def generate_sentence(self):
            return "Sorry, I didn't get that."

        def determine_next_state(self):
            return DialogState.Reply(self.history)

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

        def determine_next_state(self):
            return DialogState.AllPreferencesKnown(self.history)
    
    class Reply(BaseState):
        def __init__(self, history):
            super().__init__("USER", "Reply", history)

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
        self.matcher = PatternAndMatch(restaurant_info)

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
            preferences = self.matcher.patterns(utterance)
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
    print("CHAT TERMINATED")


if __name__ == "__main__":
    main()
