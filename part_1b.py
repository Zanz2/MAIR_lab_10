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
        # Dict with possible values per category to match input words with. For speech_act=inform these possible
        # values are are the unique values from the 'restaurant_info' dataset. For speech_act=request the possible
        # values are synonyms of the word itself.
        self.possible_values = {
            "items": {
                "food": self.__get_unique_entries("food"),
                "area": self.__get_unique_entries("area"),
                "pricerange": self.__get_unique_entries("pricerange")},
            "synonyms": {
                "food": ["food", "cuisine", "foodtype"],
                "area": ["area", "neighborhood", "region"],
                "pricerange": ["pricerange", "price", "prices", "pricyness"],
                "phone": ["phone", "phonenumber", "telephone", "telephonenumber", "number"],
                "addr": ["addr", "address", "street"],
                "postcode": ["postcode", "postal", "post", "code"]}}
        # The dontcare specifier we check with regular expressions, since they mostly consist of multiple words.
        self.dontcares = [r".*(^| )any($| ).*", r".*(^| )doesnt ?matter($| ).*", r".*(^| )dont ?care($| ).*"]
        # Different word_types have their own minimum word length, words below that length are
        # not considered for matching (unless they are a perfect match with a value from self.possible_values).
        # For example: if we misspell 'west' we would still consider 'est', but not 'st'.
        self.levenshtein_min = {
            "items": {"food": 2, "pricerange": 3, "area": 2},
            "synonyms": {"food": 3, "pricerange": 3, "area": 2, "phone": 3, "addr": 4, "postcode": 3}}
        self.blacklist = ["west", "would", "want", "world", "a", "part", "can", "what", "that", "the"]

    def __get_unique_entries(self, category):
        return list(set(r.items[category] for r in self.restaurant_info.restaurants if r.items[category] != ""))
        
    def check_levenshtein(self, relation, word_type, word):
        assert(word_type in self.levenshtein_min[relation].keys())
        # Allowed word_type: "food_type", "price_range", "location", "phone", "address" and "postcode".
        if word in self.possible_values[relation][word_type]:
            # Perfect matches are returned regardless of blacklists or wordlength.
            return word
        elif word not in self.blacklist and len(word) >= self.levenshtein_min[relation][word_type]:
            # Words that easily create confusion are blacklisted. And 'short' wordt are also not considered.
            correct_word, lv_distance = None, None
            for value in self.possible_values[relation][word_type]:
                # For all possible values of the word_type we check the Levenshtein distance, and choose the one with
                # the lowest (in case of a tie, we just pick the first one we encounter).
                distance = Levenshtein.distance(value, word)
                if lv_distance is None or distance < lv_distance:
                    lv_distance = distance
                    correct_word = value
            if lv_distance <= int(len(correct_word) / 3):
                # We only assume that this word was meant, if the Levenshtein distance is less or equal than a third
                # of the length of the original word (so for 'west', 1 edit is allowed, and for 'chinese': 2)
                return correct_word
        return None

    def keyword_match_pref(self, user_utterance):
        # this method will check whether the user mentions a word that matches with a word from the csv file
        # it also checks words that are misspelled with help of the check_levenshtein method
        # it returns a dictionary with preferences for foodtype, area and pricerange
        words = user_utterance.split(" ")
        pref_dict = {"food": None, "area": None, "pricerange": None, "": None}
        if any(re.search(dc, user_utterance) is not None for dc in self.dontcares):
            # Here we know that there is a specification of 'dontcare'. We first check if this is specific to a
            # certain category. Otherwise we assign it to the key "", which means the category has to be infered
            # from the last inquiry that the system made.
            for word in words:
                for category in ("food", "area", "pricerange"):
                    if self.check_levenshtein("synonyms", category, word) is not None:
                        pref_dict[category] = "dontcare"
            if all(pref is None for pref in pref_dict.values()):
                pref_dict[""] = "dontcare"
        # Now we check for specific preferences for specific categories (possibly overwriting a 'dontcare').
        for word in words:
            for preference in ["food", "area", "pricerange"]:
                if pref_dict[preference] is None:
                    lev_word = self.check_levenshtein("items", preference, word)
                    if lev_word is not None:
                        pref_dict[preference] = lev_word
        return pref_dict


    def keyword_match_request(self, user_utterance):
        requests = []
        words = user_utterance.split(" ")
        for word in words:
            for category in self.possible_values["synonyms"]:
                if self.check_levenshtein("synonyms", category, word) is not None:
                    requests.append(category)
        return requests


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
            "area": "in the {0}",
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
    
    def set_request(self, requests):
        for request in requests:
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
    
    def process_preferences(self, speech_act):
        if speech_act.act == "reqalts" and self.last_suggestion is not None:
            self.decline(self.last_suggestion)
        for category, preference in speech_act.parameters.items():
            if category in ("pricerange", "food", "area"):
                self.preferences[category] = preference
            elif category == "":
                self.preferences[self.last_inquiry] = preference


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
            confirm = SystemUtterance.generate_combination(self.history.speech_acts[-1].parameters, "CONFIRMATION")
            if confirm != "":
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
            if len(requests) == 0:
                # If no requests have been identified, just give all contact info.
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
            if speech_act.act in ("inform", "reqalts"):
                self.history.process_preferences(speech_act)

        def determine_next_state(self):
            return DialogState.AllPreferencesKnown(self.history)
    
    class ConfirmNegateOrInquire(BaseState):
        def __init__(self, history):
            super().__init__("USER", "ConfirmNegateOrInquire", history)

        def process_user_act(self, speech_act):
            self.history.speech_acts.append(speech_act)
            # update new information from the user, or save the user requests to answer in the next state.
            if speech_act.act in ("inform", "reqalts"):
                self.history.process_preferences(speech_act)
            elif speech_act.act == "request":
                self.history.set_request(speech_act.parameters.values())
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
        if predicted in ("inform", "reqalts"):
            # Include the found parameters for the 'inform' act (for 'reqalts' these are the same possible paramters).
            preferences = self.matcher.keyword_match_pref(utterance)
            return SpeechAct(predicted, {cat: pref for cat, pref in preferences.items() if pref is not None})
        elif predicted == "request":
            # Include the matched requests.
            requests = self.matcher.keyword_match_request(utterance)
            return SpeechAct(predicted, requests)
        # For all other acts, don't include any parameters.
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
            utterance = input("").lower().replace("'", "")
            print(f"USER: {utterance}")
        state, sentence = transitioner.transition(state, utterance)
        if sentence is not None:
            print(f"SYSTEM: {sentence}")
    print("SYSTEM: Ok, good bye! Come again!")


if __name__ == "__main__":
    main()
