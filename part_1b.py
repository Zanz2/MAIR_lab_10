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
    #     SystemState: (These states are were the system generates a sentence)
    #         Welcome
    #         ReportUnavailability
    #         ....
    #     UserState: (These states are were the user input is being handled)
    #         ExpressPreference
    #         ....
    #     EvalState: (These states are were conditions are being checked)
    #         AllPreferencesKnown
    #         ....
    # In every state we can calculate the next state (until the flowchart is exhausted).
    class BaseState:
        def __init__(self, state_type, state, history):
            self.state_type = state_type
            self.state = state
            self.history = history
    
        def determine_next_state(self):
            raise NotImplementedError()
        
    class SystemState(BaseState):
        def __init__(self, state, history):
            super().__init__("SYSTEM", state, history)
    
        def system_sentence(self):
            raise NotImplementedError()
    
        def determine_next_state(self):
            raise NotImplementedError()

    class UserState(BaseState):
        def __init__(self, state, history):
            super().__init__("USER", state, history)
    
        def process_user_act(self, speech_act):
            raise NotImplementedError()
    
        def determine_next_state(self):
            raise NotImplementedError()

    class EvalState(BaseState):
        def __init__(self, state, history):
            super().__init__("EVAL", state, history)
        
        def determine_next_state(self):
            raise NotImplementedError()
        
    class Welcome(SystemState):
        def __init__(self, history):
            super().__init__("Welcome", history)
        
        def system_sentence(self):
            return "Hi, welcome to the group 10 dialogue system. You can ask for restaurants by area , pricerange " \
                   "or food type . How may I help you?"
        
        def determine_next_state(self):
            return DialogState.ExpressPreference(self.history)
                
    class ReportUnavailability(SystemState):
        def __init__(self, history):
            super().__init__("ReportUnavailability", history)
        
        def system_sentence(self):
            specs = []
            if self.history.preferences["area"] not in (None, "dontcare"):
                specs.append(f"in the {self.history.preferences['area']} part of town")
            if self.history.preferences["food"] not in (None, "dontcare"):
                specs.append(f"serving {self.history.preferences['food']} food")
            if self.history.preferences["pricerange"] not in (None, "dontcare"):
                specs.append(f"in the {self.history.preferences['pricerange']} pricerange")
            return f"I'm sorry, there are no restaurants {', '.join(specs)}. Please change one of your preferences."
        
        def determine_next_state(self):
            return DialogState.ExpressPreference(self.history)
    
    class AskPreference(SystemState):
        def __init__(self, history):
            super().__init__("AskPreference", history)
            # choose a random preference that is not yet known to ask the user.
            open_preferences = [cat for cat, pref in self.history.preferences.items() if pref is None]
            self.history.last_preference = rnd.choice(open_preferences)
        
        def system_sentence(self):
            if self.history.last_preference == "pricerange":
                return "Would you like something in the cheap, moderate, or expensive price range?"
            elif self.history.last_preference == "area":
                return "What part of town do you have in mind?"
            elif self.history.last_preference == "food":
                return "What kind of food would you like?"
            else:
                raise NotImplementedError()
        
        def determine_next_state(self):
            return DialogState.ExpressPreference(self.history)
    
    class SuggestOption(SystemState):
        def __init__(self, history):
            super().__init__("SuggestOption", history)
            # choose a random option from the restaurants satisfying the user's conditions.
            self.history.last_suggestion = rnd.choice(self.history.restaurants())
        
        def system_sentence(self):
            specs = []
            if self.history.preferences["area"] not in (None, "dontcare"):
                specs.append(f"in the {self.history.last_suggestion.items['area']} part of town")
            if self.history.preferences["pricerange"] not in (None, "dontcare"):
                specs.append(f"the prices are {self.history.last_suggestion.items['pricerange']}")
            if self.history.preferences["food"] not in (None, "dontcare"):
                specs.append(f"they serve {self.history.last_suggestion.items['food']} food")
            if len(specs) > 0:
                specs[-1] = "and " + specs[-1]
            return f"{self.history.last_suggestion.items['restaurantname']} is a nice restaurant {', '.join(specs)}"
        
        def determine_next_state(self):
            return DialogState.Reply(self.history)
    
    class ProvideDetails(SystemState):
        def __init__(self, history):
            super().__init__("ProvideDetails", history)
        
        def system_sentence(self):
            specs = []
            suggestion = self.history.last_suggestion
            requests = self.history.get_requests()
            if len(requests) == 0:  # this is temporary, until we have a matcher for 'request' acts.
                requests = ["addr", "postcode", "phone"]
            for request in requests:
                specs.append(f"the {request} is: {suggestion.items[request]}")
            specs[-1] = "and " + specs[-1]
            return f"{suggestion.restaurantname} is a nice restaurant, {', '.join(specs)}."
        
        def determine_next_state(self):
            return DialogState.Reply(self.history)
    
    class Clarify(SystemState):
        def __init__(self, history):
            super().__init__("Clarify", history)

        def system_sentence(self):
            return "Sorry, I didn't get that."

        def determine_next_state(self):
            return DialogState.Reply(self.history)

    class ExpressPreference(UserState):
        def __init__(self, history):
            super().__init__("ExpressPreference", history)

        def process_user_act(self, speech_act):
            self.history.speech_acts.append(speech_act)
            # update new information from the user
            if speech_act.act == "inform":
                for category, preference in speech_act.parameters.items():
                    if category in ("pricerange", "food", "area"):
                        self.history.preferences[category] = preference

        def determine_next_state(self):
            return DialogState.AllPreferencesKnown(self.history)
    
    class Reply(UserState):
        def __init__(self, history):
            super().__init__("Reply", history)

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

    class AllPreferencesKnown(EvalState):
        def __init__(self, history):
            super().__init__("AllPreferencesKnown", history)
    
        def determine_next_state(self):
            if self.history.preferences_filled():
                return DialogState.SuggestionAvailable(self.history)
            else:
                return DialogState.AskPreference(self.history)

    class SuggestionAvailable(EvalState):
        def __init__(self, history):
            super().__init__("SuggestionAvailable", history)
    
        def determine_next_state(self):
            if len(self.history.restaurants()) > 0:
                return DialogState.SuggestOption(self.history)
            else:
                return DialogState.ReportUnavailability(self.history)

    class DetailsAsked(EvalState):
        def __init__(self, history):
            super().__init__("DetailsAsked", history)
    
        def determine_next_state(self):
            if self.history.speech_acts[-1].act == "request":
                return DialogState.ProvideDetails(self.history)
            else:
                return DialogState.AlternativeAsked(self.history)

    class AlternativeAsked(EvalState):
        def __init__(self, history):
            super().__init__("AlternativeAsked", history)
    
        def determine_next_state(self):
            if self.history.speech_acts[-1].act == "reqalts":
                return DialogState.SuggestionAvailable(self.history)
            else:
                return DialogState.ThanksBye(self.history)

    class ThanksBye(EvalState):
        def __init__(self, history):
            super().__init__("ThanksBye", history)
    
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
        # No matter the type of state, we will determine the next state here.
        if current_state.state_type == "SYSTEM":
            # Here we also generate a system response.
            system_sentence = current_state.system_sentence()
            next_state = current_state.determine_next_state()
        elif current_state.state_type == "USER":
            # here we process user responses.
            speech_act = self.utterance_to_speech_act(utterance)
            current_state.process_user_act(speech_act)
            next_state = current_state.determine_next_state()
        elif current_state.state_type == "EVAL":
            next_state = current_state.determine_next_state()
        else:
            raise NotImplementedError()
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


def chat(data_elements, restaurant_info):
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


# load dialog_acts and restaurant_info, and begin chat with user.
def main():
    data_elements = DataElements("dialog_acts.dat")
    restaurant_info = RestaurantInfo("restaurant_info.csv")
    chat(data_elements, restaurant_info)


if __name__ == "__main__":
    main()
