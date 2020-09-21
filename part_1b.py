import Levenshtein  # this will give you an error if you dont have it installed
from part_1a import *
import random as rnd


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

 #keyword matching + recognizing patterns, still misses levensthein distance
 class PatternAndMatch:
    def __init__(self, restaurant_info):
        self.restaurant_info = restaurant_info
        self.preference_values = {"food": [], "area": [], "price": []}
        self.pref_dict = {"food": None, "area": None, "price": None}

        
        for restaurant in self.restaurant_info.restaurants:
                if restaurant.food not in self.preference_values["food"] and restaurant.food != "food":
                    self.preference_values["food"].append(restaurant.food)
                if restaurant.area not in self.preference_values["area"] and restaurant.area != "area":
                    self.preference_values["area"].append(restaurant.area)
                if restaurant.price not in self.preference_values["price"] and restaurant.price != "pricerange":
                    self.preference_values["price"].append(restaurant.price)
    
    def patterns(self, user_utterance):
        
        user_text = user_utterance
        words = user_utterance.split(" ")
        # it first looks for patterns in the utterance and fills in the value for food or area if one of these patterns is discovered
        if (re.findall(r' serves (.*?) food', user_text)):
            self.pref_dict["food"] = (re.findall(r' serves (.*?) food', user_text))[0]
        if (re.findall(r' serving (.*?) food', user_text)):
            self.pref_dict["food"] = (re.findall(r' serving (.*?) food', user_text))[0]
        if re.findall(r' with (.*?) food', user_text):
            self.pref_dict["food"] = (re.findall(r' with (.*?) food', user_text))[0]
        if re.findall(r' in the (.*?) ', user_text):
            self.pref_dict["area"] = (re.findall(r' in the (.*?) ', user_text))[0]

        #then it checks per word if it falls in one of the categories
        #if one of the categories already has a value attributed due to a pattern, then this category will not be considered
        for word in words:
            if self.pref_dict["food"] == None: 
                if word in self.preference_values["food"]:
                    self.pref_dict["food"] = word
            if self.pref_dict["area"] == None:
                if word in self.preference_values["area"]: 
                    self.pref_dict["area"] = word
            if word in self.preference_values["price"]:
                self.pref_dict["price"] = word
        return self.pref_dict

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
                selection = [r for r in selection if r.get(category) == preference]
        selection = [r for r in selection if r not in self.declined]
        return selection


class DialogState:
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
            self.history.last_suggestion = rnd.choice(self.history.restaurants())
        
        def system_sentence(self):
            specs = []
            if self.history.preferences["area"] not in (None, "dontcare"):
                specs.append(f"in the {self.history.last_suggestion.area} part of town")
            if self.history.preferences["pricerange"] not in (None, "dontcare"):
                specs.append(f"the prices are {self.history.last_suggestion.pricerange}")
            if self.history.preferences["food"] not in (None, "dontcare"):
                specs.append(f"they serve {self.history.last_suggestion.food} food")
            if len(specs) > 0:
                specs[-1] = "and " + specs[-1]
            return f"{self.history.last_suggestion.name} is a nice restaurant {', '.join(specs)}"
        
        def determine_next_state(self):
            return DialogState.Reply(self.history)
    
    class ProvideDetails(SystemState):
        def __init__(self, history):
            super().__init__("ProvideDetails", history)
        
        def system_sentence(self):
            specs = []
            suggestion = self.history.last_suggestion
            requests = self.history.get_requests()
            for request in requests:
                specs.append(f"the {request} is: {suggestion.get(request)}")
            specs[-1] = "and " + specs[-1]
            return f"{suggestion.name} is a nice restaurant, {', '.join(specs)}."
        
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


def utterance_to_speech_act(utterance):
    return SpeechAct(utterance)


def transition(current_state, utterance):
    system_sentence = None
    if current_state.state_type == "SYSTEM":
        system_sentence = current_state.system_sentence()
        next_state = current_state.determine_next_state()
    elif current_state.state_type == "USER":
        speech_act = utterance_to_speech_act(utterance)
        current_state.process_user_act(speech_act)
        next_state = current_state.determine_next_state()
    elif current_state.state_type == "EVAL":
        next_state = current_state.determine_next_state()
    else:
        raise NotImplementedError()
    return next_state, system_sentence


def chat(restaurant_info):
    history = DialogHistory(restaurant_info)
    state = DialogState.Welcome(history)
    while state is not None:
        utterance = None
        if state.state_type == "USER":
            utterance = input("").lower()
            print(f"USER: {utterance}")
        state, sentence = transition(state, utterance)
        if sentence is not None:
            print(f"SYSTEM: {sentence}")
    print("CHAT TERMINATED")


def main():
    dialog_data = DialogData("all_dialogs.txt")
    restaurant_info = RestaurantInfo("restaurant_info.csv")
    # dialog_history = DialogHistory(restaurant_info)
    chat(restaurant_info)


if __name__ == "__main__":
    main()
