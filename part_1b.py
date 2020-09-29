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


# Here we store the logic to do an initial cleanup of input sentences. Lowercase, dual-words, etc.
class SentenceCleanser:
    DUALS = ["asian oriental", "modern european", "north american"]
    
    @classmethod
    def concat_dual_words(cls, utterance):
        for d in cls.DUALS:
            if type(utterance) is str and d in utterance:
                utterance = utterance.replace(d, d.replace(" ", "_"))
        return utterance
    
    @classmethod
    def cleanse(cls, utterance):
        return cls.concat_dual_words(utterance.replace("'", "").lower())


# here we load the restaurant info and create a list with the restaurant data (including food, area, pricerange etc.)
class RestaurantInfo:
    def __init__(self, filename):
        self.filename = filename
        self.restaurants = self.__parse_data()
        self.sets = {
            "pricerange": [
                {"cheap", "moderate"},
                {"moderate", "expensive"}],
            "area": [
                {"centre", "north", "west"},
                {"centre", "north", "east"},
                {"centre", "south", "west"},
                {"centre", "south", "east"}],
            "food": [
                {"thai", "chinese", "korean", "vietnamese", "asian_oriental"},
                {"mediterranean", "spanish", "portuguese", "italian", "romanian", "tuscan", "catalan"},
                {"french", "european", "bistro", "swiss", "gastropub", "traditional"},
                {"north_american", "steakhouse", "british"},
                {"lebanese", "turkish", "persian"},
                {"international", "modern_european", "fusion"}]}

    def __parse_data(self):
        restaurants = []
        csv_data = pd.read_csv(self.filename, header=0, na_filter=False)
        for row in csv_data.values:
            row = [SentenceCleanser.concat_dual_words(column) for column in row]
            restaurants.append(Restaurant(row))
        return restaurants
    
    def query_selection(self, preferences):
        # Return all restaurants that satisfy the conditions in preferences.
        selection = [r for r in self.restaurants]
        for category, preference in preferences.items():
            if preference not in (None, "dontcare"):
                selection = [r for r in selection if r.items[category] == preference]
        return selection
    
    def query_alternatives(self, preferences):
        # Return all restaurants that whose values are members of the same set as the specified preference.
        alternatives = []
        new_prefs = {cat: set() for cat, pref in preferences.items() if pref not in (None, "dontcare")}
        for cat in new_prefs:
            for cset in self.sets[cat]:
                if preferences[cat] in cset:
                    new_prefs[cat] = new_prefs[cat].union(cset)
        for restaurant in self.restaurants:
            if all(restaurant.items[cat] in new_prefs[cat] for cat in new_prefs):
                deviant = len([cat for cat in new_prefs if restaurant.items[cat] != preferences[cat]])
                # We also keep track of how many categories are deviant from the initial preference.
                alternatives.append((deviant, restaurant))
        # First we shuffle the results, then we sort by the number of deviant categories, such that restaurants with
        # the same number of deviant categories are listed in random order.
        rnd.shuffle(alternatives)
        alternatives.sort(key=lambda alternative: alternative[0])
        # We only return the restaurant instances.
        return [alternative[1] for alternative in alternatives]


# here we define a class for each restaurant containing the relevant information, naming them accordingly
class Restaurant:
    def __init__(self, data_row):
        self.items = {
            "restaurantname": data_row[0],
            "pricerange": data_row[1],
            "area": data_row[2],
            "food": data_row[3],
            "phone": data_row[4],
            "addr": data_row[5],
            "postcode": data_row[6],
            "good food": data_row[7],
            "good atmosphere": data_row[8],
            "big beverage selection": data_row[9],
            "spacious": data_row[10],
            "busy": None,
            "long time": None,
            "short time": None,
            "children": None,
            "romantic": None,
            "fast service": None,
            "seating outside": None,
            "good for meetings": None,
            "good for studying": None
        }
        self.score = 0
        self.preference_assesment_dict = {
            "pros": [],
            "cons": []
        }

    def apply_inferred_rules(self):
        for rule in Inference.Rules:
            rule.infere_rule(self)

    def __str__(self):
        return str(self.items)
    
    def name(self):
        return self.items['restaurantname'].title()


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
                "postcode": ["postcode", "postal", "post", "code"]
            },
            "secondary_synonyms": {
                "good food": ["good food", "amazing food", "great food", "appetizing", "tempting", "flavorsome", "tasteful", "yummy", "delicious", "tasty"],
                "good atmosphere": ["good atmosphere", "environment"],
                "big beverage selection": ["big beverage selection", "drink list"],
                "spacious": ["spacious", "roomy", "sizeable", "large space", "high-ceilinged"],
                "busy": ["busy", "hectic"],
                "long time": ["long time", "for hours"],
                "short time": ["short time", "quick meal"],
                "children": ["children" "child friendly", "childfriendly", "familyfriendly", "family friendly", "for the kids", "safe for children"],
                "romantic": ["romantic", "idyllic", "charming", "idealistic", "picturesque"],
                "fast service": ["fast service", "swift service", "quick service", "rapid service"],
                "seating outside": ["seating outside", "outdoor seating", "terrace", "outside", "garden"],
                "good for meetings": ["good for meetings", "nice for meetings", "meeting", "conference", "gathering", "convention", "summit", "get-together", "rendezvous"],
                "good for studying": ["good for studying", "nice for studying", "place of education", "learning space"]
            },
            "negations": ["shouldnt", "not", "dont", "wont", "arent", "cant"]
        }
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
        # Return all unique values from restaurants dataset. Skip over empty entries, and concat dual-words.
        uniques = list(set(r.items[category] for r in self.restaurant_info.restaurants))
        return [SentenceCleanser.concat_dual_words(u) for u in uniques if u != ""]
        
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

    def sentence_match_secondary_pref(self, user_utterance):
        negations = self.possible_values["negations"]
        synonyms = self.possible_values["secondary_synonyms"]
        end_of_word_array = []
        for preference, word_array in synonyms.items():
            for word in word_array:
                if user_utterance.find(word) > -1:
                    end_of_word_index = user_utterance.find(word) + len(word)
                    word_dict = {
                        "preference": preference,
                        "end_of_word_index": end_of_word_index
                    }
                    end_of_word_array.append(word_dict)
                    break
        end_of_word_array.sort(key=lambda dictionary: dictionary["end_of_word_index"])
        start = 0
        result_array = []
        for selection_dict in end_of_word_array:
            sentence_chunk = user_utterance[start:selection_dict["end_of_word_index"]]
            start = selection_dict["end_of_word_index"]
            result_dict = {
                "preference": selection_dict["preference"],
                "negation": False
            }
            for negation in negations:
                if negation in sentence_chunk:
                    result_dict["negation"] = True
            result_array.append(result_dict)
        return result_array
    
    @staticmethod
    def number_match(utterance, range_length):
        for i in range(range_length):
            if re.search(f".*(^|\\D){i}($|\\D).*", utterance) is not None:
                return i
        return None


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
            "food": "What kind of food would you like?",
            "secondaries": "Do you have any other wishes? Perhaps: good food / good atmosphere / big beverage "
                           "selection / spacious / not busy / long time / \nshort time / children / romantic / "
                           "fast service / seating outside / good for meetings / good for studying."
        }
    }

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
    
    @classmethod
    def offer_alternatives(cls, preferences, alternatives):
        prefs = SystemUtterance.generate_combination(preferences, "DESCRIPTION")
        sentence = f"I'm sorry, there are no restaurants that are {prefs}. But we have some other recommendations: "
        for i, a in enumerate(alternatives):
            description = SystemUtterance.generate_combination(a.items, "STATEMENT")
            sentence += f"\n    {i}. {a.name()}: {description}"
        sentence += "\nPlease select one of these numbers or change your preferences a bit."
        return sentence


class DialogHistory:
    def __init__(self, restaurant_info):
        self.restaurant_info = restaurant_info
        self.matcher = KeywordMatch(restaurant_info)
        self.preferences = {"pricerange": None, "area": None, "food": None}
        self.secondary_preferences = {
            "good food": None,
            "good atmosphere": None,
            "big beverage selection": None,
            "spacious": None,
            "busy": None,
            "long time": None,
            "short time": None,
            "children": None,
            "romantic": None,
            "fast service": None,
            "seating outside": None,
            "good for meetings": None,
            "good for studying": None}
        self.secondary_preferences_asked = False
        self.last_user_utterance = None
        self.declined = []
        self.requests = []
        self.terminate = False
        self.speech_acts = []
        self.last_suggestion = None
        self.last_inquiry = None
        self.last_alternatives = None
        self.chosen_alternative = None
    
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

    def relevant_open_preferences(self):
        # Check for which categories (pricerange/area/food) there are still multiple possibilities within the currently
        # available suggestions. For example, if all the options are in the 'north', then 'area' is not a relevant
        # category to specify anymore. Categories that are already specified also dont have to specified again.
        categories, options = [], self.restaurants()
        for cat in self.preferences:
            if self.preferences[cat] is None and len(set(r.items[cat] for r in options)) > 1:
                categories.append(cat)
        return categories
        
    def restaurants(self):
        return [r for r in self.restaurant_info.query_selection(self.preferences) if r not in self.declined]
    
    def alternatives(self, maximum):
        alternatives = [r for r in self.restaurant_info.query_alternatives(self.preferences)]
        self.last_alternatives = [r for r in alternatives if r not in self.declined][:maximum]
        return self.last_alternatives
    
    def process_preferences(self, speech_act):
        if speech_act.act == "reqalts" and self.last_suggestion is not None:
            self.decline(self.last_suggestion)
        for category, preference in speech_act.parameters.items():
            if category in ("pricerange", "food", "area"):
                self.preferences[category] = preference
            elif category == "":
                self.preferences[self.last_inquiry] = preference

    def process_secondary_preferences(self, user_utterance):
        self.secondary_preferences_asked = True
        user_utterance = user_utterance.replace("'", "")

        result_array = self.matcher.sentence_match_secondary_pref(user_utterance)
        for result in result_array:
            if result["negation"]:
                self.secondary_preferences[result["preference"]] = False
            else:
                self.secondary_preferences[result["preference"]] = True

    # show the user the suggestions, with prints when a certain restaurant complied or didnt comply to the selected rules
    # lastly refactor the restaurant.apply infered rule ifs to use the lambda design


class DialogState:
    # Here we define all the the different states of our system, corresponding to the flowchart as seen in the
    # report. Hierarchically it is structed like this:
    # BaseState:
    #     state_type == SYSTEM: (Here the system generates a sentence -> must override generate_sentence())
    #         Welcome
    #         OfferAlternativeOrChangePreference
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
                
    class OfferAlternativeOrChangePreference(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "OfferAlternativeOrChangePreference", history)
        
        def generate_sentence(self):
            alternatives = self.history.alternatives(5)
            return SystemUtterance.offer_alternatives(self.history.preferences, alternatives)
        
        def determine_next_state(self):
            return DialogState.AcceptAlternativeOrChangePreference(self.history)
    
    class AskPreference(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "AskPreference", history)
            # Choose a random preference that is not yet known (and is still relevant to ask) to ask the user.
            self.history.last_inquiry = rnd.choice(self.history.relevant_open_preferences())
        
        def generate_sentence(self):
            # First implicitly confirm the given preferences (if any), then ask the chosen inquiry.
            confirm = SystemUtterance.generate_combination(self.history.speech_acts[-1].parameters, "CONFIRMATION")
            if confirm != "":
                confirm = f"Ok, {confirm}. "
            return f"{confirm}{SystemUtterance.ask_information(self.history.last_inquiry)}"
        
        def determine_next_state(self):
            return DialogState.ExpressPreference(self.history)

    class AskSecondaryPreference(BaseState):
        # System asks for secondary preferences
        def __init__(self, history):
            super().__init__("SYSTEM", "AskSecondaryPreference", history)

        def generate_sentence(self):
            return f"{SystemUtterance.ask_information('secondaries')}"

        def determine_next_state(self):
            return DialogState.ExpressSecondaryPreference(self.history)
    
    class SuggestOption(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "SuggestOption", history)
            available_restaurants = self.history.restaurants()
            # choose a random option from the restaurants satisfying the user's conditions.
            for restaurant in available_restaurants:
                score_count = 0
                restaurant.apply_inferred_rules()  # apply the rules for 3 passes (new antecedents from consequents etc)
                restaurant.apply_inferred_rules()
                restaurant.apply_inferred_rules()
                restaurant_inferred_preferences = restaurant.items
                user_stated_preferences = history.secondary_preferences
                for preference, boolean_val in user_stated_preferences.items():
                    if boolean_val is not None:
                        if restaurant_inferred_preferences[preference] == boolean_val:
                            score_count += 1
                            restaurant.preference_assesment_dict["pros"].append(preference)
                        else:
                            score_count -= 1
                            restaurant.preference_assesment_dict["cons"].append(preference)
                restaurant.score = score_count
            restaurant_list = sorted(available_restaurants, key=lambda restaur: restaur.score)
            # Show restaurants by order of score on secondary preferences
            self.history.last_suggestion = restaurant_list[0]
            # now that the restaurants were supposedly ranked by secondary preference score
        
        def generate_sentence(self):
            sentence = SystemUtterance.generate_combination(self.history.last_suggestion.items, "STATEMENT")
            picked_restaurant = self.history.last_suggestion
            if len(picked_restaurant.preference_assesment_dict["pros"]) > 0:
                match_sentence = "\n Additionally it matches these of your prefferences: \n"
                for match in picked_restaurant.preference_assesment_dict["pros"]:
                    match_sentence += "{}, ".format(match)
                sentence += match_sentence
            if len(picked_restaurant.preference_assesment_dict["cons"]) > 0:
                missmatch_sentence = "\n However it does NOT match these of your prefferences: \n"
                for missmatch in picked_restaurant.preference_assesment_dict["cons"]:
                    missmatch_sentence += "{}, ".format(missmatch)
                sentence += missmatch_sentence
            return f"{self.history.last_suggestion.name()} is a nice restaurant: {sentence}"
        
        def determine_next_state(self):
            return DialogState.ConfirmNegateOrInquire(self.history)
    
    class SuggestAlternative(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "SuggestAlternative", history)
            self.history.last_suggestion = self.history.chosen_alternative
        
        def generate_sentence(self):
            sentence = SystemUtterance.generate_combination(self.history.last_suggestion.items, "STATEMENT")
            return f"{self.history.last_suggestion.name()} is a nice restaurant: {sentence}"
        
        def determine_next_state(self):
            return DialogState.ConfirmNegateOrInquire(self.history)
    
    class ProvideDetails(BaseState):
        def __init__(self, history):
            super().__init__("SYSTEM", "ProvideDetails", history)
        
        def generate_sentence(self):
            suggestion = self.history.last_suggestion
            requests = self.history.get_requests()
            if len(requests) == 0:  # If no requests have been identified, just give all contact info.
                requests = ["addr", "postcode", "phone"]
            specs = [f"the {req} is {suggestion.items[req]}" for req in requests]
            specs[-1] = "and " + specs[-1]
            return f"{suggestion.name()} is a nice restaurant, {', '.join(specs)}."
        
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

    class AcceptAlternativeOrChangePreference(BaseState):
        def __init__(self, history):
            super().__init__("USER", "AcceptAlternativeOrChangePreference", history)

        def process_user_act(self, speech_act):
            self.history.speech_acts.append(speech_act)
            # update new information from the user
            choice = KeywordMatch.number_match(self.history.last_user_utterance, len(self.history.last_alternatives))
            if choice is not None:
                self.history.chosen_alternative = self.history.last_alternatives[choice]
            elif speech_act.act in ("inform", "reqalts"):
                self.history.process_preferences(speech_act)

        def determine_next_state(self):
            return DialogState.AlternativeAccepted(self.history)

    class ExpressSecondaryPreference(BaseState):
        # (get express Secondary preference user input here)
        def __init__(self, history):
            super().__init__("USER", "ExpressSecondaryPreference", history)

        def process_user_act(self, speech_act):
            # This processes the sentence instead of the act, because the acts
            # of secondary preferences are usually very varied or even null
            self.history.process_secondary_preferences(self.history.last_user_utterance)

        def determine_next_state(self):
            return DialogState.SecondaryPreferencesKnown(self.history)
    
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
    class AllPreferencesKnown(BaseState):  # template for Secondary preferences asked?
        def __init__(self, history):
            super().__init__("EVAL", "AllPreferencesKnown", history)
        
        def determine_next_state(self):
            if len(self.history.relevant_open_preferences()) == 0:
                return DialogState.SuggestionAvailable(self.history)
            else:
                return DialogState.AskPreference(self.history)

    class SecondaryPreferencesKnown(BaseState):  # template for Secondary preferences asked?
        def __init__(self, history):
            super().__init__("EVAL", "SecondaryPreferencesKnown", history)

        def determine_next_state(self):
            if self.history.secondary_preferences_asked:
                return DialogState.SuggestOption(self.history)
                # return DialogState.SuggestionAvailable(self.history)
            else:
                return DialogState.AskSecondaryPreference(self.history)

    class SuggestionAvailable(BaseState):
        def __init__(self, history):
            super().__init__("EVAL", "SuggestionAvailable", history)
    
        def determine_next_state(self):
            if len(self.history.restaurants()) > 0:
                return DialogState.AskSecondaryPreference(self.history)
                # return DialogState.SuggestOption(self.history)
            else:
                return DialogState.OfferAlternativeOrChangePreference(self.history)

    class AlternativeAccepted(BaseState):
        def __init__(self, history):
            super().__init__("EVAL", "AlternativeAccepted", history)
    
        def determine_next_state(self):
            if self.history.chosen_alternative not in self.history.declined + [None]:
                return DialogState.SuggestAlternative(self.history)
            else:
                return DialogState.AllPreferencesKnown(self.history)

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
            current_state.history.last_user_utterance = utterance
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


class InferenceRule:
    id_counter = 0

    def __init__(self, antecedent, consequent, truth_value):
        InferenceRule.id_counter += 1
        self.rule_id = InferenceRule.id_counter
        self.antecedent = antecedent
        self.consequent = consequent
        self.truth_value = truth_value

    def infere_rule(self, restaurant):
        if self.antecedent(restaurant.items):
            restaurant.items[self.consequent] = self.truth_value


class Inference:
    # Dilemma: do it like this, or use only A and B style like in examples?
    # this way = more compact and succint
    # the other way = more code, more parameters, but the inference rule class is a bit simplified
    Rules = [
        InferenceRule(lambda i: i["big beverage selection"] and i["good atmosphere"], "long time", True),
        InferenceRule(lambda i: i["good food"] and i["good atmosphere"], "busy", True),
        InferenceRule(lambda i: i["good food"] and i["pricerange"] == "cheap", "busy", True),
        InferenceRule(lambda i: i["fast service"] and i["pricerange"] == "cheap", "short time", True),
        InferenceRule(lambda i: i["food"] == "spanish", "long time", True),
        InferenceRule(lambda i: i["busy"], "long time", True),
        InferenceRule(lambda i: i["long time"], "children", False),
        InferenceRule(lambda i: i["short time"], "children", True),
        InferenceRule(lambda i: i["busy"], "romantic", False),
        InferenceRule(lambda i: i["long time"], "romantic", True),
        InferenceRule(lambda i: i["children"], "good for studying", False),
        InferenceRule(lambda i: i["children"], "good for meetings", False),
        InferenceRule(lambda i: i["spacious"] and i["good atmosphere"], "good for studying", True),
        InferenceRule(lambda i: i["seating outside"] and i["good atmosphere"], "romantic", True),
        InferenceRule(lambda i: i["long time"], "good for studying", True),
        InferenceRule(lambda i: i["long time"], "good for meetings", True),
        InferenceRule(lambda i: i["pricerange"] == "expensive" and i["short time"], "busy", False),
        InferenceRule(lambda i: i["pricerange"] == "moderate" and i["long time"], "good for studying", True),
        InferenceRule(lambda i: i["pricerange"] == "expensive" and i["long time"], "good for meetings", True),
        InferenceRule(lambda i: i["seating outside"] and i["long time"], "fast service", False),
        InferenceRule(lambda i: i["pricerange"] == "expensive" and i["good atmosphere"], "romantic", True),
        InferenceRule(lambda i: i["long time"], "short time", False),
        InferenceRule(lambda i: i["short time"], "long time", False)
    ]


# load dialog_acts and restaurant_info, and begin chat with user.
def main():
    data_elements = DataElements("dialog_acts.dat")
    restaurant_info = RestaurantInfo("restaurant_info_v2.csv")
    transitioner = Transitioner(data_elements, restaurant_info)
    history = DialogHistory(restaurant_info)
    state = DialogState.Welcome(history)
    while state is not None:
        utterance = None
        if state.state_type == "USER":
            utterance = SentenceCleanser.cleanse(input(""))
            print(f"USER: {utterance}")
        state, sentence = transitioner.transition(state, utterance)
        if sentence is not None:
            print(f"SYSTEM: " + sentence.replace("\n", "\nSYSTEM: "))
    print("SYSTEM: Ok, good bye! Come again!")


if __name__ == "__main__":
    main()
