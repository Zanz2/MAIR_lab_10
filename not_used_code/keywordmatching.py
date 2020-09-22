import re
import Levenshtein

# keyword matching + recognizing patterns, still misses levenshtein distance
class PatternAndMatch:
    def __init__(self, restaurant_info):
        self.restaurant_info = restaurant_info
        self.preference_values = {"food": [], "area": [], "pricerange": []}
        self.preference_dict = {
    "food_type" : [
        "chinese",
        "swedish",
        "tuscan",
        "international",
        "catalan",
        "cuban",
        "persian",
        "bistro",
        "world"
    ],
    "price_range" : [
        "expensive",
        "moderate",
        "cheap"
    ],
    "location" : [
        "center",
        "south",
        "west",
        "east",
        "north"
        ]
    }

        for restaurant in self.restaurant_info.restaurants:
            if restaurant.food not in self.preference_values["food"]:
                self.preference_values["food"].append(restaurant.food)
            if restaurant.area not in self.preference_values["area"]:
                self.preference_values["area"].append(restaurant.area)
            if restaurant.pricerange not in self.preference_values["pricerange"]:
                self.preference_values["pricerange"].append(restaurant.pricerange)

    def check_levenshtein(self, word, type = False):
            # allowed type : "food_type" "price_range" or "location"
            # type is food location or price range, each of these have their own minimum word length
            # based on the keyword matching words, for example if we misspell west as est, we still spellcheck est
            # if we would misspell it as st then we do not consider that

            # for general use the type is just falls, for correcting specific food types then
            # the type is supplied and each type has its own minimum length (see ifs below)
           blacklist = ["west", "would", "want", "world", "a", "part", "can", "what"]
           # words that get confused and changed easily frequently belong in the blacklist
           if word in blacklist or (type is False and len(word) < 3): # general words are only allowed if they are length 3 and up
               return False
           if (type == "food_type" and len(word) < 2) or (type == "price_range" and len(word) < 3) or (type == "location" and len(word) < 2):
               return False
           match_dict = {
               "correct_word": False,
               "type": False,
               "index": -1,
               "lv_distance": 4  # max allowed distance, if its 4 at the end we return false
           }

           loop_array = ["food_type", "price_range", "location"]

           for type_index, value_type in enumerate(loop_array):
               if type is not False and value_type != type:
                   continue
               for element in self.preference_dict[value_type]:
                   lv_distance = Levenshtein.distance(element, word)
                   if lv_distance < match_dict["lv_distance"]:
                       match_dict["lv_distance"] = lv_distance
                       match_dict["type"] = value_type
                       match_dict["index"] = type_index
                       match_dict["correct_word"] = element
                       if lv_distance < 3: print(word+" changed into "+element) # debug that prints the word changes
           if match_dict["lv_distance"] < 3:
               # i set this to less than 3, because more was giving me problems, with 3 this is what happened:
               '''
               for changed into world
               food changed into world
               for changed into world
               food changed into world
               '''
               return match_dict
           return False


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
            if word == "moderately":
                word = "moderate"
            if pref_dict["food"] is None:
                if word in self.preference_values["food"]:
                    pref_dict["food"] = word
            if pref_dict["area"] is None:
                if word in self.preference_values["area"]:
                    pref_dict["area"] = word
            if word in self.preference_values["pricerange"]:
                pref_dict["pricerange"] = word

        return pref_dict


bla = PatternAndMatch(None)
bla.check_levenshtein("cheap", type="food_type")
