import pandas as pd
import csv
import random as rnd

# do 3 new states (system ask, user input, eval that they are known)
# keyword matching of user input (secondary preferences)
# rework inference rules class (lambda thing)



# implement six predefined rules (specified)
# implement six of my own rules (for own rules you can add properties to the dataset, but only for 3 rules)
# of the above 6 rules at least 3 should be aplicable only after the first iteration (see table)
# collect all recommended restaurants at the end, for each of the recommended restaurants
# apply all rules for which the properties of the restaurant satisfy the antecedent of the rule,
# this will give you newly generated rules (antecedents), with these repeat again to get new rules,
# repeat until you do not get any new information with additonal iterations
# present the reasoning steps and the conclusion (the restaurant does or does not satisfy the additional requirements)
old_csv = "restaurant_info_original.csv"
new_csv = "restaurant_info_v2.csv"

def alter_restaurants(file):
    csv_input = pd.read_csv(file)
    # 4 rules added to db, 8 have to be inferred
    csv_input["goodfood"] = "False"
    csv_input["goodatmosphere"] = "False"
    csv_input["bigbeverageselection"] = "False"
    csv_input["spacious"] = "False"
    for index, row in csv_input.iterrows():
        if rnd.random() > 0.33:
            row["goodfood"] = "True"
        if rnd.random() > 0.33:
            row["goodatmosphere"] = "True"
        if rnd.random() > 0.33:
            row["bigbeverageselection"] = "True"
        if rnd.random() > 0.33:
            row["spacious"] = "True"
    csv_input.to_csv(new_csv, index=False, quoting=csv.QUOTE_NONNUMERIC)


#alter_restaurants(old_csv)

class MockRestaurant:
    def __init__(self, restaurantname, pricerange, area, food, phone, addr, postcode, goodfood, goodatmosphere, bigbeverageselection, spacious):
        self.items = {
            "restaurantname": restaurantname,
            "pricerange": pricerange,
            "area": area,
            "food": food,
            "phone": phone,
            "addr": addr,
            "postcode": postcode,
            "goodfood": goodfood,
            "goodatmosphere": goodatmosphere,
            "bigbeverageselection": bigbeverageselection,
            "spacious": spacious,
            "busy": None,
            "longtime": None,
            "shorttime": None,
            "children": None,
            "romantic": None,
            "fastservice": None,
            "seatingoutside": None,
            "goodformeetings": None,
            "goodforstudying": None
        }
        self.inference_rules_history = []  # This array will contain the order of the inference rules that were used

    def apply_inferred_rules(self): # FIXME make this into a class
        items = self.items
        for _ in range(3):
            if items["bigbeverageselection"] and items["goodatmosphere"]:
                items["longtime"] = True
                self.inference_rules_history.append("rule 1")
            if items["goodatmosphere"] and items["goodfood"]:
                items["busy"] = True
                self.inference_rules_history.append("rule 2")
            if items["pricerange"] == "cheap" and items["goodfood"]:
                items["busy"] = True
                self.inference_rules_history.append("rule 3")
            if items["fastservice"] and items["pricerange"] == "cheap":
                items["shorttime"] = True
                self.inference_rules_history.append("rule 4")
            if items["food"] == "spanish":
                items["longtime"] = True
                self.inference_rules_history.append("rule 5")
            if items["busy"]:
                items["longtime"] = True
                self.inference_rules_history.append("rule 6")
            if items["longtime"]:
                items["children"] = False
                self.inference_rules_history.append("rule 7")
            if items["busy"]:
                items["romantic"] = False
                self.inference_rules_history.append("rule 8")
            if items["longtime"]:
                items["romantic"] = True
                self.inference_rules_history.append("rule 9")
            if items["shorttime"]:
                items["children"] = True
                self.inference_rules_history.append("rule 10")
            if items["children"]:
                items["goodforstudying"] = False
                self.inference_rules_history.append("rule 11")
            if items["children"]:
                items["goodformeetings"] = False
                self.inference_rules_history.append("rule 12")
            if items["goodatmosphere"] and items["spacious"]:
                items["goodforstudying"] = False
                self.inference_rules_history.append("rule 13")
            if items["seatingoutside"] and items["goodatmosphere"]:
                items["romantic"] = True
                self.inference_rules_history.append("rule 14")
            if items["longtime"]:
                items["goodforstudying"]
                self.inference_rules_history.append("rule 15")
            if items["longtime"]:
                items["goodformeetings"]
                self.inference_rules_history.append("rule 16")
            if items["pricerange"] == "expensive" and items["shorttime"]:
                items["busy"] = False
                self.inference_rules_history.append("rule 17")
            if items["pricerange"] == "moderate" and items["longtime"]:
                items["goodforstudying"] = True
                self.inference_rules_history.append("rule 18")
            if items["pricerange"] == "expensive" and items["longtime"]:
                items["goodformeetings"] = True
                self.inference_rules_history.append("rule 19")
            if items["seatingoutside"] and items["longtime"]:
                items["fastservice"] = False
                self.inference_rules_history.append("rule 20")
            if items["pricerange"] == "expensive" and items["goodatmosphere"]:
                items["romantic"] = True
                self.inference_rules_history.append("rule 21")
            if items["shorttime"]:
                items["longtime"] = False
                self.inference_rules_history.append("rule 22")
            if items["longtime"]:
                items["shorttime"] = False
                self.inference_rules_history.append("rule 23")
        self.items = items

    def __str__(self):
        return str(self.items)



class MockRestaurantInfo:
    def __init__(self, filename):
        self.filename = filename
        self.restaurants = self.__parse_data()

    def __parse_data(self):
        restaurants = []
        csv_data = pd.read_csv(self.filename, header=0, na_filter=False)
        for row in csv_data.values:
            restaurantname, pricerange, area, food, phone, addr, postcode, goodfood, goodatmosphere, bigbeverageselection, spacious = row
            restaurants.append(MockRestaurant(restaurantname, pricerange, area, food, phone, addr, postcode, goodfood, goodatmosphere, bigbeverageselection, spacious))
        return restaurants

test_info = MockRestaurantInfo(new_csv)
restaurants = test_info.restaurants

for restaurant in restaurants:
    restaurant.apply_inferred_rules()

for restaurant in restaurants:
    print("test")

