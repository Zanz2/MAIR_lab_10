import pandas as pd


class SurveyRun:
    def __init__(self, frequency, complexity, ease, inconsistency, learnability, inconvenience, confidence,
                 intuitiveness):
        self.frequency = frequency
        self.complexity = complexity
        self.ease = ease
        self.inconsistency = inconsistency
        self.learnability = learnability
        self.inconvenience = inconvenience
        self.confidence = confidence
        self.intuitiveness = intuitiveness
    
    def __str__(self):
        return "{" + ', '.join(f"\"{str(k)}\": {str(v)}" for k, v in self.__dict__.items()) + "}"


class SurveyEntry:
    def __init__(self, row):
        self.group = row[1].lower()
        self.first = SurveyRun(row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
        self.second = SurveyRun(row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17])
        self.remarks1 = row[18]
        self.remarks2 = row[19]
    
    def __str__(self):
        return f"{{\"group\": \"{self.group}\", \"first\": {self.first}, \"second\": {self.second}, " \
               f"\"remarks1\": \"{self.remarks1}\", \"remarks2\": \"{self.remarks2}\"}}"


class Survey:
    def __init__(self, filename):
        self.filename = filename
        self.entries = self.__parse_data()
        
    def __parse_data(self):
        answers = []
        csv_data = pd.read_csv(self.filename, header=0, na_filter=False)
        for row in csv_data.values:
            row = [column for column in row]
            answers.append(SurveyEntry(row))
        return answers
    
    def __str__(self):
        return f"{{\"filename\": \"{self.filename}\", \"entries\": [{', '.join(str(e) for e in self.entries)}]}}"


x = Survey("participant_survey.csv")
print(x)
