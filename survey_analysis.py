import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


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
    
    def score(self, question):
        if question in self.__dict__:
            return self.__dict__[question]
        raise NotImplementedError
        
    def adjusted_score(self, question):
        if question in self.__dict__:
            if question in ("frequency", "ease", "learnability", "confidence"):
                return self.__dict__[question]
            else:
                return 6 - self.__dict__[question]
        raise NotImplementedError
    
    def __str__(self):
        return "{" + ', '.join(f"\"{str(k)}\": {str(v)}" for k, v in self.__dict__.items()) + "}"


class SurveyEntry:
    def __init__(self, row):
        self.group = row[1].lower()
        self.first = SurveyRun(row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
        self.second = SurveyRun(row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17])
        self.remarks1 = row[18]
        self.remarks2 = row[19]
    
    def get_group(self, with_implicit_confirmation):
        if self.group == "a" and with_implicit_confirmation:
            return self.second
        elif self.group == "a" and not with_implicit_confirmation:
            return self.first
        elif self.group == "b" and with_implicit_confirmation:
            return self.first
        elif self.group == "b" and not with_implicit_confirmation:
            return self.second
        raise NotImplementedError()
        
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


def paired_sample_ttest(save=False):
    survey = Survey("participant_survey.csv")
    data = {"total: satisfaction": ([], [])}
    for participant in survey.entries:
        for i, condrun in enumerate((participant.get_group(True), participant.get_group(False))):
            satisfaction = 0
            for j, question in enumerate(condrun.__dict__):
                name = f"q{j + 1}: {question}"
                if name not in data:
                    data[name] = ([], [])
                data[name][i].append(condrun.score(question))
                satisfaction += condrun.adjusted_score(question)
            data["total: satisfaction"][i].append(satisfaction / len(condrun.__dict__))
    ttests = {}
    for category in data:
        with_conf, without_conf = data[category]
        ttests[category] = stats.ttest_rel(without_conf, with_conf)
    categories = [c for c in ttests]
    ticks = [i for i in range(len(categories))]
    fig, axes = plt.subplots(2, 1, sharex="all")
    axes[0].bar(ticks, [ttests[c][0] for c in categories])
    axes[1].bar(ticks, [ttests[c][1] for c in categories], color="orange")
    plt.xticks(ticks, categories)
    axes[1].set_xticklabels(categories, rotation=45)
    axes[1].hlines(0.05, -0.5, len(categories) - 0.5, alpha=0.2)
    axes[0].tick_params(axis="both", which="both", labelsize=8)
    axes[1].tick_params(axis="both", which="both", labelsize=8)
    axes[0].set_title("Paired T-test statistics", fontsize=8)
    axes[1].set_title("Paired T-test p-values", fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(f"paired_sample_ttest_results", dpi=150)
    print(data)


def main():
    paired_sample_ttest(True)


if __name__ == "__main__":
    main()
