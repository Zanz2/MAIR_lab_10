import os


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


def main():
    directory = os.path.dirname(os.path.realpath(__file__))
    directory = "\\".join(directory.split("\\")[:-1])
    dialog_data = DialogData(f"{directory}\\all_dialogs.txt")
    for i in range(10):
        print(dialog_data.dialogs[i])


if __name__ == "__main__":
    main()
