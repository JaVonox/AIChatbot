import spacy
import pandas as pd
from spacy.tokens import Doc
from sklearn.model_selection import train_test_split
from spacytextblob.spacytextblob import SpacyTextBlob
import warnings
warnings.filterwarnings("ignore")

class AIType:
    def __init__(self):
        self.type = None
        self.trainingData = []
        self.testingData = []

        self.model = None

    def Append(self, item):
        self.trainingData.append(item)

    def ReturnTrain(self):
        return self.trainingData

    def CreateDoc(self,docs):
        self.model = Doc.from_docs(docs)

    def ReturnSim(self,inputDoc):
        return self.model.similarity(inputDoc)

def FormSentence(inVal,topic):
    sentence = ""

    polar = inVal._.blob.polarity #Gets the polarity of the input - score of negative to positive (-1 to 1)

    #Choose prefix to response based on how positive/negative the message is
    if polar <= -0.2:
        sentence = topicOutput[topicOutput["TYPE"] == topic]["PREFIX-SAD"].values[0]
    elif polar >= 0.2:
        sentence = topicOutput[topicOutput["TYPE"] == topic]["PREFIX-HAPPY"].values[0]
    else:
        sentence = topicOutput[topicOutput["TYPE"] == topic]["PREFIX-NEUTRAL"].values[0]

    subj = inVal._.blob.subjectivity #Gets the subjectivity of the input - score from objective (0) to subjective (1)

    #Choose response based on subjectivity (essentially check if the input is very personal or not)
    if subj <= 0.5:
        sentence = sentence + " " +  topicOutput[topicOutput["TYPE"] == topic]["TEXT-NORMAL"].values[0]
    else:
        sentence = sentence + " " + topicOutput[topicOutput["TYPE"] == topic]["TEXT-SUBJECTIVE"].values[0]

    print(sentence)

topicData = {}

inputDataFrame = pd.read_csv(r'.\Dataset\StackExchangeMoney.csv') #Load questions
topicOutput = pd.read_csv(r'.\Dataset\Responses.csv') #Load outputs
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('spacytextblob') #adds sentiment analysis + subjectivity to the pipeline

x_train_q, x_test_q, y_train_q, y_test_q = train_test_split(inputDataFrame["TYPE"],inputDataFrame["TEXT"], test_size=0.3) #create questions training data
train_q = pd.concat([x_train_q,y_train_q],axis=1)
test_q = pd.concat([x_test_q,y_test_q],axis=1)

for iter,row in train_q.iterrows():
    type = row[0]
    contents = row[1]

    if str(type) not in topicData.keys():
        topicData[str(type)] = AIType()

    topicData[str(type)].Append(contents)

for topic in topicData:
    classItem = topicData[topic]
    bin = []
    trainingSet = classItem.ReturnTrain()

    for x in trainingSet:
        nlpDoc = nlp(x)
        bin.append(nlpDoc)

    topicData[topic].CreateDoc(bin)

score = 0
counter = 0
for iter,row in test_q.iterrows():
    realType = row[0]
    contents = nlp(row[1])

    gTopic = "Unknown"
    gTopicScore = 0.0
    for top in topicData:
        comparisonVal = topicData[top].ReturnSim(contents)

        if comparisonVal > gTopicScore:
            gTopic = top
            gTopicScore = comparisonVal

    if gTopic == realType:
        score = score + 1
    counter = counter + 1

print("Accuracy: " + str(round((score / counter)*100,2)) + "%")

print("Hi! I'm the T. Bank virtual assistant! Please describe your problem and i'll do my best to direct you where you need to go!")

while True:
    inVal = nlp(input())

    gTopic = "UNKNOWN" #The predicted topic
    gSecTopic = "UNKNOWN" #The second predicted topic
    gTopicScore = 0.0
    gSecTopicScore = 0.0
    for top in topicData:
        comparisonVal = topicData[top].ReturnSim(inVal)

        if comparisonVal > gTopicScore:
            gSecTopic = gTopic
            gSecTopicScore = comparisonVal
            gTopic = top
            gTopicScore = comparisonVal
        elif comparisonVal > gSecTopicScore:
            gSecTopic = top
            gSecTopicScore = comparisonVal

    if gTopicScore > 0.5:
        #print("Expected topic: " + gTopic + " (similarity: " + str(round(gTopicScore,2)*100) + "%)")
        pass
    else:
        #print("Unknown topic. Best guess: " + gTopic + " (similarity: " + str(round(gTopicScore,2)*100) + "%)")
        gTopic = "UNKNOWN" #set back to unknown type to display unknown text

    FormSentence(inVal,gTopic)

    print("---")

    if gTopic != "UNKNOWN":

        if gSecTopic != "UNKNOWN" and gTopicScore < 0.9:
            print("I hope I was able to answer your question! I'm still learning, so please let me know if I got the topic of your question wrong so I can try again!")
            print("(Enter N if you'd like your question reinterpreted)")
            req = input("")

            if req == "N" or req == "n":
                FormSentence(inVal, gSecTopic)
                print("---")
                print("If I still got your question wrong, please try rephrasing the question and trying again")

        #If the topic is unknown, the user has already been asked to re-enter their question
        print("Is there anything else I can help you with?")
