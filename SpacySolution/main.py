import spacy
import pandas as pd
from spacy.tokens import Doc
import warnings
warnings.filterwarnings("ignore")

class Topic:
    def __init__(self):
        self.type = None
        self.trainingData = []
        self.model = None

    def Append(self, item):
        self.trainingData.append(item)

    def ReturnTrain(self):
        return self.trainingData

    def CreateDoc(self,docs):
        self.model = Doc.from_docs(docs)

    def ReturnSim(self,inputDoc):
        return self.model.similarity(inputDoc)

topicData = {}
inputDataFrame = pd.read_csv (r'.\testdb.csv')
nlp = spacy.load("en_core_web_sm")

for iter,row in inputDataFrame.iterrows():
    type = row[0]
    contents = row[1]

    if str(type) not in topicData.keys():
        topicData[str(type)] = Topic()

    topicData[str(type)].Append(contents)

for topic in topicData:
    classItem = topicData[topic]
    bin = []
    trainingSet = classItem.ReturnTrain()

    for x in trainingSet:
        nlpDoc = nlp(x)
        bin.append(nlpDoc)

    topicData[topic].CreateDoc(bin)

while True:
    inVal = nlp(input())

    gTopic = "Unknown"
    gTopicScore = 0.0
    for top in topicData:
        comparisonVal = topicData[top].ReturnSim(inVal)

        if comparisonVal > gTopicScore:
            gTopic = top
            gTopicScore = comparisonVal

    if gTopicScore > 0.3:
        print("Expected topic: " + gTopic + " (similarity: " + str(round(gTopicScore,2)) + "%)")
    else:
        print("Unknown topic. Best guess: " + gTopic + " (similarity: " + str(round(gTopicScore,2)) + "%)")

