import spacy
import pandas as pd
from spacy.tokens import Doc
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

class Topic:
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

topicData = {}
inputDataFrame = pd.read_csv (r'.\Dataset\StackExchangeMoney.csv')
nlp = spacy.load("en_core_web_lg")

#Preprocessing
#inputDataFrame['TEXT'] = inputDataFrame['TEXT'].str.replace("@AppleSupport", "")


x_train, x_test, y_train, y_test = train_test_split(inputDataFrame["TYPE"],inputDataFrame["TEXT"], test_size=0.3)

train = pd.concat([x_train,y_train],axis=1)
test = pd.concat([x_test,y_test],axis=1)

for iter,row in train.iterrows():
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

score = 0
counter = 0
for iter,row in test.iterrows():
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
        print("Expected topic: " + gTopic + " (similarity: " + str(round(gTopicScore,2)*100) + "%)")
    else:
        print("Unknown topic. Best guess: " + gTopic + " (similarity: " + str(round(gTopicScore,2)*100) + "%)")

