import spacy
import pandas as pd
from spacy.tokens import Doc
from sklearn.model_selection import train_test_split
from spacytextblob.spacytextblob import SpacyTextBlob
import warnings
warnings.filterwarnings("ignore") #A lot of the libraries here print out warnings for things like future depreciation. Since this isnt an issue, we can just disable them.

class AIType: #Stores the model training data + the doc file
    def __init__(self):
        self.trainingData = []
        self.model = None

    def Append(self, item): #add to training data
        self.trainingData.append(item)

    def ReturnTrain(self): #Return the training data
        return self.trainingData

    def CreateDoc(self,docs): #Take all docs and convert into a single doc
        self.model = Doc.from_docs(docs)

    def ReturnSim(self,inputDoc): #Return the similarity score against an input from a trained model
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

def RemoveBadTokens(text): #Remove the stop words and punctuation from the text
    preProcText = nlp(text) #convert the text to doc preemptively
    concat = [] #set of the text tokens
    for t in preProcText: #Iterate through each token and remove the stop words and punctuation
        if t.text not in nlp.Defaults.stop_words and not t.is_punct:
            concat.append(t.text)
    outStr = " ".join(concat) #rejoin the text items and then return the string to be processed again
    return outStr

def ReturnTopicPrediction(docInput): #Returns the top two similarity topics
    gTopic = "UNKNOWN" #The predicted topic
    gSecTopic = "UNKNOWN" #The second predicted topic
    gTopicScore = 0.0 #The score of the predicted topic
    gSecTopicScore = 0.0 #The score of the second prediction topic
    for top in topicData: #Iterate through each topic model and store the first and second highest scoring topics
        comparisonVal = topicData[top].ReturnSim(docInput)

        if comparisonVal > gTopicScore: #If greater than the current stored topic score, assign this as the new primary topic
            gSecTopic = gTopic
            gSecTopicScore = comparisonVal
            gTopic = top
            gTopicScore = comparisonVal
        elif comparisonVal > gSecTopicScore: #If greater than the current second topic score, assign this as the secondary topic
            gSecTopic = top
            gSecTopicScore = comparisonVal

    return gTopic,gTopicScore,gSecTopic,gSecTopicScore

topicData = {}

inputDataFrame = pd.read_csv(r'.\Dataset\StackExchangeMoney.csv') #Load questions
topicOutput = pd.read_csv(r'.\Dataset\Responses.csv') #Load outputs
nlp = spacy.load("en_core_web_lg")
nlp.add_pipe('spacytextblob') #adds sentiment analysis + objectivity to the pipeline

x_train_q, x_test_q, y_train_q, y_test_q = train_test_split(inputDataFrame["TYPE"],inputDataFrame["TEXT"], test_size=0.3) #create questions training data

#convert the training and testing data back into a dataframe each
train_q = pd.concat([x_train_q,y_train_q],axis=1)
test_q = pd.concat([x_test_q,y_test_q],axis=1)

for iter,row in train_q.iterrows(): #iterate through each row in the training data, storing the data in the appropriate topic model
    type = row[0]
    contents = row[1]

    if str(type) not in topicData.keys():
        topicData[str(type)] = AIType()

    topicData[str(type)].Append(RemoveBadTokens(contents))

for topic in topicData: #Concatenate all the training data for each topic and then collapse said data into a single doc file
    classItem = topicData[topic]
    bin = [] #Storage of all the processed items, to add to collapse to a single doc afterwards
    trainingSet = classItem.ReturnTrain()

    for x in trainingSet:
        nlpDoc = nlp(x)
        bin.append(nlpDoc)

    topicData[topic].CreateDoc(bin) #Make all doc files collapse into one per topic

score = 0 #Stores the amount of correct guesses
counter = 0 #Stores the amount of rows iterated through
secScore = 0 #Second guess correct guesses
secCounter = 0 #Second guess counter

for iter,row in test_q.iterrows(): #Iterate through each test item, getting the highest similarity score and assigning it as the expected result.
    realType = row[0]
    contents = nlp(RemoveBadTokens(row[1]))

    expTopic,expTopicScore,sTopic,sTopicScore = ReturnTopicPrediction(contents)

    if expTopic == realType: #Evaluate if the prediction was correct
        score = score + 1
    else:
        if sTopic == realType:
            secScore = secScore + 1
        secCounter = secCounter + 1
    counter = counter + 1

print("First Guess Accuracy: " + str(round((score / counter)*100,2)) + "%") #Print the first guess accuracy score
print("Second Guess Accuracy: " + str(round((secScore / secCounter)*100,2)) + "%") #Print the second guess accuracy score

print("Hi! I'm the T. Bank virtual assistant! Please describe your problem and i'll do my best to direct you where you need to go!")

while True:
    inVal = nlp(RemoveBadTokens(input())) #Take a user input and do feature selection & pre-processing to remove the invalid tokens

    expTopic,expTopicScore,sTopic,sTopicScore = ReturnTopicPrediction(inVal)

    if expTopicScore > 0.5:
        #print("Expected topic: " + gTopic + " (similarity: " + str(round(gTopicScore,2)*100) + "%)")
        pass
    else:
        #print("Unknown topic. Best guess: " + gTopic + " (similarity: " + str(round(gTopicScore,2)*100) + "%)")
        gTopic = "UNKNOWN" #set back to unknown type to display unknown text

    FormSentence(inVal,expTopic) #Append the emotional prefix + objective response

    print("---") #Divider

    if expTopic != "UNKNOWN":

        if sTopic != "UNKNOWN" and expTopicScore < 0.9: #If the model is not entirely sure, then give the option to provide a reevaluation of the question
            print("I hope I was able to answer your question! I'm still learning, so please let me know if I got the topic of your question wrong so I can try again!")
            print("(Enter N if you'd like your question reinterpreted)")
            req = input("")

            if req == "N" or req == "n": #If the user requests reinterpretation
                FormSentence(inVal, sTopic) #Reinterpret the question with the second guess
                print("---")
                print("If I still got your question wrong, please try rephrasing the question and trying again")

        #If the topic is unknown, the user has already been asked to re-enter their question
        print("Is there anything else I can help you with?")
