from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import csv #CSV file reader

class Trainer:
    def __init__(self,dataName):
        self.trainingData = [[]] #List containing lists of each training dataset
        self.LoadInData(dataName) #Add training data to model

    def LoadInData(self,dataName):
        trainFile = open(dataName)
        reader = csv.reader(trainFile)
        #File is processed to two fields - inbound and text
        #if inbound is true, it is a question, if inbound is false it is a response
        #A training item should consist of one true text followed by one false text

        lastInput = ""
        lastIsQuestion = False
        for row in reader:
            #Iterate through each row. If a row is an inbound store it in the input and update lastIsQuestion
            #If lastIsQuestion is true, then store both the input and the response in the training data. If not, replace data.
            if row[0] == "TRUE":
                lastInput = self.FormatString(row[1])
                lastIsQuestion = True
            elif row[0] == "FALSE" and lastIsQuestion == True:
                self.trainingData.append([lastInput,self.FormatString(row[1])]) #Add training data
                lastInput = ""
                lastIsQuestion = False

        print("Training data formatted")

    def FormatString(self,stringItem): #Removes all @ references
        sepWord = stringItem.split(' ')
        completeString = ""
        for x in sepWord:
            if "@" not in x:
                completeString += x + " "

        return completeString

    def TrainModel(self,botToTrain): #Train on each conversation in turn
        print("Training model")
        trainer = ListTrainer(botToTrain,show_training_progress=False)
        iter = 0.0
        progTotal = float(len(self.trainingData)) / 100.0
        progVal = 0

        for x in self.trainingData:
            iter = iter + 1.0
            trainer.train(x)
            if iter >= progTotal:
                progVal = progVal + 1 #1% increments
                print("Training progress: " + str(progVal) + "%")
                iter = 0.0

        print("Trained model on " + str(len(self.trainingData)) + " conversations")

