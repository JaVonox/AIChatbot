from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from Dataset import DatasetTrainer

# Create a new instance of a ChatBot
bot = ChatBot(
    'Example Bot',
    storage_adapter='chatterbot.storage.SQLStorageAdapter',
    logic_adapters=[
        {
            'import_path': 'chatterbot.logic.BestMatch',
            'default_response': 'Im sorry, I do not quite understand. Could you rephrase your question?',
            'maximum_similarity_threshold': 0.90
        }
    ]
)

selec = input("Train? (y/n)")

if selec == "y":
    trainer = DatasetTrainer.Trainer('DataSet/twcs.csv')
    trainer.TrainModel(bot)

# Get a response for some unexpected input
print("Hi, im a customer service robot. What is your question?")
while(True):
    response = bot.get_response(input())
    print(response)
