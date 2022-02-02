import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import csv


#Class used to classify rotten tomato reviews
class Classifier:
    
    #Function to classify data, returns dictionary mapping sentence num -> predicted sentiment
    def classify(self):
        self.s_count = self.sentiment_count(self.convert_to_3)
        
        self.load_training_data()

        self.load_dev_data()

        if(self.test_data):
            predictions = self.predict_test()
        else:
            predictions = self.predict()
        return predictions
    
    #Return the sentiment counts
    def sentiment_count(self, convert_to_3):
        if(convert_to_3):
            return 3
        return 5
    
    #Load the training data, map 5 to 3, aswell as calculate priors
    def load_training_data(self):
        self.train_data = pd.read_csv("moviereviews/train.tsv", sep='\t')
        self.train_data["Phrase"] = self.train_data["Phrase"].str.lower()
        self.training_count = len(self.train_data)
        if(self.convert_to_3):
            self.map_5_to_3(self.train_data)
        self.priors = self.get_priors()
        self.count_word_for_each_class()  
    
    #Loads data to be classified
    def load_dev_data(self):
        if(self.test_data):
            self.dev_data = pd.read_csv("moviereviews/test.tsv", sep='\t')
            self.dev_data["Phrase"] = self.dev_data["Phrase"].str.lower()
        else:
            self.dev_data = pd.read_csv("moviereviews/dev.tsv", sep='\t')
            self.dev_data["Phrase"] = self.dev_data["Phrase"].str.lower()
            #If data is development, map 5 sentiment class to 3
            if(self.convert_to_3):
                self.map_5_to_3(self.dev_data)
            
    #Maps 5 sentiment classes to 3
    def map_5_to_3(self, data):
             for index, num, phrase, sentiment in data.itertuples():
                 if(sentiment==0 or sentiment==1):
                     data.at[index, "Sentiment"] = 0
                 elif(sentiment==2):
                     data.at[index, "Sentiment"] = 1
                 else:
                     data.at[index, "Sentiment"] = 2          
                     

    #Function used to calculate priors  
    #Counts occurences of sentiment class in training data
    def get_priors(self):
       priors = {}
       priors[0] = 0
       priors[1] = 0
       priors[2] = 0
       if(not self.convert_to_3):
           priors[3] = 0
           priors[4] = 0
           
       for index, num, phrase, sentiment in self.train_data.itertuples():
           if(sentiment==0):
               priors[0] += 1
           elif(sentiment==1):
               priors[1] += 1
           elif(sentiment==2):
               priors[2] += 1      
           elif(not self.convert_to_3):
               if(sentiment==3):
                   priors[3] += 1    
               else:
                   priors[4] += 1     
       return self.calculate_priors(priors)           
                
    #Calculates priors
    def calculate_priors(self, priors):
        for sentiment_class in priors:
            priors[sentiment_class] = priors[sentiment_class] / self.training_count
        return priors
            
    #Counts the words for each sentiment class
    def count_word_for_each_class(self):
        self.counts = {}
        self.counts[0] = {}
        self.counts[1] = {}
        self.counts[2] = {}
        if(not self.convert_to_3):
            self.counts[3] = {}
            self.counts[4] = {}
        for index, num, phrase, sentiment in self.train_data.itertuples():
            words = nltk.word_tokenize(phrase)
            #Checks to see if any processing functions need to be peformed, depending on the model
            if(self.binary and self.negation_function):
                words = self.convert_list_binary(self.negate_word_list(words)) 
            elif(self.binary):
                words = self.convert_list_binary(words)
            elif(self.negation_function):
                words = self.negate_word_list(words)
                
            for word in words:
                #Checks to see if any word lists are chosen
                if(self.use_positive_and_negative_list):
                    if (word in self.positive_list) or (word in self.negative_list):
                        if word not in self.counts[sentiment]:
                            self.counts[sentiment][word] = 1
                        else:
                            self.counts[sentiment][word] += 1
                elif(self.use_adjective_list):
                    if (word in self.adjective_list):
                        if word not in self.counts[sentiment]:
                            self.counts[sentiment][word] = 1
                        else:
                            self.counts[sentiment][word] += 1
                else:
                    if word not in self.counts[sentiment]:
                        self.counts[sentiment][word] = 1
                    else:
                        self.counts[sentiment][word] += 1
    
                                     
                      
    #Prediction function for dev data
    def predict(self):
        predictions = {}
        for index, num, phrase, sentiment in self.dev_data.itertuples():
            predictions[num] = {}
            phrase_features =nltk.word_tokenize(phrase)
            #Peforms functions needed, depending on model config
            if(self.binary and self.negation_function):
                phrase_features = self.convert_list_binary(self.negate_word_list(phrase_features)) 
            elif(self.binary):
                phrase_features = self.convert_list_binary(phrase_features)
            elif(self.negation_function):
                phrase_features = self.negate_word_list(phrase_features)
                
            for sentiment in range(self.s_count):
                likelihood_product = 1
                for feature in phrase_features:
                    #Checks to see if any word lists are chosen
                    if(self.use_positive_and_negative_list):
                        if (feature in self.positive_list) or (feature in self.negative_list):
                            likelihood_product = likelihood_product * self.compute_likelihood_smooth(feature, sentiment)
                    elif(self.use_adjective_list):
                        if (feature in self.adjective_list):
                            likelihood_product = likelihood_product * self.compute_likelihood_smooth(feature, sentiment)
                    else:
                        likelihood_product = likelihood_product * self.compute_likelihood_smooth(feature, sentiment)
                predictions[num][sentiment] = likelihood_product * self.priors[sentiment]   
        return (self.get_max_predictions(predictions))

    #Prediction function for test data
    #The same as normal prediction function, but removes sentiment column when looping through panda datastructure
    def predict_test(self):
        predictions = {}
        for index, num, phrase in self.dev_data.itertuples():
            predictions[num] = {}
            phrase_features =nltk.word_tokenize(phrase)
            #Peforms functions needed, depending on model config
            if(self.binary and self.negation_function):
                phrase_features = self.convert_list_binary(self.negate_word_list(phrase_features)) 
            elif(self.binary):
                phrase_features = self.convert_list_binary(phrase_features)
            elif(self.negation_function):
                phrase_features = self.negate_word_list(phrase_features)
                
            for sentiment in range(self.s_count):
                likelihood_product = 1
                for feature in phrase_features:
                    #Checks to see if any word lists are chosen
                    if(self.use_positive_and_negative_list):
                        if (feature in self.positive_list) or (feature in self.negative_list):
                            likelihood_product = likelihood_product * self.compute_likelihood_smooth(feature, sentiment)
                    elif(self.use_adjective_list):
                        if (feature in self.adjective_list):
                            likelihood_product = likelihood_product * self.compute_likelihood_smooth(feature, sentiment)
                    else:
                        likelihood_product = likelihood_product * self.compute_likelihood_smooth(feature, sentiment)
                predictions[num][sentiment] = likelihood_product * self.priors[sentiment]   
        return (self.get_max_predictions(predictions))
                
    
    #Given a feature and sentiment, returns the likelihood
    #Uses laplace smoothing
    def compute_likelihood_smooth(self, feature, sentiment):
        values = self.counts[sentiment].values()
        sentiment_word_count = len(values)
        total = sum(values)
        if feature in self.counts[sentiment]:
            return (self.counts[sentiment][feature] + 1) / (total + sentiment_word_count)      
        else:
            return 1 / (total + sentiment_word_count) 
               
    #Given a dictionary of predictions, it returns a dictionary
    #Dictionary returned maps num -> predicted sentiment
    def get_max_predictions(self, predictions):
        chosen = {}
        for num in predictions:
            chosen[num] = max(predictions[num], key = predictions[num].get)
        return chosen
    
    #Used to get rid of reoccuring words in a word_list
    def convert_list_binary(self, word_list):
        return list(dict.fromkeys(word_list))
    
    #Used to negate wordlists
    #Takes word_list as input
    #If a negation word is detected, all the words following will be negated
    #Up until punctuation is reached
    def negate_word_list(self, word_list):
        negation = False
        for index, word in enumerate(word_list):
            if word in self.punctuation_list:
                negation = False
            if (negation):
                word_list[index] = self.negation + word
            if word in self.negation_list:
                negation = True
        return word_list     
               
    #Creates confusion matrix
    #Returns a numpy array      
    def make_cm(self):
        cm = np.zeros((self.s_count,self.s_count))
        for index, num, phrase, true_sentiment in self.dev_data.itertuples():
            cm[true_sentiment][self.predictions[num]] += 1
        return cm
            
                
    #Used to plot the confusion matrix, using matplotlib
    def plot_confusion(self, cm_data, title="confusion matrix"):
        accuracy = np.trace(cm_data) / float(np.sum(cm_data))
        misclass = 1 - accuracy
        
        plt.figure(figsize = (8,6))
        plt.imshow(cm_data, interpolation='nearest', cmap=None)
        plt.title(title)
        plt.colorbar()
        
        thresh = cm_data.max() / 1.5
        
        plt.grid(False)
        plt.tight_layout
        plt.ylabel('True label')
        plt.xlabel('Predicted lable')
        plt.show()
        
    #Calculates all f1_scores and macro f1 score
    #Returns macro_f1 as float
    #Returns f_measures as dictionary mapping sentiment -> f1 score
    def calculate_f_measures(self):
        f_measures = {}
        f_score_sum = 0 
        self.precision = {}
        self.recall = {}
        for sentiment in range(self.s_count):
            self.precision[sentiment] = self.get_precision_or_recall(True, sentiment)
            self.recall[sentiment] = self.get_precision_or_recall(False, sentiment)
            precision = self.get_precision_or_recall(True, sentiment)
            recall = self.get_precision_or_recall(False, sentiment)
            f_measures[sentiment] = (2 * precision * recall)/(precision + recall)
            f_score_sum += f_measures[sentiment]
        macro_f1 = f_score_sum/self.s_count
        return f_measures, macro_f1
       
    #Method to calculate precision or recall
    #Params, get_precision, if true returns precision, if false returns recall
    #Calculates depending on given sentiment number
    def get_precision_or_recall(self,get_precision,sentiment_number):
        sentiment_count = self.cm.shape[0]
        numerator = self.cm[sentiment_number][sentiment_number]
        denominator = 0
        for sentiment in range(sentiment_count):
            if(get_precision):
                denominator += self.cm[sentiment_number][sentiment]
            else:
                denominator += self.cm[sentiment][sentiment_number]
        return (numerator/denominator)
    
    #Used to read word list into system
    #Takes file name as input, returns word set
    def readList(self, file):
        word_set = set()
        f = open(file, 'r')
        for line in f:
            word_set.add(line.strip())
        return word_set
    
    #Used to output results to tsv file
    #Takes file name as input
    def output_results(self, file_name ):
        df = pd.DataFrame(columns= ['SentenceId', 'Sentiment'])
        for x in self.predictions:
            df = df.append({"SentenceId": x, "Sentiment":self.predictions[x]}, ignore_index=True)
        df.to_csv(file_name, index=False, sep="\t")

                
#Initialise classifier
self = Classifier()

#Positive and negative lists
self.positive_list = self.readList('positive_list.txt')   
self.negative_list = self.readList('negative_list.txt')   

#Adjective word lists
self.adjective_list = self.readList('adjective_list.txt')   

#Used to detect negation
self.negation_list = self.readList('negation_words.txt')
#Used when negating text, to detect end of negation
self.punctuation_list = self.readList('punctuation.txt')

#Used to convert negated text
self.negation = 'NOT_'

#Model features
self.convert_to_3 = True
self.binary = False
self.negation_function = False
self.use_adjective_list = False
self.use_positive_and_negative_list = False

#Use test data or not
self.test_data = False

#Run the classifier
self.predictions = self.classify()   

#If not using test data, create evaluate data
if(not self.test_data):
    self.cm = self.make_cm() 
    self.f_measures, self.macro_f1 = self.calculate_f_measures()
    #Print evalutation data to console
    print(self.cm)
    print("Precisions- ",self.precision)
    print("Recalls- ",self.recall)
    print("F1 scores- ",self.f_measures)
    print("Macro f1- ",self.macro_f1)
    #Plot confusion matrix
    self.plot_confusion(self.cm)
    
#If you want to output to tsv file, uncomment this
#self.output_results("file_name.tsv")




                        

  