import json
import sys
import math
import numpy as np
import pandas as pd
from random import shuffle
import re



class NaiveBayeClassifier:
    def __init__(self):
        self.class_probabilities = {}
        self.word_probabilities_by_class = {}
        self.vocabulary = set()
        self.category = []
        self.accuracy_recall_results = {}
        self.laplace_smoothing = 0
    
    def load_data(self,file_path):
        dataset = []
        #extracting the data
        with open(file_path,'r') as file:
            for line in file:
                json_line = json.loads(line.strip())
                dataset.append({
                    "category":json_line['category'],
                    "data" : self.preprocess_text(f"{json_line['headline']} {json_line['short_description']}")
                })


        #updating vocabulary and category at the same time
        for doc in dataset:
            if doc["category"] not in self.category:
                self.category.append(doc["category"])
            text = doc["data"]
            self.vocabulary.update(text)
        self.laplace_smoothing = math.log(1 / (len(self.vocabulary) + 1))
        #initialisationg accuracy_recall_results with category
        #initiate at 1 to prevent division by 0
        self.accuracy_recall_results = {category:{"true_positive":1,"false_positive":1,"false_negative":1} for category in self.category}

        return dataset
    
    def binarization(self,dataset):
        new_data = []
        for doc in dataset:
            new_data.append({"category":doc["category"], 
                             "data":set(doc["data"])})
        return new_data
        
    def preprocess_text(self,text):
        text = re.sub(r'[^\w\s]', '', text.lower())
        return text.split()
    
    def train_model(self,dataset):

        #arranging docs
        docs_by_category = {category:[] for category in self.category}
        self.class_probabilities = {category:0 for category in self.category}
        for doc in dataset:
            doc_category = doc["category"]
            self.class_probabilities[doc_category]+=1
            docs_by_category[doc_category].append(doc["data"])
        

        #probability of a document happening

        number_of_docs = sum(self.class_probabilities.values())
        self.class_probabilities = {category : math.log(doc_in_category/number_of_docs) for category, doc_in_category in self.class_probabilities.items()}

        #iterating trhough each word in each category to give them a probability

        for category, big_documents in docs_by_category.items():
            words_count = {}
            for doc in big_documents:
                for word in doc:
                    if word not in words_count.keys():
                        words_count[word]=0
                    words_count[word]+=1
                
            number_of_words = sum(words_count.values())
            self.word_probabilities_by_class[category] = {
                word : math.log((iterance+1)/(number_of_words+len(self.vocabulary))) for word, iterance in words_count.items() 
            }

    def prediction(self,text):
        #doc format is the same as the one in the dataset
        
        category_scores = self.class_probabilities.copy()


        for category in self.category:
            for word in text:
                category_scores[category] += self.word_probabilities_by_class[category].get(word, self.laplace_smoothing)
        
        return max(category_scores, key=category_scores.get)
    

    def results(self,dataset):
        

        for doc in dataset:
            predicted_category = self.prediction(doc["data"])
            actual_category = doc["category"]
            if predicted_category == actual_category:
                self.accuracy_recall_results[actual_category]["true_positive"]+=1
            else:
                self.accuracy_recall_results[actual_category]["false_negative"]+=1
                self.accuracy_recall_results[predicted_category]["false_positive"]+=1
                


def main():

    classifier = NaiveBayeClassifier()
    

    raw_data = classifier.load_data(sys.argv[1])
    shuffle(raw_data)
    raw_data = classifier.binarization(raw_data)


    datasets = [raw_data[int(len(raw_data)/5)*index:int(len(raw_data)/5)*(index+1)] for index in range(5)]
    #cross validation with 5 datasets


    for index in range(5):
        classifier.class_probabilities = {}
        classifier.word_probabilities_by_class = {}
        actual_training = [item for minilist in datasets[:index]+datasets[index+1:] for item in minilist]
        classifier.train_model(actual_training)
        print(f"moitié{index}")
        classifier.results(datasets[index])
        print(f"fin {index}")


    for category, data in classifier.accuracy_recall_results.items():
        recall = data["true_positive"]/(data["true_positive"]+data["false_negative"])
        precision = data["true_positive"]/(data["true_positive"]+data["false_positive"])
        f1_score = 2*precision*recall/(precision+recall)
        print(f"Results for {category} :    -recall ={recall:.4f}       -precision={precision:.4f}      -f1 score={f1_score:.4f}")
    


    

if __name__ == "__main__":
    main()