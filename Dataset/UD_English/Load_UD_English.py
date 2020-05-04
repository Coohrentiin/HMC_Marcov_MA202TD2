import numpy as np

def load_train_ud_english(path = "./"):
    
    with open(path + "en-ud-train.conllu") as file:
        
        sentences = []
        
        sent = []
        for line in file:
            
            if line == "\n":
                sentences.append(sent)
                sent = []
                
            else:
                elements = line.split("\t")
                word = elements[1]
                tag = elements[3]
                
                sent.append((tag, word))
                
        return sentences
    
def load_test_ud_english(path = "./"):
    
    with open(path + "en-ud-test.conllu") as file:
        
        sentences = []
        
        sent = []
        for line in file:
            
            if line == "\n":
                sentences.append(sent)
                sent = []
                
            else:
                elements = line.split("\t")
                word = elements[1]
                tag = elements[3]
                
                sent.append((tag, word))
                
        return sentences

def load_dev_ud_english(path = "./"):
    
    with open(path + "en-ud-dev.conllu") as file:
        
        sentences = []
        
        sent = []
        for line in file:
            
            if line == "\n":
                sentences.append(sent)
                sent = []
                
            else:
                elements = line.split("\t")
                word = elements[1]
                tag = elements[3]
                
                sent.append((tag, word))
                
        return sentences
            
    
def load_ud_english(path = "./", only_known_test = False, proportion_train = 1):
    train_set = load_train_ud_english(path) + load_dev_ud_english(path)
    test_set = load_test_ud_english(path)
    
    return train_set, test_set
            