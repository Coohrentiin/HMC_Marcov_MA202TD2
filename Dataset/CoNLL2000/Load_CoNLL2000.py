import numpy as np

def convert_ptp_to_universal(ptp_tag):
    struct = {
        "#": "SYM",
        "$": "SYM",
        "''": "PUNCT",
        "(": "PUNCT",
        ")": "PUNCT",
        ",": "PUNCT",
        ".": "PUNCT",
        ":": "PUNCT",
        "``": "PUNCT",
        "CC": "CCONJ",
        "CD": "NUM",
        "DT": "DET",
        "EX": "PRON",
        "FW": "X",
        "IN": "ADP",
        "JJ": "ADJ",
        "JJR": "ADJ",
        "JJS": "ADJ",
        "LS": "X",
        "MD": "VERB",
        "NN": "NOUN",
        "NNS": "NOUN",
        "NNP": "PROPN",
        "NNPS": "PROPN",
        "PDT": "DET",
        "POS": "PART",
        "PRP": "PRON",
        "PRP$": "DET",
        "RB": "ADV",
        "RBR": "ADV",
        "RBS": "ADV",
        "RP": "ADP",
        "SYM": "SYM",
        "TO": "PART",
        "UH": "INTJ",
        "VB": "VERB",
        "VBD": "VERB",
        "VBG": "VERB",
        "VBN": "VERB",
        "VBP": "VERB",
        "VBZ": "VERB",
        "WDT": "DET",
        "WP": "PRON",
        "WP$": "DET",
        "WRB": "ADV",
        
        "B-NP\n": "NP",
        "B-PP\n": "PP",
        "O\n": "O",
        "I-NP\n": "NP",
        "I-PP\n": "PP",
        "B-VP\n": "VP",
        "I-VP\n": "VP",
        "B-SBAR\n": "SBAR",
        "B-ADJP\n": "ADJP",
        "I-ADJP\n": "ADJP",
        "B-ADVP\n": "ADVP",
        "I-ADVP\n": "ADVP",
        "B-INTJ\n": "INTJ",
        "I-PRT\n": "PRT",
        "I-SBAR\n": "SBAR",
        "I-UCP\n": "UCP",
        "B-PRT\n": "PRT",
        "B-LST\n": "LST",
        "I-CONJP\n": "CONJP",
        "B-CONJP\n": "CONJP",
        "I-INTJ\n": "INTJ",
        "B-UCP\n": "UCP"
    }
    
    if ptp_tag in struct:
        return struct[ptp_tag]
    else:
        return ptp_tag

def load_train_conll2000(path = "./", universal = True, option = "pos"):
    tagged_sentences = []
    
    if option == "pos":
        index_option = 1
    elif option == "chunk":
        index_option = 2
    
    sent = []
    with open(path + "train.txt", "r") as file:
        for x in file:
            if x != "\n":
                elements = x.split(" ")
                
                if elements[0] == "-LRB-" or elements[0] == "-LCB-":
                    elements[0] = "("
                elif elements[0] == "-RRB-" or elements[0] == "-RCB-":
                    elements[0] = ")"
                    
                word = elements[0]
                    
                if option == "all":
                    if universal:
                        pos = convert_ptp_to_universal(elements[1])
                        chunk = convert_ptp_to_universal(elements[2])
                    else:
                        tag = elements[index_option]
                        
                    sent.append((pos, chunk, word))
                
                else:
                    if universal:
                        tag = convert_ptp_to_universal(elements[index_option])
                    else:
                        tag = elements[index_option]
                    
                    sent.append((tag, word))
                    
            else:
                tagged_sentences.append(sent)
                sent = []
    
    return tagged_sentences

def load_test_conll2000(path = "./", universal = True, option = "pos"):
    tagged_sentences = []
    
    if option == "pos":
        index_option = 1
    elif option == "chunk":
        index_option = 2
    
    sent = []
    with open(path + "test.txt", "r") as file:
        for x in file:
            if x != "\n":
                elements = x.split(" ")
                
                if elements[0] == "-LRB-" or elements[0] == "-LCB-":
                    elements[0] = "("
                elif elements[0] == "-RRB-" or elements[0] == "-RCB-":
                    elements[0] = ")"
                
                word = elements[0]
                
                if option == "all":
                    if universal:
                        pos = convert_ptp_to_universal(elements[1])
                        chunk = convert_ptp_to_universal(elements[2])
                        tag = (pos, chunk) 
                    else:
                        tag = elements[index_option]
                        
                    sent.append((pos, chunk, word))
                
                else:
                    if universal:
                        tag = convert_ptp_to_universal(elements[index_option])
                    else:
                        tag = elements[index_option]
                    
                    sent.append((tag, word))
                    
            else:
                tagged_sentences.append(sent)
                sent = []
    
    return tagged_sentences 
        
def load_conll2000(path = "./", universal = True, option = "pos", proportion_train = 1):
    train_set = load_train_conll2000(path, universal, option)
    train_set = np.random.choice(train_set, size = int(proportion_train * len(train_set)), replace = False).tolist()
    test_set = load_test_conll2000(path, universal, option)
    
    return train_set, test_set