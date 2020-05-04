def convert_ptp_to_universal(ptp_tag):
    struct = {
        "NN|SYM": "NOUN",
        "#": "SYM",
        "$": "SYM",
        "''": "PUNCT",
        "(": "PUNCT",
        ")": "PUNCT",
        ",": "PUNCT",
        ".": "PUNCT",
        ":": "PUNCT",
        "``": "PUNCT",
        '"': "PUNCT",
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
        
         "B-NP": "NP",
        "B-PP": "PP",
        "O": "O",
        "I-NP": "NP",
        "I-PP": "PP",
        "B-VP": "VP",
        "I-VP": "VP",
        "B-SBAR": "SBAR",
        "B-ADJP": "ADJP",
        "I-ADJP": "ADJP",
        "B-ADVP": "ADVP",
        "I-ADVP": "ADVP",
        "B-INTJ": "INTJ",
        "I-PRT": "PRT",
        "I-SBAR": "SBAR",
        "I-UCP": "UCP",
        "B-PRT": "PRT",
        "B-LST": "LST",
        "I-CONJP": "CONJP",
        "B-CONJP": "CONJP",
        "I-INTJ": "INTJ",
        "B-UCP": "UCP",
        "I-LST": "LST"
    }
    
    if ptp_tag in struct.keys():
        return struct[ptp_tag]
    else:
        return ptp_tag

def entity_to_entity(entity):
    if entity[-3:] == "PER":
        return "PER"

    elif entity[-3:] == "LOC":
        return "LOC"

    elif entity[-3:] == "ORG":
        return "ORG"

    elif entity[-4:] == "MISC":
        return "MISC"

    else:
        return "O"


def load_conll2003(path = "./", options = []):
    train_set = load_file(path, "train", options) + load_file(path, "valid", options)
    test_set = load_file(path, "test", options)
    
    return train_set, test_set

def load_file(path = "./", filename = "train",  options = "ner"):
    
    dataset = []
    with open(path + filename + ".txt") as file:
        
        X1, X2, X3 = [], [], []
        Y = []
        for line in file:
            
            if line != "\n":
                
                splited_line = line.split(" ")
                Y.append(splited_line[0])
                X1.append(splited_line[1])
                X2.append(splited_line[2])
                X3.append(splited_line[3][:-1])
                
            else:
                
                Z = []
                for y, x1, x2, x3 in zip(Y, X1, X2, X3):

                    x = []

                    if "ner" in options:
                        x = entity_to_entity(x3)

                    if "chunk" in options:
                        x = convert_ptp_to_universal(x2)

                    if "pos" in options:
                        x = convert_ptp_to_universal(x1)
                        
                    if options == "all":
                        x = (entity_to_entity(x3), convert_ptp_to_universal(x2), convert_ptp_to_universal(x1))
                
                    if x1 != "-X-":
                        Z.append((x, y))
                
                if len(Z) != 0:        
                    dataset.append(Z)
                X1, X2, X3 = [], [], []
                Y = []
                
    return dataset
