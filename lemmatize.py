from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
lem_cache = {}

def lem(word):
    if word not in lem_cache:
        lem_cache[word] = \
            wnl.lemmatize(wnl.lemmatize(wnl.lemmatize(wnl.lemmatize(word), \
            'v'), 'a'), 'r')
    return lem_cache[word]
