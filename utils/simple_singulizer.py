mapping = {
    'features': 'feature',
    'orders': 'order',
    'glaciers': 'glacier',
    'languages': 'language'
}

def get_singular(word):
    if word in mapping:
        return mapping[word]
    return word