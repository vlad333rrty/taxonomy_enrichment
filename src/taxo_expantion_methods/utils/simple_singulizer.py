mapping = {
    'features': 'feature',
    'orders': 'order',
    'glaciers': 'glacier',
    'languages': 'language'
}


# todo выглядид как костыль нужно удалить
def get_singular(word):
    if word in mapping:
        return mapping[word]
    return word