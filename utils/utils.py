from nltk.corpus import wordnet31 as wn

from common.set_m import SetM


def paginate(entries, page_size):
    result = []
    k = len(entries) // page_size
    for j in range(k):
        page = []
        for r in range(page_size):
            index = j * page_size + r
            page.append(entries[index])
        result.append(page)
    page = []
    i = k * page_size
    while i < len(entries):
        page.append(entries[i])
        i += 1
    result.append(page)
    return result


def get_extended_synset_list(synsets):
    used_synsets = SetM()
    return list(
        filter(
            lambda s: len(s) > 0,
            map(
                lambda s: list(
                    filter(
                        lambda t: used_synsets.add(t.name()),
                        [elem for ss in [wn.synsets(l) for l in s.lemma_names()] for elem in ss]
                    )
                ),
                synsets
            )
        )
    )
