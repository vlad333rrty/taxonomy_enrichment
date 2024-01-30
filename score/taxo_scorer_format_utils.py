import re

SYNSET_FORMAT_PATTERN = '[A-Za-z0-9._\'/-]'

def __format_and_dump(term2parents, path):
    res = []
    for x in term2parents:
        res.append('{}\t{}'.format(x[0], ','.join(x[1])))
    with open(path, 'w') as file:
        file.write('\n'.join(res))


def __parse_parent_taxo_result(parent_raw):
    match = re.match('(.+)?(\|\|)({}+)@@@\\d+'.format(SYNSET_FORMAT_PATTERN), parent_raw.strip())
    if match is None:
        raise 'Match is None'
    parent_synset = match.group(3)
    return parent_synset


def format_taxo_result(path, dump_path):
    with open(path, 'r') as file:
        lines = file.readlines()[1:]
    term2parents = []
    for line in lines:
        match = re.match('(.+)?\t(.+)', line)
        if match is None:
            raise 'Match is None'
        term = match.group(1)
        parents_raw = match.group(2).split(',')
        parents_synstes = []
        for parent_raw in parents_raw:
            parents_synstes.append(__parse_parent_taxo_result(parent_raw))
        term2parents.append((term, parents_synstes))

    __format_and_dump(term2parents, dump_path)


def __parse_parent_diachronic_ds(parent_raw):
    match = re.match('"({}+)"'.format(SYNSET_FORMAT_PATTERN), parent_raw.strip())
    if match is None:
        raise 'Match is None'
    return match.group(1)

def format_diachronic_ds(path, dump_path):
    with open(path, 'r') as file:
        lines = file.readlines()
    term2parents = []
    for line in lines:
        match = re.match('(.+)?\t\[(.+)\]', line)
        if match is None:
            raise "Match is None"
        term = match.group(1)
        parents_raw = match.group(2).split(',')
        parents = []
        for parent_raw in parents_raw:
            parents.append(__parse_parent_diachronic_ds(parent_raw))
        term2parents.append((term, parents))
    __format_and_dump(term2parents, dump_path)


def run_formatting():
    format_diachronic_ds('data/datasets/diachronic-wordnets/en/nouns_en.2.0-3.0.tsv', 'score/golden.tsv')
    format_taxo_result('score/taxo_result_full_learn.tsv', 'score/predicted_on_full_learn.tsv')

run_formatting()