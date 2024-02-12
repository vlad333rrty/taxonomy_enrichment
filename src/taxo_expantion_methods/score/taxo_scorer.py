from nltk.corpus import WordNetCorpusReader
from nltk.corpus.reader import Synset

import src.taxo_expantion_methods.common.performance
from src.taxo_expantion_methods.common.wn_dao import WordNetDao


class TaxoScorer:
    def __init__(self, wn_reader: WordNetCorpusReader):
        self.__wn_reader = wn_reader
        self.__all_synsets = set( # todo bad idea
            map(
                lambda x: x.name(),
                self.__wn_reader.all_synsets('n')
            )
        )

    def __wup(self, s1: Synset, s2: Synset):
        need_root = s1.pos() == self.__wn_reader.VERB
        subsumers = s1.lowest_common_hypernyms(s2, need_root, use_min_depth=True)

        subsumer = subsumers[0]

        # Get the longest path from the LCS to the root,
        # including a correction:
        # - add one because the calculations include both the start and end
        #   nodes
        depth = subsumer.max_depth() + 1

        # Get the shortest path from the LCS to each of the synsets it is
        # subsuming.  Add this to the LCS path length to get the path
        # length from each synset to the root.
        len1 = s1.shortest_path_distance(subsumer, need_root)
        len2 = s2.shortest_path_distance(subsumer, need_root)

        # When the system's answer file differs in the operation (e.g., says attach
        # instead of merge), then the height correction is 1, indicating the
        # effective-depth of the synset differs from its actual location.
        len1 += depth
        len2 += depth

        return (2.0 * depth) / (len1 + len2)

    @staticmethod
    def __parse_input(path, sep='\t'):
        with open(path, 'r') as file:
            terms_and_parents = list(map(lambda x: x.split(sep), file.readlines()))
        term2parent = {}
        for pair in terms_and_parents:
            term = pair[0]
            parents = pair[1].split(',')
            if term in term2parent:
                term2parent[term] += parents
            else:
                term2parent[term] = parents
        return term2parent

    @staticmethod
    def __does_lemma_match(gold_wn_synset, system_wn_synset):
        gold_lemmas = set(str(lemma.name()) for lemma in gold_wn_synset.lemmas())
        system_lemmas = set(str(lemma.name()) for lemma in system_wn_synset.lemmas())

        for lemma in gold_lemmas:
            if lemma in system_lemmas:
                return True
        return False

    def __get_best_wup(self, potential_parents, predicted_parents):
        wup_max = 0
        for e in potential_parents:
            e_s = self.__wn_reader.synset(e)
            for p in predicted_parents:
                if p not in self.__all_synsets:
                    print('No synset in current wordnet for', p)
                    continue
                p_s = self.__wn_reader.synset(p)
                wup_cur = self.__wup(e_s, p_s)
                if wup_cur > wup_max:
                    wup_max = wup_cur
        return wup_max

    def score_taxo_results(self, golden_path, predicted_path):
        """
        format: new_term \t parent
        :param golden_path:
        :param predicted_path:
        :return: (recall, wup)
        """
        etalon_term2parent = TaxoScorer.__parse_input(golden_path)
        predicted_term2parent = TaxoScorer.__parse_input(predicted_path)
        coverage = 0
        wup = 0
        accuracy = 0
        prec_i = 0
        for term in etalon_term2parent:
            print('Processing term', term)
            if term not in predicted_term2parent:
                print('Missing predicted input for synset', term)
                continue
            coverage += 1
            expected = etalon_term2parent[term]
            predicted = predicted_term2parent[term]
            wup += self.__get_best_wup(expected, predicted)
            acc = 0
            p_set = set(predicted)
            e_set = set(expected)
            true_answs = p_set.intersection(e_set)
            accuracy += len(true_answs)
            prec_i += len(true_answs) / len(p_set)

        return coverage / len(etalon_term2parent), wup / coverage, accuracy / coverage, prec_i / len(predicted_term2parent)


def run_scorer():
    wn_reader = WordNetDao.get_wn_30()
    scorer = TaxoScorer(wn_reader)
    results = scorer.score_taxo_results('data/results/taxo_expan/golden.tsv', 'data/results/taxo_expan/predicted.tsv')
    print(results)


run_scorer()
