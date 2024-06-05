import random
import time
from pathlib import Path

from gensim.models.fasttext import load_facebook_model
from nltk.corpus import WordNetCorpusReader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

import src.taxo_expantion_methods.common.performance
from src.taxo_expantion_methods.TEMP.synsets_provider import SynsetsProvider
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.Term import Term
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.datasets_processing.FasttextVectorizer import FasttextVectorizer, BertVectorizer
from src.taxo_expantion_methods.datasets_processing.TaxoFormatter import TaxoFormatter, TaxoInferFormatter
from src.taxo_expantion_methods.datasets_processing.wordnet_parser import WordnetParser
from src.taxo_expantion_methods.datasets_processing.wordnet_subgraphs_provider import WordNetSubgraphProvider
from src.taxo_expantion_methods.engines.word_to_add_data import WordToAddDataParser
from src.taxo_expantion_methods.parent_sum.node_embeddings_provider import EmbeddingsGraphBERTNodeEmbeddingsProvider


def __write_to_file(path, data):
    with open(path, 'w') as file:
        file.write(data)

def __get_simple_name(synset_raw):
    return synset_raw.split('.')[0]


def prepare_wordnet_for_training(synsets, train_ratio: float, term_vectorizer, prefix):
    start = time.time()

    delta, terms_and_relations = performance.measure(lambda: WordnetParser.travers_synsets(synsets))
    print('Traversed wordnet in', delta, 'seconds')
    terms = list(terms_and_relations[0])
    relations = list(terms_and_relations[1])

    taxo_formatter = TaxoFormatter()
    delta, vectorization_result = performance.measure(lambda: term_vectorizer.vectorize_terms(terms))
    print('Vectorized terms in', delta, 'seconds')

    print('Splitting data in train/test with train ratio', train_ratio)
    terms_train, terms_test_val = train_test_split(terms, train_size=train_ratio)
    print('Got', len(terms_train), 'terms for training')
    terms_val, terms_test = train_test_split(terms, train_size=0.02, test_size=0.02)

    relations_formatted = taxo_formatter.taxo_relations_format(relations)
    terms_train_formatted = taxo_formatter.terms_format(terms_train)

    terms_test_formatted = taxo_formatter.terms_format(terms_test)
    terms_val_formatted = taxo_formatter.terms_format(terms_val)
    all_terms_formatted = taxo_formatter.terms_format(terms)
    term_and_embed_formatted = taxo_formatter.embed_format(vectorization_result.term_and_embed, vectorization_result.dimension)

    __write_to_file(prefix + 'wordnet_nouns.taxo', relations_formatted)
    __write_to_file(prefix + 'wordnet_nouns.terms.train', terms_train_formatted)
    __write_to_file(prefix + 'wordnet_nouns.terms.test', terms_test_formatted)
    __write_to_file(prefix + 'wordnet_nouns.terms.validation', terms_val_formatted)
    __write_to_file(prefix + 'wordnet_nouns.terms', all_terms_formatted)
    __write_to_file(prefix + 'wordnet_nouns.terms.fasttext.embed', term_and_embed_formatted)
    end = time.time()
    print('Finished in', end - start, 'seconds')


def prepare_terms_for_inference(ds_path, prefix, vectorizer):
    terms = list(
        map(
            lambda wta: Term(wta.value, wta.definition),
            WordToAddDataParser.from_pandas(ds_path, '$')
        )
    )

    terms = []
    wn = WordNetDao.get_wn_30()
    with open('data/datasets/temp_test.tsv', 'r') as file:
        lines = file.readlines()
        for line in lines:
            data = line.strip().split('\t')
            word = data[0]
            definition = wn.synset(word).definition()
            terms.append(Term(__get_simple_name(word), definition))

    vectorization_result = vectorizer.vectorize_terms(terms)

    infer_terms_formatted = TaxoInferFormatter.terms_infer_format(vectorization_result.term_and_embed)
    __write_to_file(prefix + 'wordnet_nouns.infer.terms', infer_terms_formatted)

def with_fasttext300_embeddings(embeddings_path):
    delta, model = performance.measure(
        lambda: load_facebook_model(embeddings_path))
    print('Fasttext model loaded in', delta, 'seconds')
    vectorizer = FasttextVectorizer(model)
    prefix = 'data/datasets/taxo_expan/fasttext/food'
    wn = WordNetDao.get_wn_30()
    all_synsets = SynsetsProvider.get_all_synsets_with_common_root(wn.synset('food.n.01'))
    prepare_wordnet_for_training(all_synsets, 0.8, vectorizer, prefix)

def with_bert_base_embeddings():
    device = 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    embeddings_provider = EmbeddingsGraphBERTNodeEmbeddingsProvider(bert_model, tokenizer, device)
    bert_vectorizer = BertVectorizer(embeddings_provider, WordNetDao.get_wn_30())
    prefix = 'data/datasets/taxo_expan/bert/'
    prepare_wordnet_for_training('data/wordnets/WordNet-3.0/dict', 0.6, bert_vectorizer, prefix)


def inference_with_fasttext300(embeddings_path, ds_path):
    delta, model = performance.measure(
        lambda: load_facebook_model(embeddings_path))
    print('Model loaded in', delta, 'seconds')
    vectorizer = FasttextVectorizer(model)
    prefix = 'data/datasets/taxo_expan/fasttext/food'
    prepare_terms_for_inference(ds_path, prefix, vectorizer)

inference_with_fasttext300('data/embeddings/fasttext/cc.en.300.bin.gz', 'data/datasets/semeval/training_data.csv')
# with_fasttext300_embeddings('data/embeddings/fasttext/cc.en.300.bin.gz')