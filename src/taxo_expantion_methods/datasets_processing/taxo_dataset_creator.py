import time
from pathlib import Path

from gensim.models.fasttext import load_facebook_model
from nltk.corpus import WordNetCorpusReader
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

import src.taxo_expantion_methods.common.performance
from src.taxo_expantion_methods.common import performance
from src.taxo_expantion_methods.common.wn_dao import WordNetDao
from src.taxo_expantion_methods.datasets_processing.FasttextVectorizer import FasttextVectorizer, BertVectorizer
from src.taxo_expantion_methods.datasets_processing.TaxoFormatter import TaxoFormatter, TaxoInferFormatter
from src.taxo_expantion_methods.datasets_processing.wordnet_parser import WordnetParser
from src.taxo_expantion_methods.parent_sum.node_embeddings_provider import EmbeddingsGraphBERTNodeEmbeddingsProvider


def __write_to_file(path, data):
    with open(path, 'w') as file:
        file.write(data)

def __get_simple_name(synset_raw):
    return synset_raw.split('.')[0]


def prepare_wordnet_for_training(wn_path: str, train_ratio: float, term_vectorizer, prefix):
    start = time.time()
    delta, wn_redaer = src.taxo_expantion_methods.common.performance.measure(lambda: WordNetCorpusReader(wn_path, None))
    print('Wordnet reader created in', delta, 'seconds')

    wordnet_parser = WordnetParser(wn_redaer)
    delta, terms_and_relations = src.taxo_expantion_methods.common.performance.measure(lambda: wordnet_parser.traverse_nouns())
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


def prepare_terms_for_inference(ds_path, prefix):
    with open(ds_path, 'r') as file:
        words = list(map(lambda x: x.strip(), file.readlines()))
    delta, model = src.taxo_expantion_methods.common.performance.measure(
        lambda: load_facebook_model('/home/vlad333rrty/Downloads/cc.en.300.bin.gz'))
    print('Model loaded in', delta, 'seconds')
    terms_and_embeddings = list(
        map(
            lambda term: (term, model.wv[term]),
            words
        )
    )
    infer_terms_formatted = TaxoInferFormatter.terms_infer_format(terms_and_embeddings)
    __write_to_file(prefix + 'wordnet_nouns.infer.terms', infer_terms_formatted)


def with_fasttext300_embeddings():
    delta, model = src.taxo_expantion_methods.common.performance.measure(
        lambda: load_facebook_model('/home/vlad333rrty/Downloads/cc.en.300.bin.gz'))
    print('Fasttext model loaded in', delta, 'seconds')
    vectorizer = FasttextVectorizer(model)
    prefix = 'data/datasets/taxo_expan/fasttext/'
    prepare_wordnet_for_training('data/wordnets/WordNet-3.0/dict', 0.6, vectorizer, prefix)

def with_bert_base_embeddings():
    device = 'cpu'
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
    embeddings_provider = EmbeddingsGraphBERTNodeEmbeddingsProvider(bert_model, tokenizer, device)
    bert_vectorizer = BertVectorizer(embeddings_provider, WordNetDao.get_wn_30())
    prefix = 'data/datasets/taxo_expan/bert/'
    prepare_wordnet_for_training('data/wordnets/WordNet-3.0/dict', 0.6, bert_vectorizer, prefix)

# prepare_terms_for_inference('data/datasets/semeval/no_labels_terms')
with_fasttext300_embeddings()