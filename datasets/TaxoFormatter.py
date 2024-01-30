class TaxoFormatter:
    @staticmethod
    def taxo_relations_format(relations):
        return '\n'.join(list(map(lambda x: '{}\t{}'.format(x[0], x[1]), relations)))

    @staticmethod
    def terms_format(terms):
        get_simple_name = lambda r: r.split('.')[0]
        return '\n'.join(
            list(
                map(
                    lambda x: '{}\t{}||{}'.format(x, get_simple_name(x), x),
                    terms
                )
            )
        )

    @staticmethod
    def embed_format(term_and_embedding, dim):
        header = '{} {}'.format(len(term_and_embedding), dim)
        body = '\n'.join(
            list(
                map(
                    lambda x: '{} {}'.format(x[0], ' '.join(list(map(str, x[1])))),
                    term_and_embedding
                )
            )
        )
        return '\n'.join([header, body])


class TaxoInferFormatter:
    @staticmethod
    def terms_infer_format(terms_and_embeddings):
        res = []
        for term_and_embedding in terms_and_embeddings:
            res.append('{}\t{}'.format(term_and_embedding[0], ' '.join(list(map(str, term_and_embedding[1])))))
        return '\n'.join(res)
