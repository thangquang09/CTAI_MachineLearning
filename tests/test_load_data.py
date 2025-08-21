def test_calculate_cbow_features():
    from src.load_data import calculate_cbow_features
    assert calculate_cbow_features() is not None

def test_import_word2vec():
    try:
        from gensim.models import Word2Vec
        assert True
    except ImportError:
        assert False