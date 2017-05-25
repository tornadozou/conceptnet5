import click
from .formats import (
    convert_glove, convert_word2vec, convert_fasttext, convert_polyglot,
    load_hdf, save_hdf, export_text
)
from .retrofit import sharded_retrofit, join_shards
from .merge import merge_intersect
from .evaluation import wordsim, analogy, bias
from .evaluation.compare import (
    compare_embeddings, graph_comparison, graph_bias_comparison
)
from .miniaturize import miniaturize
from .debias import de_bias_frame
from .query import VectorSpaceWrapper


ANALOGY_FILENAME = 'data/raw/analogy/SAT-package-V3.txt'


@click.group()
def cli():
    pass


@cli.command(name='filter_word_vectors')
@click.argument('dense_hdf_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('vocab_filename', type=click.Path(readable=True, dir_okay=False))
def filter_word_vectors(dense_hdf_filename, vocab_filename):
    vsw = VectorSpaceWrapper(vector_filename=dense_hdf_filename)
    for line in open(vocab_filename):
        word = line.strip()
        term = '/c/en/' + word
        vec = vsw.get_vector(term)
        line_parts = [word] + ['%6.6f' % num for num in vec]
        print(' '.join(line_parts))


@cli.command(name='retrofit')
@click.argument('dense_hdf_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('conceptnet_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
@click.option('--iterations', '-i', default=5)
@click.option('--verbose', '-v', count=True)
@click.option('--nshards', '-s', default=6)
def run_retrofit(dense_hdf_filename, conceptnet_filename, output_filename,
                 iterations=5, nshards=6, verbose=1):
    sharded_retrofit(
        dense_hdf_filename, conceptnet_filename, output_filename,
        iterations=iterations, nshards=nshards, verbose=verbose
    )


@cli.command(name='join_retrofit')
@click.argument('filename', type=click.Path(writable=True, dir_okay=False))
@click.option('--nshards', '-s', default=6)
def run_join_retrofit(filename, nshards=6):
    join_shards(filename, nshards)


@cli.command(name='convert_glove')
@click.argument('glove_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
@click.option('--nrows', '-n', default=500000)
def run_convert_glove(glove_filename, output_filename, nrows=500000):
    convert_glove(glove_filename, output_filename, nrows)


@cli.command(name='convert_fasttext')
@click.argument('fasttext_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
@click.option('--nrows', '-n', default=500000)
@click.option('--language', '-l', default='en')
def run_convert_fasttext(fasttext_filename, output_filename, nrows=500000, language='en'):
    convert_fasttext(fasttext_filename, output_filename, nrows=nrows, language=language)


@cli.command(name='convert_word2vec')
@click.argument('word2vec_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
@click.option('--nrows', '-n', default=500000)
def run_convert_word2vec(word2vec_filename, output_filename, nrows=500000):
    convert_word2vec(word2vec_filename, output_filename, nrows)


@cli.command(name='convert_polyglot')
@click.argument('polyglot_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
@click.option('--language', '-l')
def run_convert_polyglot(polyglot_filename, output_filename, language):
    convert_polyglot(polyglot_filename, output_filename, language)


@cli.command(name='intersect')
@click.argument('input_filenames', nargs=-1, type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
@click.argument('projection_filename', type=click.Path(writable=True, dir_okay=False))
def run_intersect(input_filenames, output_filename, projection_filename):
    frames = [load_hdf(filename) for filename in input_filenames]
    intersected, projection = merge_intersect(frames)
    save_hdf(intersected, output_filename)
    save_hdf(projection, projection_filename)


@cli.command(name='debias')
@click.argument('input_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
def run_debias(input_filename, output_filename):
    frame = load_hdf(input_filename)
    debiased = de_bias_frame(frame)
    save_hdf(debiased, output_filename)


@cli.command(name='evaluate')
@click.argument('filename', type=click.Path(readable=True, dir_okay=False))
@click.option('--subset', '-s', type=click.Choice(['dev', 'test', 'all']), default='dev')
@click.option('--semeval-by-language/--semeval-global', '-l', default=False)
def run_evaluate(filename, subset, semeval_by_language):
    frame = load_hdf(filename)
    if semeval_by_language:
        scope = 'per-language'
    else:
        scope = 'global'
    print(wordsim.evaluate(frame, subset=subset, semeval_scope=scope))
    print(analogy.evaluate(frame, subset=subset, analogy_filename=ANALOGY_FILENAME))
    print(bias.measure_bias(frame))


@cli.command(name='evaluate_wordsim')
@click.argument('filename', type=click.Path(readable=True, dir_okay=False))
@click.option('--subset', '-s', type=click.Choice(['dev', 'test', 'all']), default='dev')
@click.option('--semeval-by-language/--semeval-global', '-l', default=False)
def run_evaluate_wordsim(filename, subset, semeval_by_language):
    frame = load_hdf(filename)
    if semeval_by_language:
        scope = 'per-language'
    else:
        scope = 'global'
    print(wordsim.evaluate(frame, subset=subset, semeval_scope=scope))


@cli.command(name='evaluate_raw')
@click.argument('filename', type=click.Path(readable=True, dir_okay=False))
@click.option('--subset', '-s', type=click.Choice(['dev', 'test', 'all']), default='dev')
@click.option('--semeval-by-language/--semeval-global', '-l', default=False)
def run_evaluate_raw(filename, subset, semeval_by_language):
    frame = load_hdf(filename)
    if semeval_by_language:
        scope = 'per-language'
    else:
        scope = 'global'
    print(wordsim.evaluate_raw(frame, subset=subset, semeval_scope=scope))


@cli.command(name='evaluate_analogies')
@click.argument('filename', type=click.Path(readable=True, dir_okay=False))
@click.option('--subset', '-s', type=click.Choice(['dev', 'test', 'all']), default='dev')
def run_evaluate_analogies(filename, subset):
    frame = load_hdf(filename)
    print(analogy.evaluate(frame, analogy_filename=ANALOGY_FILENAME))


@cli.command(name='evaluate_bias')
@click.argument('filename', type=click.Path(readable=True, dir_okay=False))
def run_evaluate_bias(filename):
    frame = load_hdf(filename)
    print(bias.measure_bias(frame))


@cli.command(name='compare_embeddings')
@click.argument('input_filenames', nargs=-1, type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
def run_compare_embeddings(input_filenames, output_filename):
    """
    The `input_filenames` are files that can be loaded as matrices of word
    embeddings. They'll be run through the relatedness and analogy evaluations,
    and the results will be saved in an HDF5 file, `output_filename`. This
    file can be used by `comparison_graph`.

    This requires the PostgreSQL database of ConceptNet 5 to be built, because
    it finds embeddings of uncommon words on the fly by looking up their
    neighbors in the ConceptNet graph. These embeddings could have been stored
    in the matrix, but this saves memory and download time.
    """
    results = compare_embeddings(input_filenames, subset='all', tune_analogies=True)
    print(results)
    save_hdf(results, output_filename)


@cli.command(name='comparison_graph')
@click.argument('table_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('eval_graph_filename', type=click.Path(writable=True, dir_okay=False))
def run_comparison_graph(table_filename, eval_graph_filename):
    """
    Convert a table of evaluation results into a PNG or PDF graph.
    """
    graph_comparison(table_filename, eval_graph_filename)


@cli.command(name='export_text')
@click.argument('input_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
@click.option('--language', '-l', default=None)
def run_export(input_filename, output_filename, language):
    frame = load_hdf(input_filename)
    export_text(frame, output_filename, language)


@cli.command(name='miniaturize')
@click.argument('input_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('extra_vocab_filename', type=click.Path(readable=True, dir_okay=False))
@click.argument('output_filename', type=click.Path(writable=True, dir_okay=False))
@click.option('-k', default=300, help="Number of columns to reduce to")
def run_miniaturize(input_filename, extra_vocab_filename, output_filename, k):
    frame = load_hdf(input_filename)
    other_frame = load_hdf(extra_vocab_filename)
    other_vocab = list(other_frame.index)
    del other_frame
    mini = miniaturize(frame, other_vocab=other_vocab, k=k)
    save_hdf(mini, output_filename)
