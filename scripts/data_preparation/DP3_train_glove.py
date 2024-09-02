import subprocess
import os

# Define vector sizes to iterate over
vector_sizes = ['300', '500', '1000']

# Define other parameters
corpus = 'data/clean/corpus.txt'
vocab_file_base_path = 'data/models/glove/vocab.txt'
cooccur_file_base_path = 'data/models/glove/{vector_size}/cooccurence_{vector_size}.bin'
cooccur_shuf_file_base_path = 'data/models/glove/{vector_size}/cooccurence_{vector_size}.shuf.bin'
save_file_base_path = 'data/models/glove/{vector_size}/glove_{vector_size}'
verbose = '2'
memory = '4.0'
vocab_min_count = '5'
max_iter = '50'
window_size = '10'
binary = '2'
num_threads = '8'
x_max = '10'

# Define paths to GloVe executable
build_dir = 'venv/lib/glove/build'

def train_glove(
    vector_size,
    corpus,
    vocab_file_path,
    cooccur_file,
    cooccur_shuf_file, 
    save_file,
    verbose='2',
    memory='4.0',
    vocab_min_count='5',
    max_iter='15',
    window_size='15',
    binary='2',
    num_threads='8',
    x_max='10'):

    # Run GloVe commands
    subprocess.run([
        f'{build_dir}/vocab_count',
        '-min-count', vocab_min_count,
        '-verbose', verbose
    ], stdin=open(corpus, 'r'), stdout=open(vocab_file_path, 'w'))
    print(f'Vocabulary file created for vector size {vector_size}')
    
    subprocess.run([
        f'{build_dir}/cooccur',
        '-memory', memory,
        '-vocab-file', vocab_file_path,
        '-verbose', verbose,
        '-window-size', window_size
    ], stdin=open(corpus, 'r'), stdout=open(cooccur_file, 'w'))
    print(f'Cooccurrence file created for vector size {vector_size}')
    
    subprocess.run([
        f'{build_dir}/shuffle',
        '-memory', memory,
        '-verbose', verbose
    ], stdin=open(cooccur_file, 'r'), stdout=open(cooccur_shuf_file, 'w'))
    print(f'Shuffled cooccurrence file created for vector size {vector_size}')
    
    subprocess.run([
        f'{build_dir}/glove',
        '-save-file', save_file,
        '-threads', num_threads,
        '-input-file', cooccur_shuf_file,
        '-x-max', x_max,
        '-iter', max_iter,
        '-vector-size', vector_size,
        '-binary', binary,
        '-vocab-file', vocab_file_path,
        '-verbose', verbose
    ])
    print(f'GloVe training complete for vector size {vector_size}')

# Loop over vector sizes
for vector_size in vector_sizes:
    # Define paths for the current vector size
    print(f'\n\nTraining GloVe for vector size {vector_size}\n')
    vocab_file_path = vocab_file_base_path.format(vector_size=vector_size)
    cooccur_file = cooccur_file_base_path.format(vector_size=vector_size)
    cooccur_shuf_file = cooccur_shuf_file_base_path.format(vector_size=vector_size)
    save_file = save_file_base_path.format(vector_size=vector_size)
    
    # Create necessary directories if they do not exist
    os.makedirs(os.path.dirname(cooccur_file), exist_ok=True)
    
    # Run GloVe training for the current vector size
    train_glove(
        vector_size=vector_size,
        corpus=corpus,
        vocab_file_path=vocab_file_path,
        cooccur_file=cooccur_file,
        cooccur_shuf_file=cooccur_shuf_file,
        save_file=save_file,
        verbose=verbose,
        memory=memory,
        vocab_min_count=vocab_min_count,
        max_iter=max_iter,
        window_size=window_size,
        binary=binary,
        num_threads=num_threads,
        x_max=x_max
    )
    print(f'\nGloVe training complete for vector size {vector_size}\n\n')