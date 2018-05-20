import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--version', dest='data_version', default=1, type=int,
                    help='1 for separate consonants and vowels, 2 for whole charset')
parser.add_argument('-c', '--corpus', dest='corpus_file',
                    help='the text corpus')
parser.add_argument('-C' '--charset', dest='charset_file',
                    help='character set file')
parser.add_argument('-f' '--file', dest='file',
                    help='the new file to bo created')
parser.add_argument('-b' '--batch', dest='batch_size',
                    type=int, default=128, help='batch size of saving the data')
parser.add_argument('-s' '--seq', dest='seq_length',
                    type=int, default=100, help='sequence length for RNN')
parser.add_argument('-n' '--n_batches', dest='n_batches',
                    type=int, default=-1, help='the max amount of the examples')
args = parser.parse_args()

from text_preprocessor import TextPreProcessor

tp = TextPreProcessor(args.charset_file,
                      args.batch_size, args.seq_length)
n_samples = args.n_batches * args.batch_size
v_file = args.file + "_v.h5"
c_file = args.file + "_c.h5"
print(args.file)
if args.data_version == 1:
    tp.text_to_bin(args.corpus_file, c_file, v_file, n_samples=n_samples)
else:
    tp.text_to_bin_v2(args.corpus_file, args.file, n_samples=n_samples)