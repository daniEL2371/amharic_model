python .\text_preprocessor.py -v 1 -c 'data/big.txt' -C 'data/charset.txt' -f 'data/data_100_10000.h5' -b 100 -s 100 -n 10000

python .\train_generator.py -b 100 -e 50 -s 100 -i 'data/data_100_10000.h5' -t batched

