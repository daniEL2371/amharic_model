from text_preprocessor import TextPreProcessor

tp = TextPreProcessor('data/charset.txt', 100, 100)
tp.text_to_bin_v3('data/big.txt', "data/data_100.h5", 10000)
