from utilities.textreader import read_word_data, read_char_data
from train import train

__author__ = 'uyaseen'


if __name__ == '__main__':
    data, vocabulary = read_char_data('data/input.txt', seq_length=50)
    train(data, vocabulary, b_path='data/models/', rec_model='gru',
          n_h=100, use_existing_model=True, n_epochs=600)
    print('... done')
