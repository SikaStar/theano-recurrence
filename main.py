from utilities.textreader import read_word_data, read_char_data
from train import train_rnn, train_gru, train_lstm, train_birnn, train_bigru, train_bilstm

__author__ = 'uyaseen'


if __name__ == '__main__':
    data, vocabulary = read_char_data('data/input.txt', seq_length=50)
    train_gru(data, vocabulary, b_path='data/models/', use_existing_model=True,
              n_epochs=600)
    print('... done')
