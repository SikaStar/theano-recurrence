from random import randint
import numpy as np
import matplotlib.pyplot as plt

import cPickle as pkl
import timeit

import theano
import theano.tensor as T

from model.rnn import Rnn, BiRnn
from model.gru import Gru, BiGru
from model.lstm import Lstm, BiLstm

from utilities.optimizers import get_optimizer
from utilities.loaddata import load_data

__author__ = 'uyaseen'


def train_rnn(dataset, vocabulary, b_path, use_existing_model=False,
              optimizer='rmsprop', learning_rate=0.001, n_epochs=100,
              sample_length=200):
    print('train_rnn(..)')
    vocab, ix_to_words, words_to_ix = vocabulary
    vocab_enc = [words_to_ix[wd] for wd in vocab]
    train_set_x, train_set_y, voc = load_data(dataset, vocab, vocab_enc)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    index = T.lscalar('index')
    x = T.fmatrix('x')
    y = T.fmatrix('y')
    print('... building the model')
    n_x = len(vocab)  # dimension of embedding space, should be len(vocab) for one-hot-vector
    n_h = 100
    n_y = len(vocab)  # dimension of output classes
    m_path = b_path + 'rnn-best_model.pkl'

    rnn_params = None
    if use_existing_model:
        f = open(m_path, 'rb')
        rnn_params = pkl.load(f)

    rnn = Rnn(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
              params=rnn_params)
    cost = rnn.cross_entropy(y)
    updates = get_optimizer(optimizer, cost, rnn.params, learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index],
            y: train_set_y[index]
        },
        updates=updates
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    n_train_examples = train_set_x.get_value(borrow=True).shape[0]
    logging_freq = n_train_examples / 10
    sampling_freq = 10  # sampling is computationally expensive, therefore, need to be adjusted
    epoch = 0
    epochs = []  # for plotting stuff
    costs = []
    best_train_error = np.inf
    start_time = timeit.default_timer()
    done_looping = False
    while(epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_cost = 0.
        for i in xrange(n_train_examples):
            iter_start_time = timeit.default_timer()
            train_cost += train_model(i)

            # save the current best model
            if train_cost < best_train_error:
                best_train_error = train_cost
                f = open(m_path, 'w')
                pkl.dump(rnn.params, f, pkl.HIGHEST_PROTOCOL)

            if i % logging_freq == 0:
                iter_end_time = timeit.default_timer()
                print('epoch: %i/%i, sample: %i/%i, cost: %0.8f, /sample: %.4fm' %
                      (epoch, n_epochs, i, n_train_examples, train_cost/(i+1),
                       (iter_end_time - iter_start_time) / 60.))

        # sample from the model now and then
        if epoch % sampling_freq == 0:
            seed = randint(0, len(vocab)-1)
            idxes = rnn.generative_sampling(seed, emb_data=voc, sample_length=sample_length)
            sample = ''.join(ix_to_words[ix] for ix in idxes)
            print(sample)

        train_cost /= n_train_examples
        epochs.append(epoch)
        costs.append(train_cost)
    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    plt.title('RNN')
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy error')
    plt.plot(epochs, costs, color='red')
    plt.savefig(b_path + 'rnn-error-plot.png')
    plt.show()
    plt.close()


def train_gru(dataset, vocabulary, b_path, use_existing_model=False,
              optimizer='rmsprop', learning_rate=0.001, n_epochs=100,
              sample_length=200):
    print('train_gru(..)')
    vocab, ix_to_words, words_to_ix = vocabulary
    vocab_enc = [words_to_ix[wd] for wd in vocab]
    train_set_x, train_set_y, voc = load_data(dataset, vocab, vocab_enc)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    index = T.lscalar('index')
    x = T.fmatrix('x')
    y = T.fmatrix('y')
    print('... building the model')
    n_x = len(vocab)  # dimension of embedding space, should be len(vocab) for one-hot-vector
    n_h = 100
    n_y = len(vocab)  # dimension of output classes
    m_path = b_path + 'gru-best_model.pkl'

    gru_params = None
    if use_existing_model:
        f = open(m_path, 'rb')
        gru_params = pkl.load(f)

    gru = Gru(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
              params=gru_params)
    cost = gru.cross_entropy(y)
    updates = get_optimizer(optimizer, cost, gru.params, learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index],
            y: train_set_y[index]
        },
        updates=updates
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    n_train_examples = train_set_x.get_value(borrow=True).shape[0]
    logging_freq = n_train_examples / 10
    sampling_freq = 10  # sampling is computationally expensive, therefore, need to be adjusted
    epoch = 0
    epochs = []  # for plotting stuff
    costs = []
    best_train_error = np.inf
    start_time = timeit.default_timer()
    done_looping = False
    while(epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_cost = 0.
        for i in xrange(n_train_examples):
            iter_start_time = timeit.default_timer()
            train_cost += train_model(i)

            # save the current best model
            if train_cost < best_train_error:
                best_train_error = train_cost
                f = open(m_path, 'w')
                pkl.dump(gru.params, f, pkl.HIGHEST_PROTOCOL)

            if i % logging_freq == 0:
                iter_end_time = timeit.default_timer()
                print('epoch: %i/%i, sample: %i/%i, cost: %0.8f, /sample: %.4fm' %
                      (epoch, n_epochs, i, n_train_examples, train_cost/(i+1),
                       (iter_end_time - iter_start_time) / 60.))

        # sample from the model now and then
        if epoch % sampling_freq == 0:
            seed = randint(0, len(vocab)-1)
            idxes = gru.generative_sampling(seed, emb_data=voc, sample_length=sample_length)
            sample = ''.join(ix_to_words[ix] for ix in idxes)
            print(sample)

        train_cost /= n_train_examples
        epochs.append(epoch)
        costs.append(train_cost)
    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    plt.title('GRU')
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy error')
    plt.plot(epochs, costs, color='red')
    plt.savefig(b_path + 'gru-error-plot.png')
    plt.show()
    plt.close()


def train_lstm(dataset, vocabulary, b_path, use_existing_model=False,
               optimizer='rmsprop', learning_rate=0.001, n_epochs=100,
               sample_length=200):
    print('train_lstm(..)')
    vocab, ix_to_words, words_to_ix = vocabulary
    vocab_enc = [words_to_ix[wd] for wd in vocab]
    train_set_x, train_set_y, voc = load_data(dataset, vocab, vocab_enc)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    index = T.lscalar('index')
    x = T.fmatrix('x')
    y = T.fmatrix('y')
    print('... building the model')
    n_x = len(vocab)  # dimension of embedding space, should be len(vocab) for one-hot-vector
    n_h = 100
    n_y = len(vocab)  # dimension of output classes
    m_path = b_path + 'lstm-best_model.pkl'

    lstm_params = None
    if use_existing_model:
        f = open(m_path, 'rb')
        lstm_params = pkl.load(f)

    lstm = Lstm(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                params=lstm_params)
    cost = lstm.cross_entropy(y)
    updates = get_optimizer(optimizer, cost, lstm.params, learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index],
            y: train_set_y[index]
        },
        updates=updates
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    n_train_examples = train_set_x.get_value(borrow=True).shape[0]
    logging_freq = n_train_examples / 10
    sampling_freq = 10  # sampling is computationally expensive, therefore, need to be adjusted
    epoch = 0
    epochs = []  # for plotting stuff
    costs = []
    best_train_error = np.inf
    start_time = timeit.default_timer()
    done_looping = False
    while(epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_cost = 0.
        for i in xrange(n_train_examples):
            iter_start_time = timeit.default_timer()
            train_cost += train_model(i)

            # save the current best model
            if train_cost < best_train_error:
                best_train_error = train_cost
                f = open(m_path, 'w')
                pkl.dump(lstm.params, f, pkl.HIGHEST_PROTOCOL)

            if i % logging_freq == 0:
                iter_end_time = timeit.default_timer()
                print('epoch: %i/%i, sample: %i/%i, cost: %0.8f, /sample: %.4fm' %
                      (epoch, n_epochs, i, n_train_examples, train_cost/(i+1),
                       (iter_end_time - iter_start_time) / 60.))

        # sample from the model now and then
        if epoch % sampling_freq == 0:
            seed = randint(0, len(vocab)-1)
            idxes = lstm.generative_sampling(seed, emb_data=voc, sample_length=sample_length)
            sample = ''.join(ix_to_words[ix] for ix in idxes)
            print(sample)

        train_cost /= n_train_examples
        epochs.append(epoch)
        costs.append(train_cost)
    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    plt.title('LSTM')
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy error')
    plt.plot(epochs, costs, color='red')
    plt.savefig(b_path + 'lstm-error-plot.png')
    plt.show()
    plt.close()


def train_birnn(dataset, vocabulary, b_path, use_existing_model=False,
                optimizer='rmsprop', learning_rate=0.001, n_epochs=100,
                sample_length=70):
    print('train_birnn(..)')
    vocab, ix_to_words, words_to_ix = vocabulary
    vocab_enc = [words_to_ix[wd] for wd in vocab]
    train_set_x, train_set_y, voc = load_data(dataset, vocab, vocab_enc)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    index = T.lscalar('index')
    x = T.fmatrix('x')
    y = T.fmatrix('y')
    print('... building the model')
    n_x = len(vocab)  # dimension of embedding space, should be len(vocab) for one-hot-vector
    n_h = 100 / 2
    n_y = len(vocab)  # dimension of output classes
    m_path = b_path + 'birnn-best_model.pkl'

    birnn_params = None
    if use_existing_model:
        raise NotImplementedError

    birnn = BiRnn(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                  params=birnn_params)
    cost = birnn.cross_entropy(y)
    updates = get_optimizer(optimizer, cost, birnn.params, learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index],
            y: train_set_y[index]
        },
        updates=updates
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    n_train_examples = train_set_x.get_value(borrow=True).shape[0]
    sampling_freq = n_train_examples / 10
    epoch = 0
    epochs = []  # for plotting stuff
    costs = []
    best_train_error = np.inf
    start_time = timeit.default_timer()
    done_looping = False
    while(epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_cost = 0.
        for i in xrange(n_train_examples):
            iter_start_time = timeit.default_timer()
            train_cost += train_model(i)

            # save the current best model {not useful; see definition of BiRnn}
            if train_cost < best_train_error:
                best_train_error = train_cost
                f = open(m_path, 'w')
                pkl.dump(birnn.params, f, pkl.HIGHEST_PROTOCOL)

            # sample from the model now and then
            if i % sampling_freq == 0:
                iter_end_time = timeit.default_timer()
                print('epoch: %i/%i, sample: %i/%i, cost: %0.8f, /sample: %.4fm' %
                      (epoch, n_epochs, i, n_train_examples, train_cost/(i+1),
                       (iter_end_time - iter_start_time) / 60.))

        train_cost /= n_train_examples
        epochs.append(epoch)
        costs.append(train_cost)
    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    plt.title('Bi-RNN')
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy error')
    plt.plot(epochs, costs, color='red')
    plt.savefig(b_path + 'birnn-error-plot.png')
    plt.show()
    plt.close()


def train_bigru(dataset, vocabulary, b_path, use_existing_model=False,
                optimizer='rmsprop', learning_rate=0.001, n_epochs=100,
                sample_length=70):
    print('train_bigru(..)')
    vocab, ix_to_words, words_to_ix = vocabulary
    vocab_enc = [words_to_ix[wd] for wd in vocab]
    train_set_x, train_set_y, voc = load_data(dataset, vocab, vocab_enc)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    index = T.lscalar('index')
    x = T.fmatrix('x')
    y = T.fmatrix('y')
    print('... building the model')
    n_x = len(vocab)  # dimension of embedding space, should be len(vocab) for one-hot-vector
    n_h = 100 / 2
    n_y = len(vocab)  # dimension of output classes
    m_path = b_path + 'bigru-best_model.pkl'

    bigru_params = None
    if use_existing_model:
        raise NotImplementedError

    bigru = BiGru(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                  params=bigru_params)
    cost = bigru.cross_entropy(y)
    updates = get_optimizer(optimizer, cost, bigru.params, learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index],
            y: train_set_y[index]
        },
        updates=updates
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    n_train_examples = train_set_x.get_value(borrow=True).shape[0]
    sampling_freq = n_train_examples / 10
    epoch = 0
    epochs = []  # for plotting stuff
    costs = []
    best_train_error = np.inf
    start_time = timeit.default_timer()
    done_looping = False
    while(epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_cost = 0.
        for i in xrange(n_train_examples):
            iter_start_time = timeit.default_timer()
            train_cost += train_model(i)

            # save the current best model {not useful; see definition of BiGru}
            if train_cost < best_train_error:
                best_train_error = train_cost
                f = open(m_path, 'w')
                pkl.dump(bigru.params, f, pkl.HIGHEST_PROTOCOL)

            # sample from the model now and then
            if i % sampling_freq == 0:
                iter_end_time = timeit.default_timer()
                print('epoch: %i/%i, sample: %i/%i, cost: %0.8f, /sample: %.4fm' %
                      (epoch, n_epochs, i, n_train_examples, train_cost/(i+1),
                       (iter_end_time - iter_start_time) / 60.))

        train_cost /= n_train_examples
        epochs.append(epoch)
        costs.append(train_cost)
    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    plt.title('Bi-GRU')
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy error')
    plt.plot(epochs, costs, color='red')
    plt.savefig(b_path + 'bigru-error-plot.png')
    plt.show()
    plt.close()


def train_bilstm(dataset, vocabulary, b_path, use_existing_model=False,
                 optimizer='rmsprop', learning_rate=0.001, n_epochs=100,
                 sample_length=70):
    print('train_bilstm(..)')
    vocab, ix_to_words, words_to_ix = vocabulary
    vocab_enc = [words_to_ix[wd] for wd in vocab]
    train_set_x, train_set_y, voc = load_data(dataset, vocab, vocab_enc)

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    index = T.lscalar('index')
    x = T.fmatrix('x')
    y = T.fmatrix('y')
    print('... building the model')
    n_x = len(vocab)  # dimension of embedding space, should be len(vocab) for one-hot-vector
    n_h = 100 / 2
    n_y = len(vocab)  # dimension of output classes
    m_path = b_path + 'bilstm-best_model.pkl'

    bilstm_params = None
    if use_existing_model:
        raise NotImplementedError

    bilstm = BiLstm(input=x, input_dim=n_x, hidden_dim=n_h, output_dim=n_y,
                    params=bilstm_params)
    cost = bilstm.cross_entropy(y)
    updates = get_optimizer(optimizer, cost, bilstm.params, learning_rate)
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        givens={
            x: train_set_x[index],
            y: train_set_y[index]
        },
        updates=updates
    )
    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    n_train_examples = train_set_x.get_value(borrow=True).shape[0]
    sampling_freq = n_train_examples / 10
    epoch = 0
    epochs = []  # for plotting stuff
    costs = []
    best_train_error = np.inf
    start_time = timeit.default_timer()
    done_looping = False
    while(epoch < n_epochs) and (not done_looping):
        epoch += 1
        train_cost = 0.
        for i in xrange(n_train_examples):
            iter_start_time = timeit.default_timer()
            train_cost += train_model(i)

            # save the current best model {not useful; see definition of BiLstm}
            if train_cost < best_train_error:
                best_train_error = train_cost
                f = open(m_path, 'w')
                pkl.dump(bilstm.params, f, pkl.HIGHEST_PROTOCOL)

            # sample from the model now and then
            if i % sampling_freq == 0:
                iter_end_time = timeit.default_timer()
                print('epoch: %i/%i, sample: %i/%i, cost: %0.8f, /sample: %.4fm' %
                      (epoch, n_epochs, i, n_train_examples, train_cost/(i+1),
                       (iter_end_time - iter_start_time) / 60.))

        train_cost /= n_train_examples
        epochs.append(epoch)
        costs.append(train_cost)
    end_time = timeit.default_timer()
    print('The code ran for %.2fm' % ((end_time - start_time) / 60.))
    plt.title('Bi-LSTM')
    plt.xlabel('epochs')
    plt.ylabel('cross-entropy error')
    plt.plot(epochs, costs, color='red')
    plt.savefig(b_path + 'bilstm-error-plot.png')
    plt.show()
    plt.close()

