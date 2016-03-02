# theano-recurrence

This code implements **Recurrent Neural Network (RNN, GRU and LSTM) and their Bidirectional versions (BiRNN, BiGRU, BiLSTM)** in Python using Theano, the code is generic and therefore can be applied to any sequence modelling task, however, as an example I have applied these models on word & character level language modelling.

## Dependencies
* [Theano](http://deeplearning.net/software/theano/)
* NLTK

```bash
import nltk
packages = ['punkt']
nltk.download(packages)
```

## Useful Links

* [Andrej Karpathy's awesome blog post on RNN](http://karpathy.github.io/2015/05/21/rnn-effectiveness)
* [Christopher Olah's blog post on LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs)
* [Generating Text with Recurrent Neural Networks - Ilya Sutskever](http://www.cs.utoronto.ca/~ilya/pubs/2011/LANG-RNN.pdf)
* [Alex Graves PHD thesis](http://www.cs.toronto.edu/~graves/phd.pdf)
* [Gated Feedback Recurrent Neural Networks - Chung et al.](http://arxiv.org/pdf/1502.02367v4.pdf)
* [Long Short-Term Memory - Hochreiter & Schmidhuber](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
* [Bidirectional Recurrent Neural Networks - Schuster & Paliwal](http://www.di.ufpe.br/~fnj/RNA/bibliografia/BRNN.pdf)

## Usage
The code structure and code iteself is fairly straight forward and could be understood in the first glimpse.

### Data
All input data is stored inside the `data/` directory. There is already a dummy data which is in-fact excerpts from 'Beyond Good and Evil by Friedrich Nietzsche'. Karpathy have provided few more [datasets](http://cs.stanford.edu/people/karpathy/char-rnn/) which are worth trying.
If you'd like to use your own data then create a single file `input.txt` and place it in the `data/` directory. For example, `data/input.txt`.


### Switching between Word & Character Level Language Modelling
`utilities\textreader.py` provide methods to read both by 'character by character' `read_char_data(..)` or by 'word by word' `read_word_data(..)`, for the given input data. Word level language modelling is usually more accurate, however, character level language modelling often generates more interesting words/patterns. Both of above methods need `seq_length` which determines the length of each stream i.e one training sample, therefore, it specifies the limit at which the gradients can propagate backwards in time and model cannot learn dependencies longer than the `seq_length` in number of characters/words. 


### Training
`train.py` provide methods for training each model, if you want to change number of hidden neurons in each layer (at the moment only single layer models are supported to keep the things simple, although adding more layers is very trivial) then change the `n_h` inside each `train(..)`, which by default is set to 100. As the model is trained it stores the current best state of the model i.e set of weights (best = least training error), the stored model is in the `data\models\MODEL-NAME-best_model.pkl`, also this stored model can later be used for resuming training from the last point or just for prediction/sampling. If you don't want to start training from scratch and instead use the already trained model then set `use_existing_model=True` in argument to `train(..)`. 
Also optimization strategies can be specified to `train(..)` via `optimizer` parameter, currently supported optimizations are `rmsprop, adam and vanilla stochastic gradient descent` and can be found in `utilities\optimizers.py`.
`b_path`, `learning_rate`, `n_epochs` in the `train(..)` specifies the 'base path to store model' (default = `data\models\`), 'initial learning rate of the optimizer', and 'number of epochs respectively'.
During the training some logs (current epoch, sample, cross-entropy error etc) are shown on console to get an idea of how well learning is proceeding, logging frequency can be specified via `logging_freq` in the train(..).
At the end of training, a plot of `cross-entropy error vs # of iterations` gives an overview of overall training process and is also stored in the `b_path`.

### Sampling
One can sample from the model (during training or from the trained model) via `model.generative_sampling(..)` by providing the initial `seed` which could be a random element (word/character) from the vocabulary, `emb_data` which is just the embeddings of our vocabulary (in our case it's just one-hot-encoding) and `sample_length` which is the length of the sample it-self. Frequency of sampling can be specified via `sampling_freq` in the `train(..)`.

**Note:** In theano the only efficient way of implementing sequence models is by using `scan`, which provides a very convenient interface to iterate over tensors, for training everything is good, however, difficulty arises when sampling from the model in-cases where output at every time-step is input to the next time step, in such cases we cannot use the `scan` we used for training because we have to call scan multiple times and each call to scan initializes the hidden-to-hidden state vector `h0` from zero, which means that while sampling we are ignoring hidden state from previous steps which is very wrong, the ugly fix is to write another scan which will be executed only *once per sample*, let it run till `sample_length` and make hidden-state & output at every time-step recurrent by specifying them in the `outputs_info`, `generative_sampling(..)` does exactly this.

Few of the samples I got from GRU while training 'Beyond Good and Evil' (only 49.4 KB) for 600 epochs are:

```bash

. and with one's belief in his nothing else is personal something part of yourself--the secont cause and eton end minists in the heart of the sensation of the condition of same will there and to be an

 the seem of secise the  with their lee no matirst goanted and endoldge, one causa proud of have origin of a desister instinct of the superous not is precisely the hearsed the houmst and endon and ear e dlig is me the sone

byither however, let us be of the case of some ontuilly and always a suire, the sain, the entire and wosless in all seriousnel to be out imperpopsysible that it is unveittendes a

```

The samples might not look very impressive, however, notice it's just a character level language modelling (it is just reading one character at a time and it only read 50 characters at once), the model does learn some words, some combination of words and some linguistic patterns like closing of quotes, commas, full stop, new line etc. Also, I was only able to train on a very small sample of 49.4 KB, more data and longer `sequence length` will definitely results in more interesting samples/sentences.

And here's the error plot (the error was still decreasing but I had to stop training as I did not wanted to burden my poor laptop beyond it's capacity).
![GRU Error Plot](/data/models/gru-error-plot.png?raw=true)
