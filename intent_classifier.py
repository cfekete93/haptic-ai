import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# The following code was modified from
# https://medium.com/analytics-vidhya/creating-your-own-intent-classifier-b86e000a4926

# Training, testing and validation dataset was obtained from
# https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json

# Global Vectors (GloVe) provided by Stanford University from
# https://www.dropbox.com/s/a247ju2qsczh0be/glove.6B.100d.txt?dl=1


class IntentClassifier:
    def __init__(self, classes, model, tokenizer, label_encoder):
        self.classes = classes
        self.classifier = model
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder

    def get_intent(self, text):
        self.text = [text]
        self.test_keras = self.tokenizer.texts_to_sequences(self.text)
        self.test_keras_sequence = pad_sequences(self.test_keras, maxlen=16, padding='post')
        self.pred = self.classifier.predict(self.test_keras_sequence)
        return self.label_encoder.inverse_transform(np.argmax(self.pred, 1))[0]

    @classmethod
    def load_intent_classifier(cls,
                               model_save_path: str = 'models/intents.h5',
                               classes_save_path: str = 'utils/classes.pkl',
                               tokenizer_save_path: str = 'utils/tokenizer.pkl',
                               label_encoder_save_path: str = 'utils/label_encoder.pkl'):
        model, classes, tokenizer, label_encoder = load_intent_classifier(model_save_path,
                                                                          classes_save_path,
                                                                          tokenizer_save_path,
                                                                          label_encoder_save_path)
        return cls(classes, model, tokenizer, label_encoder)


def load_intent_classifier(model_save_path: str = 'models/intents.h5',
                           classes_save_path: str = 'utils/classes.pkl',
                           tokenizer_save_path: str = 'utils/tokenizer.pkl',
                           label_encoder_save_path: str = 'utils/label_encoder.pkl'):
    import pickle
    from tensorflow.keras.models import load_model

    print("Loading IntentClassifier from disk...")

    model = load_model(model_save_path)

    with open(classes_save_path, 'rb') as file:
        classes = pickle.load(file)

    with open(tokenizer_save_path, 'rb') as file:
        tokenizer = pickle.load(file)

    with open(label_encoder_save_path, 'rb') as file:
        label_encoder = pickle.load(file)

    return model, classes, tokenizer, label_encoder


def generate_intent_classifier(data_path: str = 'data_full.json',
                               global_vector_path: str = 'glove.6B.100d.txt',
                               model_save_path: str = 'models/intents.h5',
                               classes_save_path: str = 'utils/classes.pkl',
                               tokenizer_save_path: str = 'utils/tokenizer.pkl',
                               label_encoder_save_path: str = 'utils/label_encoder.pkl',
                               display_history: bool = False):
    text, labels = _load_dataset(data_path)
    train_txt, test_txt, train_label, test_labels = _create_trainer_validater(text, labels)
    classes, tokenizer, word_index = _tokenize_dataset(labels, train_txt)
    train_sequences, test_sequences = _padder(train_txt, test_txt, tokenizer)
    label_encoder, train_label, test_labels = _convert_to_one_hot_encode(classes, train_label, test_labels)
    embeddings_index = _load_global_vectors(global_vector_path)
    num_words, embedding_matrix = _reduce_global_vector_to_dataset(embeddings_index, word_index)
    model = _model_prepare(num_words, train_sequences, embedding_matrix, classes)
    history = _model_train(model, train_sequences, train_label, test_sequences, test_labels)

    if display_history:
        _model_visual_accuracy(history)
        _model_visual_loss(history)

    _model_save(model, model_save_path, classes, classes_save_path, tokenizer, tokenizer_save_path, label_encoder, label_encoder_save_path)

    print("IntentClassifier generation has completed!")

    return IntentClassifier(classes, model, tokenizer, label_encoder)


# Limit the number of words we use for our model, increase may increase computation time
_max_num_words = 40000


def _load_dataset(data_path: str):
    import json

    print("Loading dataset from disk...")

    with open(data_path) as file:
        data = json.loads(file.read())

    # Loading out-of-scope intent data
    val_oos = np.array(data['oos_val'])
    train_oos = np.array(data['oos_train'])
    test_oos = np.array(data['oos_test'])

    # Loading other intents data
    val_others = np.array(data['val'])
    train_others = np.array(data['train'])
    test_others = np.array(data['test'])

    # Merging out-of-scope and other intent data
    val = np.concatenate([val_oos, val_others])
    train = np.concatenate([train_oos, train_others])
    test = np.concatenate([test_oos, test_others])
    data = np.concatenate([train, test, val])
    data = data.T

    return data


def _load_global_vectors(file_path: str):
    print("Loading GloVe from disk...")

    embeddings_index = {}
    with open(file_path, encoding='utf8') as file:
        for line in file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    return embeddings_index


def _create_trainer_validater(text, labels):
    from sklearn.model_selection import train_test_split

    print("creating trainer and validaters...")

    train_txt, test_txt, train_label, test_labels \
        = train_test_split(text, labels, test_size=0.3)

    return train_txt, test_txt, train_label, test_labels


def _tokenize_dataset(labels, train_txt):
    from tensorflow.keras.preprocessing.text import Tokenizer

    print("tokenizing dataset...")

    global _max_num_words
    classes = np.unique(labels)

    tokenizer = Tokenizer(num_words=_max_num_words)
    tokenizer.fit_on_texts(train_txt)
    word_index = tokenizer.word_index

    return classes, tokenizer, word_index


def _padder(train_txt, test_txt, tokenizer):
    print("padding our training dataset...")

    ls = []
    for c in train_txt:
        ls.append(len(c.split()))
    max_len = int(np.percentile(ls, 98))
    train_sequences = tokenizer.texts_to_sequences(train_txt)
    train_sequences = pad_sequences(train_sequences, maxlen=max_len, padding='post')
    test_sequences = tokenizer.texts_to_sequences(test_txt)
    test_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')
    return train_sequences, test_sequences


def _convert_to_one_hot_encode(classes, train_label, test_labels):
    from sklearn.preprocessing import OneHotEncoder, LabelEncoder

    print("converting labels to one-hot encoding form...")

    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(classes)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoder.fit(integer_encoded)

    train_label_encoded = label_encoder.transform(train_label)
    train_label_encoded = train_label_encoded.reshape(len(train_label_encoded), 1)
    train_label = onehot_encoder.transform(train_label_encoded)

    test_labels_encoded = label_encoder.transform(test_labels)
    test_labels_encoded = test_labels_encoded.reshape(len(test_labels_encoded), 1)
    test_labels = onehot_encoder.transform(test_labels_encoded)

    return label_encoder, train_label, test_labels


def _reduce_global_vector_to_dataset(embeddings_index, word_index):
    print("reducing GloVe size to size of our dataset corpus...")

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()

    global _max_num_words
    num_words = min(_max_num_words, len(word_index)) + 1

    embedding_dim = len(embeddings_index['the'])

    embedding_matrix = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= _max_num_words:
            break
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return num_words, embedding_matrix


def _model_prepare(num_words, train_sequences, embedding_matrix, classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation, Bidirectional, Embedding

    print("preparing model...")

    model = Sequential()

    model.add(Embedding(num_words, 100, trainable=False, input_length=train_sequences.shape[1], weights=[embedding_matrix]))
    model.add(Bidirectional(LSTM(256, return_sequences=True, recurrent_dropout=0.1, dropout=0.1), 'concat'))
    model.add(Dropout(0.3))
    model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(classes.shape[0], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

    return model


def _model_train(model, train_sequences, train_label, test_sequences, test_labels):
    import datetime
    print("training model (this may take a few hours)...")
    time = datetime.datetime.now()
    print(f'Start Time: {time}')

    history = model.fit(
        train_sequences,
        train_label,
        epochs=20,
        batch_size=64,
        shuffle=True,
        validation_data=[test_sequences, test_labels]
    )
    now = datetime.datetime.now()
    print(f'End Time: {now} | Total Time: {now-time}')

    return history


def _model_visual_accuracy(history):
    import matplotlib.pyplot as plt
    # %matplotlib inline  # Only use in Jupiter Notebook!

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def _model_visual_loss(history):
    import matplotlib.pyplot as plt
    # %matplotlib inline  # Only use in Jupiter Notebook!

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def _model_save(model, msp: str, classes, csp: str, tokenizer, tsp: str, label_encoder, lesp: str):
    import pickle

    print("saving trained model files...")

    model.save(msp)

    with open(csp, 'wb') as file:
        pickle.dump(classes, file)

    with open(tsp, 'wb') as file:
        pickle.dump(tokenizer, file)

    with open(lesp, 'wb') as file:
        pickle.dump(label_encoder, file)
