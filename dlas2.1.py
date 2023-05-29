# %%
import nltk
nltk.download('punkt')
# import nltk
nltk.download('wordnet')
nltk.download('stopwords')


# %%
import pandas as pd
import numpy as np
import re

import string
from string import digits

import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


from nltk.tokenize import word_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# %%
df = pd.read_csv("data/IMDB Dataset.csv")
df.head()

# %%
from sklearn import preprocessing
le =  preprocessing.LabelEncoder()
df["sentiment"] = le.fit_transform(df['sentiment'])

# %%
df.head

# %%
df.isnull().sum()


# %%
X = df["review"]
y = df["sentiment"]

# %%
def stringprocess(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    
    return text

# %%
def  textpreprocess(text):
    
    text = map(lambda x: x.lower(), text) 
    text = map(lambda x: re.sub(r"https?://\S+|www\.\S+", "", x), text) 
    text = map(lambda x: re.sub(re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});"),"", x), text) 
    text = map(lambda x: re.sub(r'[^\x00-\x7f]',r' ', x), text) 

    text = map(lambda x: x.translate(str.maketrans('', '', string.punctuation)), text) # Remove punctuations
    
    
    remove_digits = str.maketrans('', '', digits)
    text = [i.translate(remove_digits) for i in text]
    text = [w for w in text if not w in stop_words]
    text = ' '.join([lemmatizer.lemmatize(w) for w in text])
    text = text.strip()
    return text

# %%
!unzip /usr/share/nltk_data/corpora/wordnet.zip -d /usr/share/nltk_data/corpora/

# %%


# %%
X = X.apply(lambda x: stringprocess(x))
word_tokens = X.apply(lambda x: word_tokenize(x))

preprocess_text = word_tokens.apply(lambda x: textpreprocess(x))
preprocess_text[0]

# %%
training_portion = 0.8
train_size = int(len(preprocess_text) * training_portion)

train_data = preprocess_text[0: train_size]
train_labels = np.array(y[0: train_size])

validation_data = preprocess_text[train_size:]
validation_labels = np.array(y[train_size:])


print(len(train_data))
print(len(train_labels))
print(len(validation_data))
print(len(validation_labels))

# %%
vocab_size = 500
oov_tok = '<OOV>'

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index
dict(list(word_index.items())[0:10])

# %%
train_sequences = tokenizer.texts_to_sequences(train_data)
print(train_sequences[10])

# %%
embedding_dim = 50
max_length = 70

trunc_type = 'post'  
padding_type = 'post'

# %%
train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
print(len(train_sequences[0]))
print(len(train_padded[0]))

# %%
train_padded[0]

# %%
validation_sequences = tokenizer.texts_to_sequences(validation_data)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)

# %%
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_data(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
print(decode_data(train_padded[10]))
print('---')
print(train_data[10])

# %%
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.LSTM(64,activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()

# %%
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# %%
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# %%
num_epochs = 5
history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(validation_padded, validation_labels), verbose=2)

# %%
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# %%

seed_text = "wonderful little production br br filming technique unassuming old time bbc fashion give comforting sometimes discomforting sense realism entire piece br br actor extremely well chosen michael sheen got polari voice pat truly see seamless editing guided reference williams diary entry well worth watching terrificly written performed piece masterful production one great master comedy life br br realism really come home little thing fantasy guard rather use traditional would ream technique remains solid disappears play knowledge sens particularly scene concerning orton halliwell set particularly flat halliwell mural decorating every surface terribly well done"
token_list = tokenizer.texts_to_sequences([seed_text])[0]
token_list = pad_sequences([token_list], maxlen=max_length-1, padding=padding_type, truncating=trunc_type)
predicted = (model.predict(token_list, verbose=0) > 0.5).astype("int32")

if predicted[0][0] == 0:
    print("Negative")
else:
    print("Positive")


# %%
preprocess_text[1]

# %%



