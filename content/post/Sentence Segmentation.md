---
author: "Paul Jeffrey"
title: "Unraveling the Secrets of Raw Text: A Journey Through Word, Sentence Segmentation and Capitalization with Python (Part 2)"
date: "2023-11-21"
draft: false
cover:
    image: "/static/images/Text segmentation/Sentence Segmentation.jpg"
    alt: 'Sentence Segmentation'
description: "A Journey Through Word, Sentence Segmentation and Capitalization with Python (Part 2)"
tags: [
    "neural networks","natural language processing","word segmentation"
]

---

# Title: Unraveling the Secrets of Raw Text: A Journey Through Word, Sentence Segmentation and Capitalization with Python (Part 2)

In the Part 1 of this article, we focused on training a Neural Network that can segment a raw sequence (stream) of text characters into words and we were able to achieve an accuracy of about 95% using some complex feature engineering. By the way, if you have not checked out the Part 1 of this lovely project's article, you can find it  here . I advice that you follow the first part of this project first before reading this. In this article, we are going to focus on predicting/forming full sentences given a corpus of words (without sentence demarcations) predicted by our "word segmentor" model from a raw sequence of text characters. We will call this text segmentation.

Example:
```My name is mary I am 10 years old I live in Nigeria``` 
is segmented into 
```My name is mary. I am 10 years old. I live in Nigeria.```


## Introduction 

Once the words have been predicted by the "word segmentor" model, the next challenge is to segment these words into meaningful sentences. This is crucial for understanding the context and flow of the text.

For this task, we utilize a combination of rule-based and machine learning techniques. Rule-based methods involve defining a set of rules that identify sentence boundaries based on punctuation marks, conjunctions, and other linguistic cues. Machine learning methods, on the other hand, train a model on annotated data to predict sentence boundaries. We will be making use of the brown corpus (found in the NLTK library) as we did in the last article.

Let's go!

We will be preparing data mostly in the same format as in the previous article to extract features from data. Here, we are also going to be extracting features from the neighbouring words.

In addition to all the calculations we made in the previous article during feature extraction, we are going to make use of the 'Part of Speech' tags of each word here. This POS tags come with every word in the brown corpus. For the modelling of commas and full stop insertions, this POS tags play a huge role. For example, we know that many principles of use of the comma punctuation mark solely involves the identification of the part of speech of the words. We will also be using the length of words, tag congruity between neigbouring words etc. Everything is documented below. Please read through the code to understand the concepts and ideas.


```python
# This function extracts a list of list of tagged_sentences from the brown_corpus
# Given its argument, it gets the whole corpus or the number of genres provided.
def ext_tagged_corpus(num_of_genre=None):
  #print('yes')
  if num_of_genre: # if genre is given, then get only the sentences in that genre
    tagged_corpus = []
    for i in range(num_of_genre):
      tagged_corpus.extend(brown.tagged_sents(categories=brown.categories()[i]))
    return tagged_corpus

  else: # Else, get all sentences.
    return brown.tagged_sents()
```

For the sentences, we are going to tag a word with 'E' if it ends the sentence (comes before the full stop), and tag a word with 'P' (pause) if it comes before a comma and tag a word inside a sentence with an 'I'.

PS: Some functions here handle processing and generation of data for both sentence segmentation task and capitalization task which we will cover in part 3 of this series.

```python
sent_cat_targ = {'E' : 0, 'P': 1, 'I': 2}

#For true casing of words in sentences , we are going to tag a word as 'T' for titled if the #first letter of the word is in capital
# and tag it 'N' for not titled if the first letter isnt a capital letter.

case_cat_targ = {'T' : 0 , 'N': 1}

# This function creates the dataset, extracting all the words , part of speech tag associated with them and the target_tags in different lists.

def create_sent_data(tagged_sent,casing=False):
  words = []
  tags = []
  target_data = []

  if not casing: # if it is not the casing task (e.g John met the President) but segmentation task

    for each_sent in tagged_sent: # Go through each sentence in the list of sentences.
      for index, word in enumerate(each_sent): # Enumerate and go through words in each sentence
        if word[0] == '.' or  word[0] == ',': # if the current word is '.' or ',' then, continue to the next word
          continue

        if index < len(each_sent) -1: # if its not the last word ('.') in this sentence.

          if each_sent[index+1][0] == '.' : # if the current word is not the last and the word in front of the current word is a '.' , append the appropriate tag
            target_data.append('E')

          elif index != each_sent[-1] and each_sent[index+1] == ',': # Do the same for ','
            target_data.append('P')

          else: # Do the same for every other word
            target_data.append('I')

          words.append(word[0].lower()) # Append word in lower case
          tags.append(word[1])
  else:
    for each_sent in tagged_sent: # Go through each sentence in the list of sentences.

      for index, word in enumerate(each_sent): # Enumerate and go through words in each sentence

          if word[0].istitle() : # if the first letter of the word is cased
            target_data.append('T') # return 'T' for titled
          else: # Else return 'N' for not titled
            target_data.append('N')

          words.append(word[0].lower())
          tags.append(word[1])

  return words, tags, target_data # return all words, their tags and their target_data.

# Function that calculates the length of a word.
def word_length(word):
  return len(word)
```

Now, let's check out the length of our training corpus

```python
corpus = ext_tagged_corpus(8) # extract training corpus
len(corpus)
```
```
35104
```
Next, we extract words, tags and target data from this corpus below:

```python
words, tags , target_data = create_sent_data(corpus) # extract the words, various tags and target data
print(len(words) , len(target_data))
```
```
657589 657589
```

Let's take a look at the words and their corresponding tags.

```python
sent_df = pd.DataFrame({'words': words, 'tags': tags})
sent_df.head()
```
<img src="/static/images/Text segmentation/part2/Table 1.png" alt="Table 1">

Now, let's save the a list of the unique tags to a file permananently. We do this so that our models never return an error when they come across an entirely new data that has a tag it has never seen before. This way, we can set these tags to 'UNK' values.

```python
tag_list = sent_df['tags'].unique()
save_to_file(tag_list, os.path.join(FILE_PATH,'tags_list'))

print('Number of unique words in dataset: ', len(sent_df['words'].unique()))
print('Number of unique tags in dataset: ', len(sent_df['tags'].unique()))
````
```
Number of unique words in dataset:  38191
Number of unique tags in dataset:  437
```

We have about 38,191 unique words and it will be difficult to create a one_hot_encoding with this amount because the matrix will be very scarce. Instead, we could create a co-occurrence matrix (function defined below) to scale the data with StandardScaler() and then perform a PCA transform to reduce the dimensionality to a few hundred columns. This will help us represent the words uniquely. The co-occurrrence matrix will be a (38191 x 38191) matrix since we have 38,191 unique words in our extracted corpus. Then we can reduce this dimension using PCA to a few hundreds.

This will require setting a unique representation for any unknown (new) word the model will encounter during test. We can do this by replacing the least occuring word in the train set with an 'UNK' tag for example.

However, to reduce computational complexity, we are going to represent each word by its word length, frequency, probability in corpus, tagset,tagset of neigbouring words, transitional probabilities into neigbouring words and probabilities of the various sequences. We do this because we also know that the frequency and probability of the word in the whole corpus is unique to each word no matter where they occur.

I believe that this may reduce the performance of the model. This is because this model makes some assumptions. one of the very obvious one is that every unique word has a unique freq and probability of occurrence.

```python
from collections import defaultdict

# I do not advise this for this task though.
def co_occurrence(words, window_size): #
    d = defaultdict(int)
    vocab = set()
    for i,word in enumerate(words):
      vocab.add(word)  # add to vocab
      next_token = words[i+1 : i+1+window_size]
      for t in next_token:
        key = tuple( sorted([t, word]) )
        d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
      df.at[key[0], key[1]] = value
      df.at[key[1], key[0]] = value
    return df

sent_df['word_length'] = sent_df['words'].apply(word_length) # calculate length of words.
sent_df.head()
```
Let's find the least occuring tag in the 'tags' column. We wiil use this to represent any new tag the transformer model we meet in the future when it encounters an entirely new text corpus.

```python
sent_df.groupby('tags').count()
```

<img src="/static/images/Text segmentation/part2/Table 2.png" alt="Table 2">

We see that the 'WRB+DOZ' occurs just once in the corpus. We shall use this as our fill_value.

```python
least_tag = 'WRB+DOZ'

# we convert the word_length to categories of exactly 1, 2 or greater than 2
def convert_length(word_length):
  if word_length == 1:
    return 0
  elif word_length == 2:
    return 1
  else:
    return 2

# Function gets the universal (non-specific) POS tags for each word. e.g 'NOUN' instead of 'NP' that stands for 'proper noun'.
def get_universal_tag(word):
  tag = nltk.pos_tag([word],tagset='universal')
  #print(tag)
  return tag[0][1]

# function checks if index word is a punctuation. We do this because we are sure that punctuations cant exist before a '.' or ','
def is_punc(word):
  if (word == '?' ) or (word == '!') or (word == ':') or (word == ';') or (word == '-'):
    return True
  else:
    return False

# function check if preceeding tag = tag of index character  = subsequent tag
# This very important for comma insertion . If the have the same tag, probably they are a set of listed items
# e.g I have mangoes ,apples, pineapples. This statement contains a list of 3 proper nouns.

def check_tag_congruity(row):
  if row['pre-1 tags'] == row['tags'] and row['tags'] == row['post-1 tags']:
    return 0
  elif row['pre-1 tags'] == row['tags']:
    return 1
  elif row['tags'] == row['post-1 tags']:
    return 2
  else:
    return 3

# This function process the whole data for the subsequent transformers in our pipeline.
# A lot of the functions used for the word segmentation model were also used here so they represent the same
# basic calculations and extractions.

def process_sent_data(df,n_gram=1,fill_value = 'WRB+DOZ'):
  df['word_length'] = df['words'].apply(word_length)
  df['word_length_class'] = df['word_length'].apply(convert_length)
  df['uni_tags']  = df['words'].apply(get_universal_tag)


  df = cal_freq_prob(df, 'words',index=True,return_freq=True) # frequency and prob of 'words'

  df = cal_freq_prob(df, 'tags', return_freq=True) # freq and prob of 'tags'

  # block of code extract context and its features (words around index word)
  # We are not bothered about the fill value for the words column here because the 'words' column will be deleted later.
  df = extract_context(df, 'words','words', n_gram=n_gram,fill_value=fill_value)
  df = extract_context(df,'tags','tags',n_gram=n_gram, fill_value=fill_value)

  # # block of code generates sequences by concatenating the index word to the 'before' and 'after' words
  # # block of code also calculates other properties : transitional prob, prob. of word sequence etc

  # Create pre, post and whole sequence
  df = process_seq_columns(df,'tags','tags',n_gram)
  df = process_seq_columns(df, 'words','words',n_gram)

  # Calculate frequency, prob of sequence and transitional probability for each character.

  # We check tag congruity between the pre-1 tag, index tag and post-1 tag
  df['tag_congruity']  = df.apply(check_tag_congruity,1)

  # Check that the current word is a punctuation mark
  df['is_punc'] = df['words'].apply(is_punc)

  # calculate frequency , probability of occurence of all extracted sequences.
  for col in ['tags', 'words']:
    for i in range(1, n_gram + 1):
      df = cal_freq_prob(df, f'pre-{i} seq {col}',return_freq=True,seq=True) #for presequence
      df = cal_freq_prob(df, f'post-{i} seq {col}',return_freq=True,seq=True) # for post sequence
      df = cal_freq_prob(df,f'whole-seq {col}',return_freq=True,seq=True) # for whole sequence with index in middle

  # calculate the trans. prob. of the index character to the next and for increasing order of sequence to the next character.
  # Do the same for the 'tags' column.
  df = cal_transitional_prob(df, 'words' ,n_gram)
  df = cal_transitional_prob(df, 'tags', n_gram)

  # Since we represent words here by their freq and prob, to give each word more uniqueness, we shift the columns up and down
  # By a number of steps so they get infomation about the characters that exist before and after them.
  df = extract_context(df,'words-freq','words-freq',n_gram=1, fill_value= 0)
  df = extract_context(df,'words-prob','words-prob',n_gram=1, fill_value= 0)

  # Now we drop all the columns we dont need after we have extracted and calculated the ones we need below:
  if n_gram == 1: # if we considered only one character pre and post
    drop = [col for col in df.columns if (col.endswith('-freq') and ('words' not in col))]
    drop.append('whole-seq words')
    drop.append('whole-seq tags')
    drop.append('pre-1 seq words')
    drop.append('pre-1 seq tags')
    drop.append('post-1 seq words')
    drop.append('post-1 seq tags')
    drop.append('post-1 words')
    drop.append('pre-1 words')

  else: # if we considered more than one character pre and post.
    drop = [col for col in df.columns if (col.endswith('-freq') and ('words' not in col))]  + \
        [ col for col in df.columns if (col.startswith('pre-2') and col.endswith('tags'))] + \
        [col for col in df.columns if (col.startswith('post-2') and col.endswith('tags'))] + \
        [col for col in df.columns if (col.startswith('pre-2') and col.endswith('words'))] + \
        [col for col in df.columns if (col.startswith('post-2') and col.endswith('words'))]# + \
        #[col for col in df.columns if (col.startswith('post-2') and col.endswith('tags'))]
    drop.append('whole-seq words')
    drop.append('whole-seq tags')
    drop.append('pre-1 seq words')
    drop.append('pre-1 seq tags')
    drop.append('post-1 seq words')
    drop.append('post-1 seq tags')
    drop.append('post-1 words')
    drop.append('pre-1 words')


  drop.append('words')

  df = drop_column(df, drop)

  return df

sent_cat = [ 'tags', 'word_length', 'uni_tags', 'pre-1 tags', 'post-1 tags', 'tag_congruity', 'is_punc']
```

Now, we compute and extract all our features so they can be preprocessed into the format that is acceptable to our machine learning model.

```python
train = process_sent_data(sent_df, 2) # process the data.
train.head()
```
<img src="/static/images/Text segmentation/part2/Table 3.png" alt="Table 3">

```python
print(len(train['tags'].unique()))
print(len(train['pre-1 tags'].unique()))
print(len(train['post-1 tags'].unique()))
```
```
437
437
437
```
Before preprocessing data, we need to include the 'UNK' tag we used to fill up the 'pre' and 'post' tags column we created to the 'tags' column. We do this so they all have equal representation dimensions.

```python
# Extract the unique values in categories and append 'UNK' to the tags column's unique values
sent_cat_values = [list(train[col].unique()) for col in sent_cat ]
sent_cat_values[0].append('UNK')

# We sort all the values in each category because the Ordinal Encoder will not accept numerical values that are not sorted.
for cat in sent_cat_values:
  cat.sort()
```

We then preprocess data using the same function we used for the word segmentation model.

```python

train , pipeline = preprocess_data(train, sent_cat, pipe_name='sentence_transformer')
train
```
```
<657589x1393 sparse matrix of type '<class 'numpy.float64'>'
	with 19760410 stored elements in Compressed Sparse Row format>
```

```python
# Convert target_data variables from string format to numerical format
target_data = to_cat(target_data, sent_cat_targ)

# change target to categorical format (One hot encoding)
target_data = tf.keras.utils.to_categorical(target_data)

# Split data into train and test set
X_train, X_test , y_train, y_test  = train_test_split(train, target_data, test_size= 0.2,
                                                      random_state=42,shuffle=False)
```

Now, Let's create our deep learning model for training. I also suggest the use of SVMs for training because they should work well as classifiers when data is represented in a high dimensional format. Xgboost is another option to consider.

```python
# Get input and output dim
input_dim = X_train.shape[1]
output_dim = 3 # because we have only 3 targets

model = create_model(input_dim, 1600, 1000, output_dim)
model.summary()
```
```
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_4 (Dense)             (None, 1600)              2230400   
                                                                 
 dense_5 (Dense)             (None, 1000)              1601000   
                                                                 
 dense_6 (Dense)             (None, 1000)              1001000   
                                                                 
 dense_7 (Dense)             (None, 3)                 3003      
                                                                 
=================================================================
Total params: 4835403 (18.45 MB)
Trainable params: 4835403 (18.45 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```

Let's train the model now!
```python
# Schedule early stopping so that model can be stopped when there is no longer improvement
# after at least 5 steps.
epochs = 20
batch_size  = 512
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Schedule learning rate when you hit a plateau
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=1E-7)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test, y_test),callbacks=[es,lr])
```
```
Epoch 1/20
1028/1028 [==============================] - 27s 23ms/step - loss: 0.1124 - accuracy: 0.9577 - val_loss: 0.0985 - val_accuracy: 0.9631 - lr: 0.0010
Epoch 2/20
1028/1028 [==============================] - 22s 21ms/step - loss: 0.1006 - accuracy: 0.9607 - val_loss: 0.0956 - val_accuracy: 0.9639 - lr: 0.0010
Epoch 3/20
1028/1028 [==============================] - 16s 14ms/step - loss: 0.0965 - accuracy: 0.9619 - val_loss: 0.0952 - val_accuracy: 0.9644 - lr: 0.0010
Epoch 4/20
1028/1028 [==============================] - 16s 14ms/step - loss: 0.0928 - accuracy: 0.9631 - val_loss: 0.0959 - val_accuracy: 0.9637 - lr: 0.0010
Epoch 5/20
1028/1028 [==============================] - 15s 14ms/step - loss: 0.0895 - accuracy: 0.9638 - val_loss: 0.1010 - val_accuracy: 0.9634 - lr: 0.0010
Epoch 6/20
1028/1028 [==============================] - 15s 14ms/step - loss: 0.0861 - accuracy: 0.9651 - val_loss: 0.1011 - val_accuracy: 0.9638 - lr: 0.0010
Epoch 7/20
1028/1028 [==============================] - 15s 14ms/step - loss: 0.0773 - accuracy: 0.9678 - val_loss: 0.1139 - val_accuracy: 0.9626 - lr: 1.0000e-04
Epoch 8/20
1028/1028 [==============================] - 15s 14ms/step - loss: 0.0741 - accuracy: 0.9689 - val_loss: 0.1226 - val_accuracy: 0.9628 - lr: 1.0000e-04
Epoch 8: early stopping
```
Let's plot the traning history.

```python
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

```sentence segmentor image here```

Let's save our model.

```python
model_name = 'Comma_sentence_segmentor'
model_path = os.path.join(FILE_PATH,model_name)

model.save(model_path)

```

We see that our model generalizes well and has about 96.6% accuracy on the validation set!

That's all for this part. Check out the last part of this series <a href="/projects/True Casing.md">here</a>. It is going to focus on True casing (Capitalization) of important words in the sentences formed. In the next article you will also get to see all of our trained models in action as they attempt to process a raw sequence stream of text characters to cased sentences which is very easy to read and understand.

See you <a href="/projects/True Casing.md">there</a>!.

