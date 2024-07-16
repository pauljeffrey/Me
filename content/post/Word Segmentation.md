---
author: "Paul Jeffrey"
title: "Unraveling the Secrets of Raw Text: A Journey Through Word, Sentence Segmentation and Capitalization with Python (Part 1)"
date: "2023-11-03"
draft: false
cover:
    image: "/static/images/Text segmentation/word segmentation.jpg"
    alt: 'Word Segmentation'
description: "A Journey Through Word, Sentence Segmentation and Capitalization with Python (Part 1)"
tags: [
    "neural networks","natural language processing","word segmentation"
]

---

# Title: Unraveling the Secrets of Raw Text: A Journey Through Word, Sentence Segmentation and Capitalization with Python (Part 1)

## Introduction

In the realm of Natural Language Processing (NLP), engineers/data scientist can sometimes be faced with raw, unprocessed text which may present a unique challenge. Unlike structured or clean data, raw text may lack word boundaries, sentence boundaries, and proper noun identification. It's a jumbled mess of letters that can leave even the most seasoned NLP engineer scratching their heads.

But fear not, for I embarked on a captivating journey to develop a machine learning system that transforms this unstructured chaos into comprehensible, well-demarcated words and sentences. Through the magic of word segmentation, sentence segmentation, and capitalization, this system achieved an astonishing accuracy of approximately 97%! For this task, we will be making use of the brown corpus from the NLTK library.

This is the first of a series of 3 articles and all are interdependent. Here is a summary of what we want to achieve below:

```python
# Raw text
"danmorgantoldhimselfhewouldforgetannturnerhewaswellridofherhecertainlydidn'twantawifewhowasfickleasannifhehadmarriedherhe'dhavebeenaskingfortroublebutallofthiswasrationalizationsometime..."

# Final output
"dan morgan told himself he would forget ann turner he was well rid of her he certainly did n't want a wife who was fickle as ann if he had married her he 'd have been asking for trouble but all of this was rationalization sometime..."
```

Are you ready? Let's go!

## Data Download
First, we import all the necessary libraries (we will be using tensorflow for our models here):

```python
import nltk
import sklearn
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder
import os
import pickle
from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
```

Also, we will be making use of a small chunk of the brown text corpus found in the nltk library. We will remove all upper cases, sentence demarcations, word demarcations to get our raw text.

```python
nltk.download('brown')
from nltk.corpus import brown
from matplotlib import pyplot

nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
nltk.download('punkt')
```

First, let's check out the number of words in the brown corpus:

```python
print('Length of all words in all genres: ', len(brown.words()))

```
```
Length of all words in all genres:  1161192
```

Here, we see that there are more than 1 million words in all genre of the brown corpus. To model word segmentation, the list of characters will grow exponentially. Therefore, we will consider just a few number of genre to train our word segmentation model.

```python
# extract corpus for word segmentation training/sentence segmentation.
genres = brown.categories() # get genre categories
def extract_corpus(doc= 'char',num_of_genre= 9):
  corpus = '' #extract words from all genre into this variable
  if doc == 'char':
    for i in range(num_of_genre):
      corpus += ' '.join(brown.words(categories=genres[i]))
      corpus = corpus.replace('.','') 
      corpus = corpus.replace(',','') 
    return corpus
  else:
    corpus += ' '.join(brown.words(categories=genres[i]))
    corpus = corpus.replace('.','')
    corpus = corpus.replace(',','')
    return corpus

# extract Text
corpus = extract_corpus(num_of_genre=8)
print(len(corpus))
```
```
3817507
```
We will use this chunk to create our raw data. 

## Word Segmentation
The first step in our quest is to tackle word segmentation. This involves breaking down the continuous stream of characters into individual words. It's like taking a jumbled puzzle and painstakingly piecing it together, one word at a time.

To accomplish this, we must create a model that can segment raw texts (continuous stream of characters) into individual words. Therefore, we need to create a training dataset to train this model. 

By feeding the model with a sequence of characters and their corresponding features (such as part-of-speech tags), it learns to predict the boundaries between words.

To achieve this, we will create characters (in the order they occur in the raw data) and targets for each character in our raw data. The target labels will be as follows:
- S for single word
- E for end of word
- I for inside word
- B for beginning of word. 

We could also add an extra tag for 'end of sentence' if we consider a joint (word and sentence segmentation). we will store the training data in an array. Then, we shall model this problem as a classification problem where we map each character to its corresponding target as stated above.
Here is the code to handle all of this:

```python
word_cat_targ = {
    'B' : 0, #
    'I' : 1,
    'E' : 2,
    'S' : 3
}
# Convert target to categorical variables
def to_cat(targ, cat_target):
  target = []
  for i in targ:
    target.append(cat_target[i])
  return np.array(target)

def create_train_data(corpus):
  # if ' ' exists before and after, its a single word
  # if ' ' exists only after character then it marks end of word
  # if ' ' exists only before character then its the beginning of the word
  # if ' ' does not exist before or after the character, then character exists within a word.
  train_data = []
  target_data = []
  length = len(corpus)
  for index,char in enumerate(corpus):
    # ignore space characters
    if char == ' ':
      continue

    train_data.append(char) # append character

    if index == 0:# if beginning of corpus, tag character as 'B'
      print(char)
      target_data.append('B')
      continue


    if index +1 < length : # If character is not the last character in the corpus
      #if space precedes and supersedes character
      if corpus[index-1] == ' ' and corpus[index+1] == ' ':
        target_data.append('S')
      # if space exist only before char
      elif corpus[index-1]== ' ' :
        target_data.append('B')
      # if space exists after character
      elif corpus[index+1] == ' ':
        target_data.append('E')
      # if no space before and after character
      else :
        target_data.append('I')
    # if last character in the corpus
    else:
      target_data.append('E')

  return train_data, target_data
```

Let's generate the data:
```python
train_data, target_data = create_train_data(corpus.lower())
target_data = to_cat(target_data, word_cat_targ)
```
Next, we do a little analysis on the training data. We store it in a dataframe for easy analysis.

```python
train_data = pd.DataFrame({'Characters': train_data})
train_data.tail()
```
<img src="images/Text segmentation/part1/word_segmentation_Table 1.png" alt="Table 1">

### Model's Objective:

Let's talk about the model a little. The task here is to classify each character of a sequence (raw text) as end of word E, beginning of word B, inside word I, single word S.

The major problem here is that these characters have no cues or have no features associated with them. Therefore, important features that will aid our model classify the characters accurately have to be generated.


### Feature Extraction and Engineering :

We need to generate features that will increase the accuracy of the classification probability. We will consider the character itself and the properties of the sequence (n-gram) of characters before and after it.

Using this method, we get not only the feature representation of the index character, but also the contextual information (in what context the character was used ) since we consider the characters existing before and after it. In this project, we will consider the following characteristics for feature engineering:


1. Features of the index character:

- one hot encoding representation
- type of char : letter, number, symbol (punctuation)
- class of char : vowel ,consonant, punctuation (e.g '!'), sign (e.g '$')

2. Properties of each character associated/surrounding the index character (This will serve as the context (sequence pattern) for the index character):

- One hot encoding representation
- type of char
- class of char


3. properties of sequence pattern (i.e pre-character(s), index character,post-character(s)).
We consider the characters that occur before and after the index character. To this, we consider the following:

4. frequency of sequence pattern in corpus.

5. Transitional probability: This is the probability of transitioning into another non-space character given the index character. We shall look at this property in both forward and backward direction to capture the pre and post characters. This is an intuitive entity. If the likelihood of seeing another non-space character after a particular index character is high, it means that both characters (index and one after) are more likely to be part of a word and so there should be no space between them. If its low, it means its less likely that these characters exist together in a word and so should be separated by a space forming different words. It is calculated by :

\[ \text{Probability}(x, y) = \frac{\text{Frequency}(x \text{ and } y \text{ occurring together})}{\text{Frequency}(x)} \]

where:
- \( x \) is the index character
- \( y \) is the pre or post-character

we can also take this a step further by calculating the probability of a sequence of character transitioning into another character. For example, given the word 'dinner', we can consider the transitional probability of the sequence 'di' transitioning into the letter 'n'. The higher this probability, the more likely that these three letters are part of a word and should not be separated by a space.

\[ \text{Probability}(\text{sequence} \rightarrow y) = \frac{\text{Frequency}(\text{sequence} \text{ followed by } y)}{\text{Frequency}(\text{sequence})} \]

6. probability of the sequence pattern : For example, the probability of the sequence pattern 'the' given the corpus if we consider a tri-gram pattern (3 character sequence) is given by :


\[ \text{Probability}(\text{pattern}) = \frac{\text{Frequency}(\text{pattern})}{\text{Total number of characters in corpus}} \]

Another is:

\[ \text{Probability}(\text{n-gram}) = \frac{\text{Frequency}(\text{n-gram})}{\text{Total number of occurring n-grams}} \]


- we may also consider the prob. of the class pattern of the sequence (e.g the character pattern 'the' is a 'consonant consonant vowel' pattern). It will be given by:

\[ \text{Probability}(\text{class pattern}) = \frac{\text{Frequency}(\text{class sequence})}{\text{Total number of sequence types}} \]

#### Note:

'n-gram' here refers to the character(s) before and after the current character. It was used here for convenience.

Since, the following models will be heavily reliant on frequencies and probabilities, the models will do way better with a large corpus! Secondly, text processing will definitely be slower here because it will take more time extracting these features from a large enough dataset. Therefore, we trade computation, time for better perfomance of the model in all tasks in this notebook.

Lets check out the unique characters in the dataset. We have 55 unique characters in the dataset.

```python
print(len(train_data['Characters'].unique()))
train_data['Characters'].unique()
```

Now, we write some functions that will convert the meaningful properties of each character into class label encoding. Also as said earlier , we will create context for each characters by creating columns for the set of characters (n-gram) that exist both before and after them. This can be likened to helping the model see the future and past characters before making a decision.

This gives an idea of the likelihood of this sequence of string being part of a word (using the frequency of occurence of the sequence pattern in the whole corpus.

We do this by shifting the columns ('Character column') by a number of steps both up and down. We also use a custom label for all NaN values that appear in the newly created columns. This custom label will represent the null values.

For this first model, we use a 'UNK' to represent null values. We also use an n_gram of 2. This means we will extract the 2 characters before and after every character.

Create function that generates n-gram characters preceding and superceding the index character. We will work with just 1 or 2 characters pre and post index character. The reason is because since we are using a one hot encoding representation, we do not want too much of a sparse matrix as data for our model. We first define vowels, consonants, punctuations and symbols.

```
array(['d', 'a', 'n', 'm', 'o', 'r', 'g', 't', 'l', 'h', 'i', 's', 'e',
       'f', 'w', 'u', 'c', 'y', "'", 'k', 'v', 'b', 'z', 'p', 'x', ';',
       'j', 'q', '`', '?', '-', ':', '!', '2', '8', '(', ')', '&', '3',
       '0', '1', '9', '5', '$', '6', '7', '4', '%', '[', ']', '/', '*',
       '+', '{', '}'], dtype=object)
```

```python
vowels = ['a','e','i','o','u']
#numbers = ['0','1','2','3','4','5','6','7','8','9']
punctuations = [";","'",'?',':','!',';','"','.',',']

symbols = ['$','[',']','/','%','$','&','(',')','-','_','`','#','@','^','*','+','=','}','{',
        '>','<','~',"\\","|"]
```

Other preprocessing helper functions/ Please, read through to understand what the code does:
```python
# Function shifts the columns by a number of step so we can align any character with the character(s) that occur before and after it.
def extract_context(dataframe,column='Characters',name='',n_gram=1,fill_value= 'UNK'):
  for i in range(1,n_gram+1):
    dataframe[f'pre-{i} {name}'] = dataframe[column].shift(i, fill_value= fill_value) # shift down
    dataframe[f'post-{i} {name}'] = dataframe[column].shift(-i, fill_value=fill_value)# shift up
  return dataframe


# Function typify characters into 'alpha', 'dig', 'sym' for letters, numbers, symbols respectively.
def get_type(char):
  if char.isalpha():
    return 'a'
  elif char.isdigit():
    return 'd'
  else:
    return 's'

# Function classifies character into 'vow','con','dig','punc','sign' for vowel, consonant, number, punctuation and sign respectively
def get_class(char):
  if char.isdigit():
    return 'd'
  elif char in vowels:
    return 'v'

  elif char in punctuations:
    return 'p'
  elif char in symbols:
    return 's'
  else:
    return 'c'# character here has to be a consonant.

def get_congruity(dataframe):
  # This function checks the character before and after the index character to see if they belong to the
  # same type,class.
  dataframe['is_congruos_type'] = dataframe['pre-1 type'] == dataframe['post-1 type']
  dataframe['is_congrous_class'] = dataframe['pre-1 class'] == dataframe['post-1 type']
  return dataframe
```
Now, let's process these characters.

```python
train = extract_context(train_data.copy(), n_gram=2)
train.head()
```
<img src="images/Text segmentation/part1/word_segmentation_Table 2.png" alt="Table 2">


```python
train.tail(10)
```
<img src="images/Text segmentation/part1/word_segmentation_Table 3.png" alt="Table 3">


Next, we calculate and extract the frequency, prob of the index character, transitional probabilities with neighbouring characters, frequency and probability of whole sequence in the whole sequence. We can consider 3 character sequences and 5 character sequences.

First, we concatenate the sequences appropriately into new columns before computing the frequency of each. For sequences, we can consider 3 types of sequences:

Letter sequences (e.g 't','h','e' as sequence 'the')

class sequences (e.g 't','h','e' as sequence 'consonant-consonant-vowel')

type sequences (e.g 't', 'h','e' as sequence 'letter-letter-letter')

The most important is the first!

Let's write code for this (I'll leave exhaustive comments)
```python
char_properties = ['Characters','class','type'] # Properties of each character
def get_order(index_column='Characters',append_name='', n_gram=1):
  # Function gets the correct order of the characters that appear before and after the index character.
  order = [f'pre-{i} {append_name}' for i in range(1,n_gram+1)] + [index_column] +[f'post-{i} {append_name}' for i in range(1,n_gram+1)]
  return order


# This function generates all possible combinations (sequences) between index character and
# character(s) before and after.
# Note: This function is symmetric in action and was built to only consider exactly 1 or 2 characters
# before and after the index character.
# I did this to reduce the computational complexity. I also only used exactly 2 characters before and after
# the index character for this training.
def process_seq_columns(dataframe,column='Characters',append_name='',n_gram=1):
  order = get_order(column, append_name, n_gram)
  #print(order)
  if n_gram == 1: # If we are using just 1 character before and after, shift up by 1 step and down by 1 step
    dataframe[f'pre-1 seq {append_name}'] = dataframe[order[0]] + dataframe[order[1]]
    dataframe[f'post-1 seq {append_name}'] = dataframe[order[1]] + dataframe[order[2]]
    dataframe[f'whole-seq {append_name}'] = dataframe[order[0]] + dataframe[order[1]] + dataframe[order[2]]

  else: # Else shift 2 steps up and then down.
    dataframe[f'pre-1 seq {append_name}'] = dataframe[order[0]] + dataframe[order[2]]
    dataframe[f'pre-2 seq {append_name}'] = dataframe[order[1]] + dataframe[order[0]] + dataframe[order[2]]

    dataframe[f'post-1 seq {append_name}'] = dataframe[order[2]] + dataframe[order[3]]
    dataframe[f'post-2 seq {append_name}'] = dataframe[order[2]] + dataframe[order[3]] + dataframe[order[4]]
    dataframe[f'whole-seq {append_name}'] = dataframe[order[1]] + dataframe[order[0]] + dataframe[order[2]] + \
                            dataframe[order[3]] + dataframe[order[4]]

  return dataframe

# Function calculates the frequency and probability of occurrence of any given column
# It check the frequency of a character or sequence in the dataset and divides it by the number of samples in dataset or number of unique sequences in corpus.
# e.g how often the sequence 'to' occurs in the dataset given the number of unique 2 letter sequences in the dataset

def cal_freq_prob(dataframe,column, index=False,return_freq=False, seq=False):
  # group data by the unique values of a column and count their occurrence
  freq = dict(dataframe.groupby(column)[column].count())
  #print('length of freq keys: ', len(freq.keys()))
  if return_freq: # return frequency
    #print(column + '-freq')
    dataframe[column +'-freq'] = dataframe[column].apply(lambda x: int(freq[x]))


  num_of_samples = len(dataframe) # number of samples in dataframe

  if index:  # if its the 'Characters' column the find the percentage of occurence.
    dataframe[column + '-prob'] = dataframe[column].apply(lambda x : int(freq[x])/num_of_samples)
    return dataframe

  if seq: # if column represents a sequence then get the probability of sequence and occurence/number of unique (n_gram)sequence.
    # get the number of unique sequences after grouping by sequence column above
    num_of_seq = len(freq)
    #print(total)

    # Calculate both entities discussed above.
    dataframe[column + '-samprob'] = dataframe[column].apply(lambda x: int(freq[x])/ num_of_seq)
    dataframe[column + '-prob'] = dataframe[column].apply(lambda x : int(freq[x])/num_of_samples)


  return dataframe


# This function calculates the transitional probability using the formula as discussed above
# Function was initially for the words segmentation model but I later generalised the function so it can be used for the sentence  as well.
# given a sequence xyxf, we calculate the trans. prob. for y being the next letter given x, for x being the next letter given 'xy'
# and then for 'f' being the next letter given 'xyx'.
# All the columns ending with -freq suffix and pre/post prefix have already being automatically generated by other functions during
# processing.
# But they just create various sequence of diff. length given the n_gram: e.g for 'xyzf' we have : 'xy','xyz','xyzf' sequences.
def cal_transitional_prob(dataframe,index='Characters',n_gram=1):
  # if we are dealing with the 'Characters' column:
  if index == 'Characters':
    if n_gram == 1 : # if we consider just one character both before and after index character:
      dataframe['trans-pre'] = dataframe['pre-1 seq -freq'].astype(int)/dataframe[index+'-freq'].astype(int)
      dataframe['trans-post'] = dataframe['post-1 seq -freq'].astype(int)/ dataframe[index+'-freq'].astype(int)

    else: # if we consider more than one character
      for i in range(1, n_gram+1):
        if i == 1:
          dataframe['trans-pre-1'] = dataframe['pre-1 seq -freq'].astype(int)/dataframe[index+'-freq'].astype(int)
          dataframe['trans-post-1'] = dataframe['post-1 seq -freq'].astype(int)/ dataframe[index+'-freq'].astype(int)
        else:
          dataframe[f'trans-pre-{i}'] = dataframe[f'pre-{i} seq -freq'].astype(int)/dataframe[f'pre-{i-1} seq -freq'].astype(int)
          dataframe[f'trans-post-{i}'] = dataframe[f'post-{i} seq -freq'].astype(int)/ dataframe[f'pre-{i-1} seq -freq'].astype(int)

    return dataframe

  else: # If we are not dealing with 'Characters' column then do this below. Its basically the same code as that above.
    if n_gram == 1:
      dataframe[f'trans-pre {index}'] = dataframe[f'pre-1 seq {index}-freq'].astype(int)/dataframe[index+'-freq'].astype(int)
      dataframe[f'trans-post {index}'] = dataframe[f'post-1 seq {index}-freq'].astype(int)/ dataframe[index+'-freq'].astype(int)

    else:
      for i in range(1,n_gram+1):
        if i == 1: # if we are dealing with sequences that involve the closest set of characters to the index character.
        # for example: in 'xyxf', the closest characters to 'x' (3rd one) are 'y' and 'f'
          dataframe[f'trans-pre-{i} {index}'] = dataframe[f'pre-{i} seq {index}-freq'].astype(int)/dataframe[index+'-freq'].astype(int)
          dataframe[f'trans-post-{i} {index}'] = dataframe[f'post-{i} seq {index}-freq'].astype(int)/ dataframe[index+'-freq'].astype(int)
        else: # Else, consider the distant characters too
          dataframe[f'trans-pre-{i} {index}'] = dataframe[f'pre-{i} seq {index}-freq'].astype(int)/dataframe[f'pre-{i-1} seq {index}-freq'].astype(int)
          dataframe[f'trans-post-{i} {index}'] = dataframe[f'post-{i} seq {index}-freq'].astype(int)/ dataframe[f'pre-{i-1} seq {index}-freq'].astype(int)


  return dataframe
```
Now, we will make use of all the functions (put them all in one function) above to process the data and engineer our features in this format:

- Extract and process characters (remove '.',',' and uppercase etc)

- Extract properties of each features (e.g. alphabet, digit or symbol etc)

- calculate the frequency of each character in the corpus.

- Extract context : shift each character (a new column) by a number of steps so we can extract the character(s) that come before and after it.

- Calculate the frequency of occurence of the each character with the character(s) that come before and after it in the corpus.

- Check for congruity: If the characters before index character have the same type with the character after index character, return True otherwise, return false.

- Calculate the probability of the sequence being a valid subword from the corpus.

- Calculate the transitional probability of the index character to the next and previous characters. We do this for the characters.

- Encode columns with class labels (one-hot encoding).


After this processing, we will train with an MLP model for convenience. 


```python
# Function drops any set of columns we do not want to use for our task after we have computed and extracted the relevant features
def drop_column(dataframe, pattern): # pattern here represents a column name in dataset
    if type(pattern) == list: # if given a list of columns, do this: delete all.
      for i in pattern:
        for col in dataframe.columns:
          if col.endswith(i):
            dataframe = dataframe.drop(columns=[col])
    else: # else, delete the one column given
      for col in dataframe.columns:
        if col.endswith(pattern):
          dataframe = dataframe.drop(columns=[pattern])
    return dataframe

# Now, the main function that ties the whole task together: extract all features of index characters and its context.
def process_char_dataset(dataframe,char_fill= 'UNK', class_type_fill = 'UNK',n_gram=1):
  # block of code extractes properties of each character:
  dataframe['class'] = dataframe['Characters'].apply(get_class)
  dataframe['type'] = dataframe['Characters'].apply(get_type)
  dataframe = cal_freq_prob(dataframe, 'Characters',index=True,return_freq=True)

  # block of code extract context and its features
  dataframe = extract_context(dataframe, n_gram=n_gram,fill_value=char_fill)
  dataframe = extract_context(dataframe,'class','class',n_gram=n_gram,fill_value=class_type_fill)
  dataframe = extract_context(dataframe,'type','type',n_gram=n_gram, fill_value=class_type_fill)
  dataframe = get_congruity(dataframe)


  # block of code generates sequences by concatenating the index character to the 'before' and 'after' character(s)
  # block of code also calculates other properties : transitional prob, prob. of sequence etc

  dataframe = process_seq_columns(dataframe, n_gram= n_gram) # Create pre, post and whole sequences

  # Calculate frequency, prob of sequence and transitional probability for each character.
  for i in range(1, n_gram +1):
    #print('yes')
    dataframe = cal_freq_prob(dataframe, f'pre-{i} seq ',return_freq=True,seq=True) #for presequence
    dataframe = cal_freq_prob(dataframe, f'post-{i} seq ',return_freq=True,seq=True) # for post sequence

  dataframe = cal_freq_prob(dataframe,f'whole-seq ',return_freq=True,seq=True) # for whole sequence with index in middle

  # dataframe = cal_freq_prob(dataframe, 'pre-1 seq ',return_freq=True,seq=True) #for presequence
  # dataframe = cal_freq_prob(dataframe, 'post-1 seq ',return_freq=True,seq=True) # for post sequence
  # dataframe = cal_freq_prob(dataframe, 'whole-seq ',return_freq=True,seq=True) # for whole sequence with index in middle

  # calculate the transitional probabilities of all sequences/character into the character that immediately follows it.
  dataframe = cal_transitional_prob(dataframe, 'Characters' ,n_gram)

  # Now we drop all the columns we dont need after we have extracted and calculated the ones we need.

  drop = []
  for col in dataframe.columns :
    if (col.startswith('pre') and col.endswith('seq ')) or (col.startswith('post') and col.endswith('seq ')):
      drop.append(col)
  drop.append('whole-seq ')


  dataframe = drop_column(dataframe,drop)

  return dataframe
```

Next, we preprocess the data

```python
train = process_char_dataset(train,n_gram=2)
```

Let's look at the final data

```python
train.head()
```
<img src="images/Text segmentation/part1/word_segmentation_Table 4.png" alt="Table 4">


Next, we create our preprocessing transformers that will encode our categorical columns and scale our real valued columms. We will use the MinMax scaler for the real valued columns and the ordinal + one hot encoder for the categorical columns. Scaling is very important for deep learning frameworks as it gives a stable performance.


```python
# We store all categorical features in a list so we can apply the right encoding to them.
categorical_columns = ['Characters', 'class', 'type', 'pre-1 ', 'post-1 ','pre-2 ','post-2 ',
       'pre-1 class', 'post-1 class', 'pre-2 class','post-2 class','pre-1 type', 'post-1 type',
       'pre-2 type', 'post-2 type','is_congruos_type', 'is_congrous_class']

# Define categories that will be used by the Ordinal encoder.
# We add an extra character ',' to the unique values in the 'Characters' column so that
# 'Characters','pre-1','post-1' columns have the same kind of one-hot encoding representation.
# Therefore, we want both the 'Characters','type' and 'class' columns to have the same representation
# as their counterparts.
# It takes a dictionary of the colum to characters to append to represent the null_values
# So we define this dictionary before the function.
# I make this functionalities as functions so they can be reused downstream if needed.
append_chars = {'Characters': 'UNK',
                'class': 'UNK',
                'type': 'UNK'}

# Function as described above.
def get_categories(dataframe,append_chars):
  categories = []
  for cat in categorical_columns:
    if cat in append_chars.keys():
      values = list(dataframe[cat].unique())
      values.append(append_chars[cat])
      categories.append(values)
    else:
      categories.append(list(dataframe[cat].unique()))
  return categories


# To preprocess the data, we need to process the categorical features and continuous features in different ways
# so we write two classes that extract the categorical columns and drop the categorical columns so they can be processed
# differently by the different transformers.

class ColumnExtractor(BaseEstimator, TransformerMixin):
  def __init__(self,col_extract):
    super(ColumnExtractor, self).__init__()
    self.col_extract = col_extract

  def fit(self, X, y=None):
    return self

  def transform(self,X):
    return X[self.col_extract]

class ColumnDropper(BaseEstimator,TransformerMixin):
  def __init__(self,col_drop):
    super(ColumnDropper, self).__init__()
    self.col_drop = col_drop

  def fit(self, X, y=None):
    return self

  def transform(self,X):
    Xt = X.drop(columns=self.col_drop)
    return Xt

# function saves models to file.
def save_to_file(obj,file_path):
  with open(file_path,'wb') as f:
    pickle.dump(obj,f)
  return

# function loads saved models from file.
def load(file_path):
  with open(file_path, 'rb') as f:
    obj = pickle.load(f)
  return obj

# Function preprocesses and transforms data into sth acceptable by our deep learning model.
# we then store the fitted transformer to our drive so it can always be used anytime we want to segment raw text.
def preprocess_data(dataframe,cat_columns,cat_values=None,file_path=FILE_PATH,pipe_name=None,save=True):
  # define custom Transformers
  cat_transformer = ColumnExtractor(cat_columns) # extract categorical columns
  cont_transformer = ColumnDropper(cat_columns) # drop cat columns

  if cat_values:
    # define Scaler transformers appropriately
    scaler_pipeline = Pipeline([('transformer', cont_transformer),('scaler',MinMaxScaler())])
    encoder_pipeline = Pipeline([('transformer', cat_transformer),
                    ('label_encoder',OrdinalEncoder(categories=cat_values)),('one_hot',OneHotEncoder())])
  else:
    # define Scaler transformers appropriately
    scaler_pipeline = Pipeline([('transformer', cont_transformer),('scaler',MinMaxScaler())])
    encoder_pipeline = Pipeline([('transformer', cat_transformer),
                    ('label_encoder',OrdinalEncoder()),('one_hot',OneHotEncoder())])

  # create a pipeline that transforms the data simultaeneously
  feature_pipeline = FeatureUnion([('scaler_p',scaler_pipeline),('encoder_p',encoder_pipeline)])
  train_x = feature_pipeline.fit_transform(dataframe)

  # save transformer to file.
  if save and pipe_name:
    file_path = os.path.join(file_path,pipe_name)
    save_to_file(feature_pipeline,file_path)

  if save and (not pipe_name):
    file_path = os.path.join(file_path, 'demo')
    save_to_file(feature_pipeline, file_path)

  return train_x , feature_pipeline

```

```python
categories = get_categories(train,append_chars)
train , transformer = preprocess_data(train, categorical_columns,                       categories, pipe_name='char_pipeline_transformer')
train
```

```
<3088940x352 sparse matrix of type '<class 'numpy.float64'>'
	with 116829993 stored elements in Compressed Sparse Row format>
```

We have a sparse matrix because of the one hot encoding of a categorical variables with a lot of unique values.

```python
# We convert the target data into categorical data( one hot encoding) for training our deep learning model
target_data = tf.keras.utils.to_categorical(target_data)

# we split data into train and test set (0.8:0.2)
X_train, X_test , y_train, y_test = train_test_split(train, target_data,test_size=0.2,random_state=30, shuffle=False)

X_train, X_test
```

```
(<2471152x352 sparse matrix of type '<class 'numpy.float64'>'
 	with 93445997 stored elements in Compressed Sparse Row format>,
 <617788x352 sparse matrix of type '<class 'numpy.float64'>'
 	with 23383996 stored elements in Compressed Sparse Row format>)
```

## Train Model
```python

# define model, set batch size and number of epochs for training.
batch_size = 512
epochs = 10

input_dim = train.shape[1] # set the input dimensions for the neural network
output_dim = target_data.shape[1] # set the output_dim for the neural network

# This function creates a full model.
def create_model(input_dim,first_layer_units=500, hidden_layer_units=700,output_units=4):
  model = Sequential()
  model.add(Dense(first_layer_units,input_dim=input_dim ,activation='relu', kernel_initializer='he_uniform'))

  model.add(Dense(hidden_layer_units, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dense(hidden_layer_units, activation='relu', kernel_initializer='he_uniform'))

  model.add(Dense(output_units, activation='softmax'))
  model.compile(loss= tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])
  #model.summary()

  return model

model = create_model(input_dim,1000,700,output_dim)
model.summary()
```

``
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense (Dense)               (None, 1000)              353000    
                                                                 
 dense_1 (Dense)             (None, 700)               700700    
                                                                 
 dense_2 (Dense)             (None, 700)               490700    
                                                                 
 dense_3 (Dense)             (None, 4)                 2804      
                                                                 
=================================================================
Total params: 1547204 (5.90 MB)
Trainable params: 1547204 (5.90 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```


```python
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

# Schedule learning rate when you hit a plateau. The learning rate drops to a smaller value when performance hits a plateau.
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_delta=1E-7)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test, y_test),callbacks=[es,lr])

```
```
Epoch 1/10
4827/4827 [==============================] - 77s 14ms/step - loss: 0.2218 - accuracy: 0.9163 - val_loss: 0.1808 - val_accuracy: 0.9346 - lr: 0.0010
Epoch 2/10
4827/4827 [==============================] - 63s 12ms/step - loss: 0.1707 - accuracy: 0.9364 - val_loss: 0.1738 - val_accuracy: 0.9378 - lr: 0.0010
Epoch 3/10
4827/4827 [==============================] - 58s 11ms/step - loss: 0.1563 - accuracy: 0.9415 - val_loss: 0.1636 - val_accuracy: 0.9412 - lr: 0.0010
Epoch 4/10
4827/4827 [==============================] - 58s 11ms/step - loss: 0.1473 - accuracy: 0.9445 - val_loss: 0.1614 - val_accuracy: 0.9433 - lr: 0.0010
Epoch 5/10
4827/4827 [==============================] - 60s 12ms/step - loss: 0.1408 - accuracy: 0.9466 - val_loss: 0.1625 - val_accuracy: 0.9435 - lr: 0.0010
Epoch 6/10
4827/4827 [==============================] - 58s 11ms/step - loss: 0.1358 - accuracy: 0.9482 - val_loss: 0.1661 - val_accuracy: 0.9437 - lr: 0.0010
Epoch 7/10
4827/4827 [==============================] - 61s 11ms/step - loss: 0.1321 - accuracy: 0.9494 - val_loss: 0.1649 - val_accuracy: 0.9442 - lr: 0.0010
Epoch 8/10
4827/4827 [==============================] - 58s 11ms/step - loss: 0.1286 - accuracy: 0.9503 - val_loss: 0.1674 - val_accuracy: 0.9444 - lr: 0.0010
Epoch 9/10
4827/4827 [==============================] - 63s 12ms/step - loss: 0.1262 - accuracy: 0.9511 - val_loss: 0.1701 - val_accuracy: 0.9446 - lr: 0.0010
Epoch 9: early stopping
```

```python
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

<img src="images/Text segmentation/word_segment_train.png" alt="Training history">

Here, we see that the model performed very well and generalized pretty well to the data. However, its most likely that the performance of the model will differ based on the length of the full text it was given because of the features that were manually extracted for the model that took length (number of samples) into account.

Let's save the model to file memory.

```python
model_name = 'word_segmentor'
model_path = os.path.join(FILE_PATH,model_name)

model.save(model_path)
```

Whoops! That was a whole lot. We finally have a trained model for word segmentation with pretty good accuracy! Thats' all for word segmentation. To see and appreciate the output of this model, check out the <a href="/projects/Sentence Segmentation.md">Part 2 here</a>.

Thank you for your time and see you soon!!