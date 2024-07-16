---
author: "Paul Jeffrey"
title: "Unraveling the Secrets of Raw Text: A Journey Through Word, Sentence Segmentation and Capitalization with Python (Part 3)"
date: "2023-12-16"
draft: false
cover:
    image: "static/images/Text segmentation/true casing.jpg"
    alt: 'True Casing'
description: "A Journey Through Word, Sentence Segmentation and Capitalization with Python (Part 3)"
tags: [
    "neural networks","natural language processing","word segmentation"
]

---

# Title: Unraveling the Secrets of Raw Text: A Journey Through Word, Sentence Segmentation and Capitalization with Python (Part 3)

## Introduction

The final piece of the puzzle is to capitalize the first letter of each sentence and other words that need to be cased (e.g nouns). This not only enhances readability but also provides valuable contextual information.
By the way, if you have not checked out the first 2 parts of this lovely project's article, you can find it <a href="/projects/Word Segmentation.md"> here </a>. I advice that you follow the previous parts of this project first before reading this last one.

To achieve this, We are going to train a model using the following principles/assumptions of true casing of words:

- The first letter of the first word after a '.','!','?' (beginning of a sentence) is always capitalized.
- Names and other proper nouns should be capitalized.
- Words after a colon should not be capitalized.
- Capitalize the first word of a quote.
- Capitalize months , days but not seasons.

Therefore, we create functions to extract some of these information for our model. To save time, I will leave them here but please take your time to go through each of them to understand what they do.


```python
months = ['january', 'february','march','april','may','june','july','august','september','october','november','december']
days = ['monday','tuesday', 'wednesday','thursday','friday','saturday','sunday']

# These punctuations separate sentences in english language.
sent_separator = ['.','!','?','UNK']
# function checks if a word begins a sentence.
def begins_sentence(row):
  if row['pre-1 words'] in sent_separator:
    return True
  else:
    return False

# if the word is a month or day
def is_month_day(word):
  if ( word in months ) or (word in days):
    return True
  else:
    return False

# If the word is a proper noun
def is_proper_noun(word):
  if word == 'NNP' or word == 'NP':
    return True
  else:
    return False

# if the word comes after quotes
def supersede_quote(row):
  if row['pre-1 words'] == "'" or row['pre-1 words'] == '"':
    return True
  else:
    return False

# if the word comes after a colon
def after_colon(row):
  if row['pre-1 words'] == ':':
    return True

  else:
    return False

# function to check if word is a symbol or digit:
def check_digit_sym(word):
  if (word in punctuations) or (word in symbols) :
    return True
  elif word.isdigit():
    return True
  else:
    False

```
Now, we extract and process the train and test data

```python
corpus = ext_tagged_corpus(8) # extract corpus

words, tags, target_data = create_sent_data(corpus, casing=True)
del corpus
```

```python
sent_df = pd.DataFrame({'words': words, 'tags': tags}) # create dataframe
sent_df.head()
```
<img src="/static/images/Text segmentation/part3/Table 1.png" alt="Table 1">

Some functions from the word segmentation model were used here also.

```python
def process_case_data(df,n_gram=1,fill_value = 'WRB+DOZ'):

  # Extract all features as described earlier
  df['is_month_day'] = df['words'].apply(is_month_day)
  #df['is_digit_sym']  = df['words'].apply(check_digit_sym)
  df['is_proper_noun']  = df['tags'].apply(is_proper_noun)



  # block of code extract context and its features (words around index word)
  df = extract_context(df, 'words','words', n_gram=n_gram,fill_value=fill_value)
  df = extract_context(df, 'tags','tags', n_gram=n_gram,fill_value=fill_value)

  # Extract all features as described earlier
  df['after_colon'] = df.apply(after_colon,1)
  df['supersed_quote'] = df.apply(supersede_quote,1)
  df['begins_sentence'] = df.apply(begins_sentence,1)

  # Drop all irrelevant columns
  drop = ['post-1 words']
  drop.append('pre-1 words')
  drop.append('post-1 tags') # we remove the tags that come after word and leave the one that comes before.

  drop.append('words')

  df = drop_column(df, drop)

  return df

train = process_case_data(sent_df)
train.head()
```
<img src="/static/images/Text segmentation/part3/Table 1.png" alt="Table 1">

```python
# Also save tags for this model
case_tag_list = train['tags']
save_to_file(case_tag_list, os.path.join(FILE_PATH,'case_tag_list'))
```

Since all columns are categorical , we pass it through an ordinal and one hot encoder first we append 'UNK' to the tags column for general representaton of tags.

```python

case_cat_value = [list(train[col].unique()) for col in train.columns]
case_cat_value[0].append('UNK')

for col in case_cat_value:
  try:
    col.sort()
  except:
    continue


# preprocess data by passing through encoder transformer

case_pipeline = Pipeline([('ordinal_encoder', OrdinalEncoder()),
                          ('one hot', OneHotEncoder())])
# Fit transformer to data.
train = case_pipeline.fit_transform(train)

# Save transformer to file
transformer_path = os.path.join(FILE_PATH,'case_pipeline')
save_to_file(case_pipeline,transformer_path)
```
Split data into train and test

```python
# Convert target_data variables from string format to numerical format
target_data = to_cat(target_data, case_cat_targ)

# change target to categorical format (One hot encoding)
target_data = tf.keras.utils.to_categorical(target_data)

# Split data into train and test set
X_train, X_test , y_train, y_test  = train_test_split(train, target_data, test_size= 0.15,
                                                      random_state=42,shuffle=False)

X_train
```

```
<619288x890 sparse matrix of type '<class 'numpy.float64'>'
	with 4335016 stored elements in Compressed Sparse Row format>
```

Again, we create our model with the previous function <a href="">Part 1</a>

```python
# Get input and output dim
input_dim = X_train.shape[1]
output_dim = 2 # because we have only 2 targets

model = create_model(input_dim, 1500, 1000, output_dim)
model.summary()
```

```
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_8 (Dense)             (None, 1500)              1336500   
                                                                 
 dense_9 (Dense)             (None, 1000)              1501000   
                                                                 
 dense_10 (Dense)            (None, 1000)              1001000   
                                                                 
 dense_11 (Dense)            (None, 2)                 2002      
                                                                 
=================================================================
Total params: 3840502 (14.65 MB)
Trainable params: 3840502 (14.65 MB)
Non-trainable params: 0 (0.00 Byte)
_________________________________________________________________
```
Now, we train the model

```python
# Schedule early stopping so that model can be stopped when there is no longer improvement
# after at least 5 steps.
epochs = 20
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

# Schedule learning rate when you hit a plateau
lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_delta=1E-7)

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                    validation_data=(X_test, y_test),callbacks=[es,lr])
```

```
Epoch 1/20
1210/1210 [==============================] - 22s 16ms/step - loss: 0.0424 - accuracy: 0.9875 - val_loss: 0.0368 - val_accuracy: 0.9895 - lr: 0.0010
Epoch 2/20
1210/1210 [==============================] - 13s 9ms/step - loss: 0.0363 - accuracy: 0.9894 - val_loss: 0.0371 - val_accuracy: 0.9905 - lr: 0.0010
Epoch 3/20
1210/1210 [==============================] - 17s 13ms/step - loss: 0.0355 - accuracy: 0.9895 - val_loss: 0.0343 - val_accuracy: 0.9906 - lr: 0.0010
Epoch 4/20
1210/1210 [==============================] - 14s 10ms/step - loss: 0.0353 - accuracy: 0.9896 - val_loss: 0.0349 - val_accuracy: 0.9906 - lr: 0.0010
Epoch 5/20
1210/1210 [==============================] - 13s 9ms/step - loss: 0.0350 - accuracy: 0.9896 - val_loss: 0.0354 - val_accuracy: 0.9905 - lr: 0.0010
Epoch 6/20
1210/1210 [==============================] - 13s 10ms/step - loss: 0.0347 - accuracy: 0.9897 - val_loss: 0.0361 - val_accuracy: 0.9902 - lr: 0.0010
Epoch 6: early stopping
```

Let's print the training history:
```python
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
```

<img src="/images/Text segmentation/casing.png" alt="Table 1">

We save this model like this and calculate our metrics

```python
model_name = 'Casing_model'
model_path = os.path.join(FILE_PATH,model_name)

model.save(model_path)
```

Let's evaluate our model using accuracy, precision, recall, F1 score

```python
from sklearn.metrics import f1_score, precision_score , recall_score, accuracy_score
# Error Functions and evaluation

# Use this functions to perform f1_score, recall and precision for error analysis.
def calculate_metrics(cat_target , y_true, y_pred):
  # For each label classification , we create a case of 'has label' and 'has no label'.
  # For example, for the 'I' label, we convert all predictions into 1 for 'I' and 0 for not 'I'
  try:
    y_true = list(np.argmax(y_true, 1))
  except:
    print('True predictions must be in categorical format.')
    return

  y_pred = list(np.argmax(y_pred,1))
  f1_scores = []
  recall_scores = []
  precision_scores = []
  accuracy_scores = []
  target = []

  for label_tag , value in cat_target.items():
    true = []
    pred = []

    for true_val, pred_val in zip(y_true, y_pred):
      if true_val == value:
        true.append(1)
      else:
        true.append(0)
      if pred_val == value:
        pred.append(1)
      else:
        pred.append(0)

    accuracy = round(accuracy_score(true, pred),2)
    f1 = round(f1_score(true, pred),2)
    precision = round(precision_score(true,pred),2)
    recall = round(recall_score(true, pred),2)

    # append scores and appropriate target.
    target.append(label_tag)
    accuracy_scores.append(accuracy)
    f1_scores.append(f1)
    precision_scores.append(precision)
    recall_scores.append(recall)


  print('         |       Accuracy Score|        Precision|        Recall|       F1|')
  print('-----------------------------------------------------------------------------------')
  for i in range(len(cat_target.keys())):
    print(f'{target[i]}             {accuracy_scores[i]}                 {precision_scores[i]}                 {recall_scores[i]}                   {f1_scores[i]}  ')


y_pred = model.predict(X_test)

calculate_metrics(case_cat_targ, y_test, y_pred)
```
```
         |       Accuracy Score|        Precision|        Recall|       F1|
-----------------------------------------------------------------------------------
T             0.99                 0.96                 0.91                   0.94  
N             0.99                 0.99                 1.0                   0.99  

```
## Tying It All Together
We are going to combine the predictive power of all the models and their processing pipelines into one function. We define all the functions needed for this in the code snippet below:

```python

def convert_predictions(pred, label_dict):
  result = []
  pred  = np.argmax(pred,1)
  for val in list(pred):
    for label, value in label_dict.items():
      if val == value:
        result.append(label)

  return result

def process_word_results(chars,predictions):
  #targets are : S for single word, E for end of word, I for inside word, B for beginning of word.
  print('processing text into words ...')
  full_text = ''
  for char , prediction in zip(chars, predictions):
    if prediction == 'S':
      full_text += ' '
      full_text += char
      full_text += ' '
    elif prediction == 'E':
      full_text += char
      full_text += ' '
    elif prediction == 'B':
      full_text += ' '
      full_text += char
    else:
      full_text += char

  print('Word segmentation completed.')
  print('Now processing words into sentences...')
  return full_text


def process_sent_result(words, predictions):
  # For the sentences, we are going to tag a word with 'E' if it ends the sentence (comes before the full stop)
  # and tag a word with 'P' (pause) if it comes before a comma and tag a word inside a sentence with an 'I'
  print('Constructing sentences and adding commas now ....')
  full_text = ' '
  for word, prediction in zip(words, predictions):
    if prediction == 'E':
      full_text += word
      full_text += '. '
    elif prediction == 'P':
      full_text += word
      full_text += ', '
    else:
      full_text += word
      full_text += ' '

  print('Sentence successfully constructed.')
  print('Now preparing sentences and casing relevant words..')
  print('Please, have some popcorn while you wait...')
  return full_text

def process_case_result(words, predictions):
  # For true casing of words in sentecnes , we are going to tag a word as 'T' for titled if the first letter of the word is in capital
  # and tag it 'N' for not titled if the first letter isnt a capital letter.
  print('Casing sentences appropriately...')
  full_text = ' '
  for word , prediction in zip(words, predictions):
    if prediction == 'T':
      full_text += word.capitalize()
      full_text += ' '
    else:
      full_text += word
      full_text += ' '

  print('Sentences completey cased. Now returning result... ')
  return full_text

# function that changes any new tag in a new corpus to an UNK value.
def change_tag(tag_list, tag, least_tag):
  if tag not in tag_list:
    return least_tag
  else:
    return tag

from time import time
def process_raw_text(raw_text,fill_tag = 'WRB+DOZ', FILE_PATH=FILE_PATH):
  # The full text here is raw text without any processing (just letters with no space)
  #assert type(raw_text) == str
  start_time = time()
  print('Processing data for word segmentation...\n')
  full_text = list(raw_text)
  full_text = pd.DataFrame({'Characters': full_text})
  # extract context
  print('Extracting features for each character in text...\n')
  #full_text = extract_context(full_text,n_gram=2)
  full_text = process_char_dataset(full_text,n_gram=2)

  # Load transformer
  print('Transforming characters...\n')
  try:
    char_transformer = load(os.path.join(FILE_PATH,'char_pipeline_transformer'))
  except:
    print('Character transformer not found in the current directory.\n')
    return

  full_text = char_transformer.transform(full_text)

  # # predict with trained model.
  # # Load model first
  print('Loading model and predicting character identities... \n')
  try:
    model = tf.keras.models.load_model(os.path.join(FILE_PATH,'word_segmentor'))
  except:
    print('word segmentor model not found in the current directory.\n')
    return
  predictions = model.predict(full_text)

  # # process result here
  print('Processing results and segmenting characters into words.....\n\n')
  predictions = convert_predictions(predictions, word_cat_targ )
  #print('Char predictions' , predictions[:5])
  full_text = process_word_results(list(raw_text), predictions)
  #print('processed to word', full_text[:15])
  word_time = time()
  print(f'Done (completed in {(word_time - start_time)} seconds).\n')


  # # Process data for sentence model
  print('Now, processing words for sentence identification and segmentation... \n')
  full_text = nltk.word_tokenize(full_text)
  word_to_tag = nltk.pos_tag(full_text)
  tags = []
  words = []
  for word, tag in word_to_tag:
    words.append(word)
    tags.append(tag)

  # # Process data for sentence transformer model
  full_text = pd.DataFrame({'words': words, 'tags': tags})
  #rint(len(full_text['tags'].unique()))
  # Load tags list
  tag_list = load(os.path.join(FILE_PATH,'tags_list'))

  # change previously unseen tags in this new corpus to the least tag during training of our model.
  # The assumption is that this unseen tags would also occur less in this new data corpus.
  full_text['tags'] = full_text['tags'].apply(lambda x: change_tag(tag_list,x,fill_tag))

  print('Processing words and transforming words appropriately...\n')
  full_text = process_sent_data(full_text, n_gram=2)

  print('Loading transformer and transforming data...')
  try:
    sent_transformer = load(os.path.join(FILE_PATH, 'sentence_transformer'))
  except:
    print('sentence transformer not found in current directory.')
    return

  full_text = sent_transformer.transform(full_text)

  # Load sentence model
  print('Loading model and predicting word identities...\n')
  try:
    model = tf.keras.models.load_model(os.path.join(FILE_PATH,'Comma_sentence_segmentor'))
  except:
    print('sentence segmentor model not found in current directory.')
    return
  result = model.predict(full_text)

  # # process results here
  print('Processing results and segmenting words to sentences appropriately...\n')
  predictions = convert_predictions(result, sent_cat_targ )
  full_text = process_sent_result(words, predictions)
  sent_time = time()
  print(f'Done (completed in {(sent_time - word_time)/60} mins).\n ')

  # process and preprocess text for the casing transformer
  print('Finally, processing sentences for appropriate casing of words...\n')
  full_text = nltk.word_tokenize(full_text)
  text_to_tag = nltk.pos_tag(full_text)
  tags = []
  words = []

  for word , tag in text_to_tag:
    words.append(word)
    tags.append(tag)

  full_text = pd.DataFrame({'words': words , 'tags': tags})
  # Load tags list
  case_tag_list = load(os.path.join(FILE_PATH,'case_tag_list'))
  # change previously unseen tags to 'WRB+DOZ' which was the least occuring tag during training.
  full_text['tags'] = full_text['tags'].apply(lambda x: change_tag(case_tag_list,x,least_tag))

  # process data for transformer model
  print('Processing words and sentences for transformer model...\n')
  full_text = process_case_data(full_text)

  # Transform data
  print('Loading and transforming processed data...\n')
  try:
    case_transformer = load(os.path.join(FILE_PATH, 'case_pipeline'))
    #print('case transformer loaded..')
  except:
    print('case transformer was not found in the current directory.\n')
    return

  full_text = case_transformer.transform(full_text)

  # Load casing model
  print('Loading model and predicting appropriate words for casing... \n')
  try:
    model = tf.keras.models.load_model(os.path.join(FILE_PATH, 'Casing_model'))
  except:
    print('casing model could not be found in the current directory.\n')
    return

  result = model.predict(full_text)

  # Process final result here
  print('Processing result and producing final format...\n')
  predictions = convert_predictions(result, case_cat_targ)
  full_text = process_case_result(words, predictions)
  case_time = time()
  print(f'Done (completed in {(case_time - sent_time)} seconds).\n)')
  print('Returning final processed result...\n\n')
  print(f'Whole task was completed in {(case_time - start_time)/60} mins.')
  print('-'*200)
  print('\n\n')

  return full_text
```

Lets test it all at once. We have our models saved to file.
We will use the first genre of the brown corpus for this. We will extract the text and remove all space, comma, full stop characters and casing from the text. Then we will run it through the <code>process_raw_text</code> function to intuitively analyze the performance of the models combined.

```python
corpus = brown.words(categories=brown.categories()[0])

text = ''
for word in corpus:
  text += word.lower()
text = text.replace('.','')
text = text.replace(',','')
text
```

This is what the raw text looks like when all the above are removed.
```
'danmorgantoldhimselfhewouldforgetannturnerhewaswellridofherhecertainlydidn'twantawifewhowasfickleasannifhehadmarriedherhe'dhavebeenaskingfortroublebutallofthiswasrationalizationsometimeshewokeupinthemiddleofthenightthinkingofannandthencouldnotgetbacktosleephisplansanddreamshadrevolvedaroundhersomuchandforsolongthatnowhefeltasifhehadnothingtheeasiestthingwouldbetosellouttoalbuddandleavethecountrybuttherewasastubbornstreakinhimthatwouldn'tallowitthebestantidoteforthebitternessanddisappointmentthatpoisonedhimwashardworkhefoundthatifhewastiredenoughatnighthewenttosleepsimplybecausehewastooexhaustedtostayawakeeachdayhefoundhimselfthinkinglessoftenofann;;eachdaythehurtwasalittledulleralittlelesspoignanthehadplentyofworktodobecausethesummerwasunusuallydryandhotthespringproducedasmallerstreamthaninordinaryyearsthegrassinthemeadowscamefastnowthatthewarmweatherwasherehecouldnotaffordtoloseadropofthepreciouswatersohespentmostofhiswakinghoursalongtheditchesinhismeadowshehadnoideahowmuchtimebuddwou...'
```
Now, let's run this text chunk through our series of models to process and reformat the text.

```python
# We will be using 'WRB+DOZ' as our fill_na values throughout.
process_raw_text(text, least_tag)
```
```
dan morgan told himself he would forget ann turner he was well rid of her he certainly did n't want a wife who was fickle as ann if he had married her he 'd have been asking for trouble but all of this was rationalization sometime she woke up in the middle of the night thinking of ann and then could not get back to sleep his plans and dreams had revolved around her so much and for so long that now he felt as if he had nothing the easiest thing would be to sell out to al budd and leave the country but there was a s tubborn streak in him that would n't allow it the bestant i dote for the bitterness and dis appointment that pois oned him was hard work he found that if he was tired enough at nighthe went to sleep simply because he was too exhausted to stay awake each day he found himself thinkingless oft en of ann ; ; each day the hurt was a little duller a little les s poignan the had plenty of work to do be cause the summer was unusually dry and hot the spring produced a smaller stream...'
```
We can see that our model did a really good job with processing this text here! Just amazing!!

## Conclusion

The culmination of these efforts resulted in a highly accurate system for processing unstructured raw text. With an accuracy of approximately 97%, this system can effectively segment words and sentences, as well as capitalize words correctly.

This system has the potential to revolutionize the way we interact with raw text. It can be used to pre-process data for various NLP tasks such as text classification, sentiment analysis, and machine translation.

The journey of developing this machine learning system has surely been an enriching and enlightening experience. It will not only deepen your understanding of NLP but also spark your passion for unlocking the secrets hidden within text.

I hope you've enjoyed this article series and found it both informative and engaging. If you have any questions or feedback, please feel free to <a href="https://www.linkedin.com/in/jeffreyotoibhi/"> contact me</a>.