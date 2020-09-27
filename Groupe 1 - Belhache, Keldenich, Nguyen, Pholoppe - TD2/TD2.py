from pandas import DataFrame
import pandas as pd
import os
import string
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
from collections import Counter
from sklearn.model_selection import train_test_split #to split train set and test set

# Function : transform a list of sub-lists to a list with all elements of the sublists
# Parameter : list_to_convert, list of lists
# Returns : list, with all the elements of the sublists
def sublist_to_list(list_to_convert):
  a = list(list_to_convert)
  flat_list = [item for sublist in a for item in sublist]
  return flat_list

# Function : get the name of a file according to its path
# Parameter : path, a string
# Returns : string, name of the file
def get_file_name(path):
    return os.path.basename(path).split(".")[0].strip().lower()

# Function : to preprocess words before the wordcloud
# Parameter : textParam, text
# Returns : list of the most meaningful words (preprocessed)
def preProcessing(textParam):
  items = textParam.lower() #transform to lower case
  items = items.translate(rmv_punct) #remove punctuation
  items = re.sub(r'\d+','',items) #remove numbers
  tokens = word_tokenize(items)
  result = [i for i in tokens if not i in stopwords] # list of the words in the text without stop words
  return (result)

# Function : calculates the frequency of a word in the spam words dictionnary
# Parameter : word, a string
# Returns : a float, >0 if the word exists in the dictionnary, -1 otherwise
def freq_one_spam(word):
  if (word in dict_spam_train_set):
    f_word_spam = dict_spam_train_set[word]
    return f_word_spam
  else:
    return (-1)

# Function : calculates the frequency of a word in the ham words dictionnary
# Parameter : word, a string
# Returns : a float, >0 if the word exists in the dictionnary, -1 otherwise
def freq_one_ham(word):
  if (word in dict_ham_train_set):
    f_word_ham = dict_ham_train_set[word]
    return f_word_ham
  else:
    return (-1)

# Function : calculates the probability of a given word
# Parameter : word, a string
# Returns : float, probability of a word by spam or ham
def bayes_one_word(word):
  if (word in dict_train_set):
    f_word_total = dict_train_set[word]

    #if the word only exists in the dictionnary of ham words
    if (freq_one_spam(word)==-1):
      return (0.7)
    #if the word only exists in the dictionnary of spam words
    elif (freq_one_ham(word)==-1):
      return 1
    #if the word exists in both ham and spam dictionnaries
    else:
      p_word_ham = freq_one_ham(word)/f_word_total
      p_word_spam = freq_one_spam(word)/f_word_total
      return (p_word_spam/p_word_ham)
  #if the word has never been in any dictionary
  else:
    return 1

# Function : applies the bayes formula to a given list of words, decides if the message is spam or ham
# Parameter : list_words, a list of string
# Returns : a string, "spam" if Bayes formula returns a value >=1, "ham" otherwise
def bayes_sentence(list_words):
  A = 1
  for word in list_words:
    A = A*bayes_one_word(word) #if A>1, the message is considered a spam
  A = A*(p_spam/p_ham)
  return (spam_or_ham(A))

# Function : returns a string depending on value in parameter
# Parameter : A, a float
# Returns : a string, "spam" if the parameter is >=1, "ham" otherwise
def spam_or_ham(A):
  if A>=1:
    return ("spam")
  else:
    return ("ham")

# Function : returns the accuracy, the precision and the recall of the predicted test set and the original test set
# Parameters : y_predict, list of the values predicted. y_test, list of real values
def performance(y_predict, y_test):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_predict)):
        if y_test.iloc[i] == y_predict[i]:
            if y_predict[i] == 'spam':
                TP = TP+1
            else:
                TN = TN+1
        else:
            if y_predict[i] == 'spam':
                FP = FP+1
            else:
                FN = FP+1

    accuracy = (TP+TN)/(TP+FP+TN+FN)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    F1_Score = (2*recall*precision)/(recall+precision)
    print("F1 Score: ",F1_Score)

name = get_file_name('SMSSpamCollection')
with open('SMSSpamCollection') as f:
    list_type = []
    list_message = []
    df = DataFrame(columns=['type','message'])
    i=0
    for line in f:
        data = line.split("\t")
        type = data[0]
        message = data[1]
        list_type.append(type)
        list_message.append(message)
        i=i+1


stopwords = set(stopwords.words('english'))
stopwords.update(['u','n','k','e','per','c','dont','r','im','yup','thk','cos','even',"im'",'f','got','ü','ok','ive','youre','pls','huh','okie','nite','ill',':)', ':', ','])
rmv_punct = str.maketrans('','',string.punctuation) #setting of the punctuation removal


list_clean = []
for i in range(len(list_message)):
  list_clean.append(preProcessing(list_message[i]))

df = pd.DataFrame(list(zip(list_type, list_clean)), columns =['Type', 'Message'])

#split data set 70% train 30% test with random index
X_train, X_test, y_train, y_test= train_test_split(df.iloc[:,1], df.iloc[:,0], test_size=0.3,random_state=109)

# creating a DataFrame with the SMS
df_train_split = pd.DataFrame(list(zip(y_train, X_train)), columns =['Type', 'Message'])
df_test_split = pd.DataFrame(list(zip(y_test, X_test)), columns =['Type', 'Message'])

# splitting the DataFrame into two dataframes according to their type, spam or ham
Spam_train = df_train_split.loc[df_train_split['Type'] == 'spam', 'Message']
Ham_train = df_train_split.loc[df_train_split['Type'] == 'ham', 'Message']
Spam_train_list = sublist_to_list(Spam_train) #transform the list of lists to a simple list with all elements
Ham_train_list = sublist_to_list(Ham_train)
words_train_set = sublist_to_list(X_train)

Spam_test = df_test_split.loc[df_test_split['Type'] == 'spam', 'Message']
Ham_test = df_test_split.loc[df_test_split['Type'] == 'ham', 'Message']
Spam_test_list = sublist_to_list(Spam_test)
Ham_test_list = sublist_to_list(Ham_test)
words_test_set = sublist_to_list(X_test)

p_spam = len(Spam_train)
p_ham = len(Ham_train)
p_spam_test = len(Spam_test)
p_ham_test = len(Ham_test)

dict_train_set = Counter(words_train_set) #creation of a dictionnary of all words
dict_spam_train_set = Counter(Spam_train_list) # creation of a dictionnary with words in spam sms only
dict_ham_train_set = Counter(Ham_train_list) #creation of a dictionnary with words in ham sms only

spamS = 0
hamS = 0
spamH = 0
hamH = 0

for i in range (0,len(Spam_test)):
  x32 = Spam_test.iloc[i]
  if (bayes_sentence(x32)=="spam"):
    spamS +=1
  else:
    hamS+=1

for i in range (0,len(Ham_test)):
  x33 = Ham_train.iloc[i]
  if (bayes_sentence(x33)=="spam"):
    spamH+=1
  else:
    hamH+=1

#calcul de la performance
y_predict = []
for i in range(len(y_test)):
    y_predict.append(bayes_sentence(X_test.iloc[i]))


print("\n - For a total of "+str(len(Ham_test))+" ham SMS in the test set")
print("TP : "+str(hamH)+'/'+str(len(Ham_test)))
print("FP : "+str(spamH)+'/'+str(len(Ham_test)))
print("The error percentage is : "+str(spamH/len(Ham_test)*100))


print("\n - For a total of "+str(len(Spam_test))+" spam SMS in the test set")
print("TN : "+str(spamS)+'/'+str(len(Spam_test)))
print("FN : "+str(hamS)+'/'+str(len(Spam_test)))
print("The error percentage is : "+str(hamS/len(Spam_test)*100)+"\n")

performance(y_predict, y_test)
