import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer

data = pd.read_csv("datasets/dataset.csv", encoding = "ISO-8859–1")
Q = pd.read_pickle("models/Q_matrix.pkl")
user_df = data[['user_id','text']]
user_df = user_df.groupby('user_id').agg({'text': ' '.join})
user_feature_object = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=500)
user_feature = user_feature_object.fit_transform(user_df['text'])

def clean_reviews(text):
    text = text.translate(string.punctuation)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    text = " ".join(text)
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)    
    return text

def get_predictions(input_user_id):
	inp_text = pd.DataFrame([data[data['user_id']==input_user_id]['text'].values[0]], columns = ['text'])
	inp_text['text'] = inp_text['text'].apply(clean_reviews)
	test_feature = user_feature_object.transform(inp_text['text'])
	test_P = pd.DataFrame(test_feature.toarray(), index=inp_text.index, 
		                     columns=user_feature_object.get_feature_names())

	predict=pd.DataFrame(np.dot(test_P.loc[0],Q.T),index=Q.index,columns=['Ratings'])
	recomd =pd.DataFrame.sort_values(predict,['Ratings'],ascending=[0])[:5]
	return recomd
	
print(get_predictions('bHXujstlLp-QuNr72Meprw'))


