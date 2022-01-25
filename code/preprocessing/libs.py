# this file contains all the libs that are used in the main.inpy file
import pandas as pd
import numpy as np
from nltk import bigrams, ngrams
from nltk.lm import Laplace
from nltk.lm.models import InterpolatedLanguageModel
from nltk.lm.preprocessing import pad_both_ends, flatten
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import re
import nltk
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import TweetTokenizer
import preprocessor as p
import copy
from wordcloud import WordCloud
from nltk.util import ngrams
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
import contractions 
from nltk.lm import MLE, Laplace, WittenBellInterpolated, KneserNeyInterpolated
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, roc_curve, auc, RocCurveDisplay
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.lm import Vocabulary