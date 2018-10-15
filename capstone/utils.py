import re
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer as Vec
from sklearn.decomposition import LatentDirichletAllocation as LDA
import numpy as np
import pandas as pd
import string
import nltk
import datetime
import time
import json
from matplotlib import pyplot as plt

<<<<<<< HEAD
trumplab = '/run/media/derekgm@byu.local/FAMHIST/Data/final_project/trump.txt'
clintonlab = '/run/media/derekgm@byu.local/FAMHIST/Data/final_project/clinton.txt'
trumpmint = '/media/derek/FAMHIST/Data/final_project/trump.txt'
clintonmint = '/media/derek/FAMHIST/Data/final_project/clinton.txt'
tmacbook = '/Volumes/FAMHIST/Data/final_project/trump.txt'
cmacbook = '/Volumes/FAMHIST/Data/final_project/clinton.txt'

def get_file():
    print("""\n\tOptions\n
            1: trump from lab computer\n
            2: trump from linux mint\n
            3: clinton from lab computer\n
            4: clinton from linux mint\n
            5: trump from macbook air\n
            6: clinton from macbook air\n\n""")
=======
trumplab = '/run/media/derekgm@byu.local/FAMHIST/Data/final_project/trump2.txt'
clintonlab = '/run/media/derekgm@byu.local/FAMHIST/Data/final_project/maga.txt'
trumpmint = '/media/derek/FAMHIST/Data/final_project/trump2.txt'
clintonmint = '/media/derek/FAMHIST/Data/final_project/maga.txt'

def get_file():
    print("""\n\tOptions\n
            1: trump2 from lab computer\n
            2: trump2 from linux mint\n
            3: maga from lab computer\n
            4: maga from linux mint\n\n""")
>>>>>>> 707a6586ac341dbe50de0e23ff7c877427a826cc
    name = raw_input("Enter number >> ")
    if name == "1":
        return trumplab
    elif name == "2":
        return trumpmint
    elif name == "3":
        return clintonlab
    elif name == "4":
        return clintonmint
    elif name == "5":
        return tmacbook
    elif name == "6":
        return cmacbook
    else:
        print "invalid input"

class TwitterCorpus(object):
    
    def __init__(self,filename,n=None,m=None):
        """
        ATTRIBUTES:
            data (string) the tweet data from a txt file
            tweets (list) of user tweet content
            user_stats (list) of number of followers, friends, and user tweets to date
            timestamps (list) of floats indicating UTC timestamp
            time (list) of timestamps converted to datetime objects
            n_mentions (list) of the number of mentions in a tweet
            n_hashtags (list) of the number of hashtags in a tweet
            n_weblinks (list) of the number of external links in a tweet
            retweets (list) of booleans indicating whether the tweet was a retweet
        """
        print("Loading file...\n")
        start = time.time()
        self.data = open(filename,'r').readlines()[n:m]
        self.json_data = []
        self.tweets = []
        self.user_stats = []
        self.timestamps = []
        self.time = []
        err = 0
        for i,line in enumerate(self.data):
            line = json.loads(line)
            self.json_data.append(line)
            if i == 0:
                print line.keys()
            # get everything except for the tweet
            try:
                # number of followers, statuses, and friends
                self.user_stats.append([line['user']['screen_name'],line['user']['description']])
                # time that the tweet was sent
                self.timestamps.append(int(line['timestamp_ms']))
                # content of the tweet
                self.tweets.append(line['text'])
            except:
                #print i
                err += 1
        print "Errors: " + str(err)
        # convert to numpy array
        self.timestamps = np.array(self.timestamps)
        self.user_stats = np.array(self.user_stats)
        self.n_mentions = []
        self.n_hashtags = []
        self.n_weblinks = []
        self.retweets = []
        end = time.time()
        print("Time: %s" % (end-start))
        
    def clean_text(self,remove_vars_from_tweet=True):
        """
        Cleans the text and extracts information from the tweet
        """
        print("Cleaning text...")
        start = time.time()
        tweetwords = []
        u_h = []
        u_m = []
        for s in self.tweets:
            m_str = ""
            h_str = ""
            s = s.replace('"""','')
            s = s.replace("'","")
            s = s.lower()
            mentions = re.findall(r'@\w*',s)
            hashtags = re.findall(r'#\w*',s)
            weblinks = re.findall(r'http\S*',s)
            retweets = re.findall('^rt ',s)
            numbers = re.findall(r'[0-9]+',s)
            self.n_mentions.append(len(mentions))
            self.n_hashtags.append(len(hashtags))
            self.n_weblinks.append(len(weblinks))
            self.retweets.append(len(retweets))
            for m in mentions:
                u_m.append(m)
            for h in hashtags:
                u_h.append(h)
            if remove_vars_from_tweet:
                for m in mentions:
                    s = s.replace(m,'')
                for h in hashtags:
                    s = s.replace(h,'')
                for w in weblinks:
                    s = s.replace(w,'')
                for r in retweets:
                    s = s.replace(r,'')
                for n in numbers:
                    s = s.replace(n,'')
            tweetwords.append(s)
        self.mentions = u_m
        self.hashtags = u_h
        self.u_mentions = np.unique(u_m)
        self.u_hashtags = np.unique(u_h)
        self.tweets = tweetwords
        end = time.time()
        print("Time: %s" % (end-start))
        
    def remove_keywords(self, keywords):
        """
        Remove the specified keywords from the list. Updates self.tweets
        INPUT
            keywords (list) of keywords to remove from tweets
        """
        print("Removing keywords...")
        start = time.time()
        new_tweets = []
        for t in self.tweets:
            for k in keywords:
                t = t.lower().replace(k.lower(),"")
            new_tweets.append(t)
        self.tweets = new_tweets
        print("Time: %s" % (time.time()-start))
    
    def convert_time(self):
        """
        converts timestamp to datetime object, stored as self.time
        """
        print("Converting time to datetime object...")
        start = time.time()
        self.time = pd.to_datetime(self.timestamps,unit='ms') - pd.DateOffset(hours=7)
        end = time.time()
        print("Time: %s" % (end-start))

    def make_df(self,time_index=False):
        """
        Creates a dataframe of the twitter data
        INPUT
            time_index (bool) whether to set the dataframe index as the time variable, default=False
        RETURNS
            df (pandas DataFrame) of twitter data
        """
        if time_index:
            df = pd.DataFrame(index=self.time)
        else:
            df = pd.DataFrame()
            df['time'] = self.time
        df['screen_name'] = self.user_stats[:,0]
        df['description'] = self.user_stats[:,1]
        df['n_weblinks'] = self.n_weblinks
        df['n_mentions'] = self.n_mentions
        df['n_hashtags'] = self.n_hashtags
        df['RT'] = self.retweets
        return df
    
    def tokenize_hashtags(self):
        """
        Tokenize the hashtags using TfidfVectorizer
        """
        start = time.time()
        self.V = Vec(max_features=100,
                     min_df=100,
                     max_df=.95,
                     sublinear_tf=True,
                     use_idf=True)
        H = self.V.fit_transform(self.hashtags)
        end = time.time()
        print("Time: %s" % (end-start))
        return H
    
    def tokenize_mentions(self):
        """
        Tokenize mentions using TfidfVectorizer
        RETURNS
            M (TfidfVectorizer) fitted to mentions data, parameters sublinear_tf=True and use_idf=True
        """
        start = time.time()
        self.V = Vec(sublinear_tf=True,
                     use_idf=True)
        M = self.V.fit_transform(self.mentions)
        end = time.time()
        print("Time: %s" % (end-start))
        return M
        
    def get_topics(self,n_topics=3,n_features=1000,ngram=(4,6),n_top_words=1):
        """
        Use LDA for topic extraction
        """
        print("Topic Extraction...")
        def print_top_words(model, feature_names, n_top_words=1):
            for topic_idx, topic in enumerate(model.components_):
                print("Topic #%d:" % topic_idx)
                print "\t"," ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print()
        start = time.time()
        TFvec = CountVectorizer(max_df=0.95, min_df=2, ngram_range=ngram, max_features=n_features, stop_words='english')
        tf = TFvec.fit_transform(self.tweets)
        lda = LDA(n_topics=n_topics, learning_method='online', learning_offset=50., random_state=0)
        lda.fit(tf)
        tf_feature_names = TFvec.get_feature_names()
        print("Time: %s" % (time.time() - start))
        print_top_words(lda, tf_feature_names)
        #print "\n\n\n",TFvec.get_feature_names()


def load_candidate(n=None,m=None):
    """
    Trump: (1284126,)
    Clinton: (,)
    """
    filename = get_file()
    c = TwitterCorpus(filename,n,m)
    c.clean_text()
    c.convert_time()
    return c


