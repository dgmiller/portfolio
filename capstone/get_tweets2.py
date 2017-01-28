# Collect Twitter data from the Tonight Show starring Jimmy Fallon
from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import pandas as pd
import json
import os
import time

ckey = 'IWSJe1TLJBvSOQyahRUrL6t13'
csecret = 'bnw4Trgd7y2zEGZ3kN8jbkaDQuArDfbIExEXPOoOeAm1BdG0rV'
atoken = '2331310884-rSIiYPVgCZROan1aBNmYWAyJYtggEDJy2NLzKJL'
asecret = 'PQl6pl6YLpBva22cpJVYwmk28m6NGOJgPleggzmWnYrad'


track_kw = raw_input("Track Keyword: ")
filename = raw_input("filename: ")
# Authentication tokens and keys

start = time.time()
# The listener receives the data
class listener(StreamListener):
    def on_data(self, data):
        # Twitter returns data in JSON format which needs to be decoded
        try:
            with open(filename,'a') as F:
                F.write(data)
        except:
            print('error')
        len_of_time = 60*60*8 # 2 hours = 60*60*2
        #len_of_time = 10
        if time.time() - start > len_of_time:
            return False
        return True
        
    def on_error(self, status):
        print status

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)
twitterStream = Stream(auth, listener())

twitterStream.filter(track=[track_kw])
