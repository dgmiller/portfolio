### ###
# please note: this file has been modified from its original version to protect company information

# Helpful functions for clv analysis
from scipy.spatial.distance import cosine
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import math
from scipy.sparse import lil_matrix
from scipy.stats import beta
from collections import defaultdict
import networkx as nx
import pyodbc as odbc
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster import vq
import re
import time
from mpl_toolkits.basemap import Basemap
import geopy
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

class MultiArmedBandit(object):
    """
    Simulates a row of slot machines in order to address the exploration vs exploitation dilemma
    """
    def __init__(self,n):
        """
        INPUT:
            n (int) number of arms of the bandit
        METHODS:
            next_arm()
            update_arm(arm_index,successful)
            display(names)
        """
        self.n = n
        self.arms = []
        for i in xrange(self.n):
            # initialize each arm with zero successes and zero failures
            self.arms.append([0,0])
    
    def next_arm(self):
        """
        Draws randomly from the posterior distributions of each arm and returns the index of the maximum value
        """
        theta = [] # a list to store our random draws from the various distributions
        for a in xrange(self.n):
            # the beta distribution is used to draw random samples from the various arms
            # it takes two parameters corresponding to success and failure
            # since the beta distribution is undefined where a=b=0, we add 1 to a and b
            theta.append(np.random.beta(self.arms[a][0] + 1,self.arms[a][1] + 1))
        return np.argmax(theta) # returns the index of the maximum value in the list
    
    def update_arm(self,index,successful):
        """
        Update the indicated bandit arm.
        INPUT:
            index (int) of the arm to update
            successful (bool) set to True if the outcome was a success
        """
        if successful:
            self.arms[index][0] += 1
        else:
            self.arms[index][1] += 1

    def display(self,names=None):
        """
        Plot the posterior distributions of the bandit arms.
        INPUT:
            names (list) of string names of the bandit arms; defaults to None; optional
        """
        for i in xrange(self.n):
            a = self.arms[i][0] + 1
            b = self.arms[i][1] + 1
            # generate values that will define the line of the beta distribution
            x = np.linspace(beta.ppf(0.01, a, b),beta.ppf(0.99, a, b),100)
            l = str(i)
            if names:
                plt.plot(x,beta(a,b).pdf(x),label=names[i]) # plot the pdf of the beta distribution
            else:
                plt.plot(x,beta(a,b).pdf(x),label=l)
        plt.legend(loc=2)
        plt.title(str(self.arms))
        plt.show()

# tests the multi armed bandit class
def testMAB(n,probabilities,names=None,niters=100):
    """
    Tests the Thompson Sampling algorithm in the MultiArmedBandit object.
    INPUT:
        n (int) number of arms
        probabilities (list) of floats between 0 and 1 indicating the expected payout of each arm
        names (list) of arm names niters (int) number of iterations to perform, default=100
    OUTPUT:
        Displays the distributions of each arm
    """
    if len(probabilities) != n:
        raise ValueError('mismatched arm probabilities')
    M = MultiArmedBandit(n)
    for i in range(niters):
        if i%25 == 0 and i != 0:
            M.display(names)
        arm = M.next_arm()
        M.update_arm(arm,np.random.binomial(1,probabilities[arm]))
    M.display(names)

class recommender(object):
    """
    A recommendation engine object
    """
    def __init__(self,datafile):
        """
        set up the recommender object from the given datafile
        IN
            datafile (string) the filepath of the data used to generate the adjacency matrix
                                see '../data/PTTPARTS.csv' for an example
        attributes
            self.customers (array) of customer ids
            self.to_index (dict) converts customer ids to the corresponding matrix index in self.A
            self.from_index (dict) converts indices to customer ids
            self.U (array) of unique product types
            self.data (list of lists) each list represents a customer and what they have purchased
            self.m (int) the number of rows in self.A
            self.n (int) the number of columns of self.A
            self.A (matrix) of zeros and ones made from self.data; has dimension mxn
            self.D (matrix) set to None as default; we calculate it using self.similarity_matrix()

        METHODS
            self.similarity_matrix(): calculates self.D
            self.similar_to(): calculates similar customers
            self.recommend(): recommends products to customers
        """
        to_name = make_to_name()
        # load the customer id and product name data
        customer_parts = np.loadtxt(datafile,dtype=str,delimiter='\t')
        # the first row of customer_parts contains the customer ids
        self.customers = np.unique(customer_parts[:,0])
        self.to_index = dict(zip(self.customers,range(len(self.customers))))
        self.from_index = dict(zip(self.to_index.values(),self.to_index.keys()))
        self.U = np.unique(customer_parts[:,1]) # this row contains the product name
        # initialize a dictionary of lists, arbitrarily named F
        F = {}
        # make a list for each customer
        for row in customer_parts:
            F[row[0]] = []
        # append each product to the customer id
        for row in customer_parts:
            #row[1] != "Salt Standards" and row[1] != "Sample Cups":
            F[row[0]].append(row[1])
        # create a list containing the lists of products
        data = []
        for i in self.customers:
            data.append(F[i])
        self.data = data
        self.m = len(self.customers) # number of users
        self.n = len(self.U) # number of products
        # A is the adjacency matrix
        A = np.zeros((self.m,self.n))
        for i in range(self.m):
            for j in range(self.n):
                if self.U[j] in self.data[i]:
                    A[i,j] += 1
        self.A = A
        self.D = None

    def similarity_matrix(self,transpose=False):
        """
        computes the similarity matrix using cosine similarity
        IN
            transpose (bool) default=False, whether to transpose the adjacency matrix
                            determines similarity between customers or products
        OUT
            matrix of similarity scores
        """
        if transpose:
            # take the transpose of A (see numpy documentation)
            A = self.A.T
            # dimension of the matrix D
            dim = self.n
        else:
            # dimension of the matrix D
            A = self.A.copy()
            dim = self.m
        print('start time')
        print(time.ctime())
        print('computing...')
        start = time.time()
        # lil_matrix allows us to efficiently store data in memory (see scipy documentation)
        D = lil_matrix((dim,dim))
        for i in xrange(dim):
            if i % 1000 == 0:
                print(float(i)/dim)
            for j in xrange(dim):
                # compute cosine similarity between row i and column j
                D[i,j] = -1*(cosine(A[i],A[j]) - 1)
        # convert D to a dense (rather than sparse) matrix
        self.D = D.todense()
        end = time.time()
        t = end-start
        print('finished in %d seconds') % t

    def similar_to(self,user0,user_index=False):
        """returns a sorted array of similar users with the degree of (cosine) similarity"""
        if user_index == False:
            user0 = self.to_index[str(user0)]
        pairs = []
        for user1,f1 in enumerate(self.D[:,user0]):
            # if user0 and user1 are different but have at least some similarity
            if user0 != user1 and f1 > 0:
                # use customer id
                #pairs.append((self.customers[user1],f1))
                pairs.append((user1,f1))
        dtype = [('user_id',int),('relation',float)]
        pairs = np.array(pairs, dtype=dtype)
        return np.sort(pairs,order='relation')[::-1]

    def recommend(self,user0,user_index=False):
        """
        recommends products based on similar users
        IN
            user0 (int) the index of the customer to recommend to
        OUT
            the top ten recommendations for user0
        """
        S = {}
        for i in self.U:
            S[i] = 0
        for user1,f1 in self.similar_to(user0,user_index=user_index):
            for i in self.data[user1]:
                S[i] += f1
        # sort values by weight
        R = sorted(S.items(), key=lambda (_,weight): weight, reverse=True)
        # return the top ten recommendations
        if user_index == False:
            j = self.to_index[str(user0)]
        else:
            j = int(user0)
        return np.array([(r,w) for r,w in R if r not in self.data[j]])[:10]


def expand(matrix,power):
    """
    helper function for markov_cluster_algorithm
    matrix exponentiation representing a markov process
    """
    # csr_matrix allows for efficient storage and fast operations
    # see numpy documentation for details
    M = np.linalg.matrix_power(matrix.todense(), power)
    return csr_matrix(M)

def inflate(matrix, n):
    """
    helper function for markov_cluster_algorithm
    entrywise exponentiation of a matrix
    """
    M = matrix.power(n)
    return M

def markov_cluster_algorithm(matrix,expow,inpow,niters=100):
    """
    A clustering algorithm that iteratively breaks down connections in a network
    to form groups or clusters of similar objects. These networks are known as graphs.
    (see Wikipedia: Graph Theory for more information)
    IN
        matrix (ndarray) the matrix representation of the network to cluster
        expow (int) exponent power of the matrix, simluating a markov process
        inpow (int) 'inflation' power for entry wise exponentiation
        niters (int) how many iterations to perform, default=100
    OUT
        M (matrix) represents the graph (network) where objects are clustered
    """
    matrix = csr_matrix(matrix)
    M = normalize(matrix,norm='l1')
    for i in xrange(niters):
        if i%25 == 0:
            print(i)
        # use normalize from sklearn
        # the l1 norm is usually sufficient
        M = normalize(inflate(expand(M,expow),inpow),norm='l1')
    return M.todense()


class Nightmap(object):
    """
    This class creates a dataframe for clv data and location.
    Plotting methods are available as well as some algorithms to perform on the data.

    requires matplotlib Basemap library
    """
    def __init__(self,filename,latitudes=None,longitudes=None):
        """
        IN
            filename (str) the file that contains the data to use for plotting
            latitudes (tuple) limit the display to customers who fall between a minimum and maximum latitude
            longitudes (tuple) limit the display to customers who fall between a min and max longitude

        attributes
            self.data (DataFrame) the customer id, clv, gps coordinates, and level of gps accuracy

        METHODS
            nightplot(kmeans=None) plots clv as brightness on a dark geographical map
            plot_kmeans_err() plots the kmeans error as k gets larger
            worldplot(kmeans=None,proj='merc') plots customers on a map of the continental US
        """
        gps = pd.read_csv(filename,names=['id','clv','lat','lon','level'],skiprows=1,delimiter=',')
        if latitudes and longitudes:
            gps = gps[gps['lon']<longitudes[0]]
            gps = gps[gps['lon']>longitudes[1]]
            gps = gps[gps['lat']<latitudes[1]]
            gps = gps[gps['lat']>latitudes[0]]
        else:
            gps = gps[gps['lon']<-60.]
            gps = gps[gps['lon']>-133.]
        self.data = gps
        print('clv for area:')
        print(gps['clv'].sum())

    def nightplot(self,kmeans=None):
        """
        Plots a geographic map of the data based on GPS location and a luminosity value corresponding to clv.
        IN
            kmeans (int) the number of 'clusters' or means of the gps points
        """
        plt.scatter(self.data['lon'],self.data['lat'],c=np.log(self.data['clv']),s=20,cmap='viridis_r',alpha=.6,linewidths=0)
        ax = plt.gca()
        ax.set_axis_bgcolor('black')
        if kmeans:
            data_in = self.data.drop(['clv','city','state','country','err','log_clv'],axis=1)
            # compute means using k-means clustering algorithm (see wikipedia for details)
            # vq is scipy's vector quantization module
            output,distortion = vq.kmeans(data_in,kmeans)
            plt.scatter(output[:,1],output[:,0],s=35,c="red",linewidths=0)
        plt.show()
    
    
    def plot_kmeans_err(self):
        """
        Performs k-means clustering for various sizes of k. Helps to determine which k to use.
        Generally, we see diminishing returns around k=12
        """
        L = []
        data_in = self.data.drop(['id','clv','level'],axis=1)
        for i in range(1,50):
            # vq is scipy's vector quantization module
            output = vq.kmeans(data_in,i)
            L.append(output[1])
        plt.plot(range(1,50),L)
        plt.show()
    
    def worldplot(self,kmeans=None,proj='merc'):
        """
        plots customer GPS location on a map with state and national boundaries.
        IN
            kmeans (int) number of means for k-means clustering, default=None
            proj (string) the map projection to use, use 'robin' to plot the whole earth, default='merc'
        """
        # create a matplotlib Basemap object
        if proj == 'robin':
            my_map = Basemap(projection=proj,lat_0=0,lon_0=0,resolution='l',area_thresh=1000)
        else:
            my_map = Basemap(projection=proj,lat_0=33.,lon_0=-125.,resolution='l',area_thresh=1000.,
                    llcrnrlon=-130.,llcrnrlat=25,urcrnrlon=-65., urcrnrlat=50)
        my_map.drawcoastlines(color='grey')
        my_map.drawcountries(color='grey')
        my_map.drawstates(color='grey')
        my_map.drawlsmask(land_color='white',ocean_color='white')
        my_map.drawmapboundary() #my_map.fillcontinents(color='black')
        x,y = my_map(np.array(self.data['lon']),np.array(self.data['lat']))
        my_map.plot(x,y,'ro',markersize=3,alpha=.4,linewidth=0)
        if kmeans:
            # k-means clustering algorithm---see wikipedia for details
            data_in = self.data.drop(['id','clv','level'],axis=1)
            # vq is scipy's vector quantization module
            output,distortion = vq.kmeans(data_in,kmeans)
            x1,y1 = my_map(output[:,1],output[:,0])
            my_map.plot(x1,y1,'ko',markersize=20,alpha=.4,linewidth=0)
        plt.show()
        return output

class Geocode(object):
    """
    Pulls customer address information from a database and turns them into gps coordinates.
    """
    def __init__(self):
        """
        attributes:
        self.error = []
        self.cmd = <sql command string>
        self.full = dict from customer id number to full address string
        self.city = dict from customer id number to city level address string
        self.zipcode = dict from customer id number to zipcode level address string
        """
        self.error = []
        self.cmd = """<SQL DATABASE QUERY GOES HERE>"""
        # c contains the rows of the query result; see below for information about query()
        c = query(self.cmd)
        L = []
        for row in c:
            # pull data from row
            num,name,address,city,state,zp,country = row
            # names need to be converted to ascii format to avoid errors
            num = num.encode('ascii','ignore')
            name = name.encode('ascii','ignore')
            address = address.encode('ascii','ignore')
            city = city.encode('ascii','ignore')
            state = state.encode('ascii','ignore')
            zp = zp.encode('ascii','ignore')
            country = country.encode('ascii','ignore')
            # the use() function checks if the name contains 'do not use' or 'inactive'
            if use(name):
                cdata = {'id':num,'address':address,'city':city,'state':state,'zip':zp,'country':country}
                L.append(cdata)
        # put data into dataframe for convenient modification
        df = pd.DataFrame(L,columns=['id','name','city','state','zip','country','address'])
        city = np.array(df['city'] + ", " + df['state'] + " " + df['country'])
        zipcode = np.array(df['state'] + " " + df['zip'] + " " + df['country'])
        full = np.array(df['address'] + " " + df['city'] + ", " + df['state'] + " " + df['zip'] + " " + df['country'])
        # a dictionary mapping customer id to their address (formatted as a string)
        self.full = dict(zip(df.id,full)) # makes updating gps coordinates by id easier
        self.city = dict(zip(df.id,city))
        self.zipcode = dict(zip(df.id,zipcode))

    def to_gps(self):
        """
        writes customer id, latitude, longitude, address level to a csvfile called 'new_gps.csv'

        computation time is approx 1-4 seconds per address
        """
        filename = open('new_gps.csv','w')
        filename.write('id,latitude,longitude,level')
        filename.write('\n')
        print("\ncomputing...")
        geoLocation = geopy.Nominatim()
        count = 0
        start = time.time()
        for a in self.full.keys(): # the keys are the customer ids
            # we try to get full address first, then zip code, then city with an error if they all fail to geocode
            try:
                # access the API and geocode the address
                location = geoLocation.geocode(self.full[a])
                # store the results
                line = str(a) + ',' + str(location.latitude) + ',' + str(location.longitude) + ',full address' + '\n'
                filename.write(line)
                # record that there was no error
                self.error.append(0)
            except:
                try:
                    location = geoLocation.geocode(self.zipcode[a])
                    line = str(a) + ',' + str(location.latitude) + ',' + str(location.longitude) + ',zip code' + '\n'
                    filename.write(line)
                    self.error.append(0)
                except:
                    try:
                        location = geoLocation.geocode(self.city[a])
                        line = str(a) + ',' + str(location.latitude) + ',' + str(location.longitude) + ',city' + '\n'
                        filename.write(line)
                        self.error.append(0)
                    except:
                        line = str(a) + ',0.0,0.0,error\n'
                        filename.write(line)
                        self.error.append(1)
            count += 1
            if count % 100 == 0:
                print(count)
        filename.close()
        end = time.time()
        t = end - start
        # total number of addresses attempted
        total = len(self.error)
        # number of geocode errors
        num_err = sum(self.error)
        print('finished in %s seconds with %s errors out of %s total') % (t,num_err,total)

    def try_again(self,f='new_gps.csv'):
        """
        gets failed attempts from previous file and tries to geocode again
        note to self: REDO THIS FUNCTION
        """
        A = pd.read_csv(f)
        A = A[A['level']=='error']
        self.error = []
        count = 0
        filename = open('retry.csv','w')
        filename.write('id,latitude,longitude,level')
        filename.write('\n')
        print("\ncomputing...")
        geoLocation = geopy.Nominatim()
        count = 0
        start = time.time()
        for a in A.id:
            a = str(a)
            try:
                # access the API and geocode the address
                location = geoLocation.geocode(self.full[a])
                # store the results
                line = str(a) + ',' + str(location.latitude) + ',' + str(location.longitude) + ',full address' + '\n'
                filename.write(line)
                # record that there was no error
                self.error.append(0)
            except:
                try:
                    location = geoLocation.geocode(self.zipcode[a])
                    line = str(a) + ',' + str(location.latitude) + ',' + str(location.longitude) + ',zip code' + '\n'
                    filename.write(line)
                    self.error.append(0)
                except:
                    try:
                        location = geoLocation.geocode(self.city[a])
                        line = str(a) + ',' + str(location.latitude) + ',' + str(location.longitude) + ',city' + '\n'
                        filename.write(line)
                        self.error.append(0)
                    except:
                        line = str(a) + ',0.0,0.0,error\n'
                        filename.write(line)
                        self.error.append(1)
            count += 1
            if count % 100 == 0:
                print(count)
        filename.close()
        end = time.time()
        t = end - start
        # total number of addresses attempted
        total = len(self.error)
        # number of geocode errors
        num_err = sum(self.error)
        print('finished in %s seconds with %s errors out of %s total') % (t,num_err,total)

def make_to_name():
    """
    generates a dictionary mapping customer ids to customer names using names.csv
    The file names.csv may need to be updated with the code in update_names.py
    """
    return dict(np.genfromtxt('../data/names.csv',dtype=str,delimiter='\t'))

def make_from_name(to_name):
    """
    generates a dictionary mapping customer names to ids using dictionary to_name
    """
    from_name = dict(zip(to_name.values(),to_name.keys()))
    return from_name

def query(cmd,filename=None,dsn):
    """
    INPUT:
        cmd (str) sql command to run
        dsn (str) server name
    OUTPUT:
        c (pyodbc object) contains the rows of data generated from the query
    """
    cnxn_name = "DSN=%s" % dsn
    connection = odbc.connect(cnxn_name) # use to access the database
    c = connection.cursor() # generate cursor object
    c.execute(cmd) # connect to the database using the above sql code
    return c

def name_by_area(lat,lon,filename):
    """
    generate names of customers in a certain geographic area
    this was used to generate the data found in all_cities.txt
    IN
        latitudes (tuple) min and max latitude lines
        longitudes (tuple) min and max longitude lines
        filename (string) the name of the file
    """
    to_name = make_to_name()
    N = Nightmap(filename,latitudes=lat,longitudes=lon)
    cust_id = np.array(N.data.id)
    names = gen_names(cust_id,to_name)
    N.worldplot()
    from_name = make_from_name(to_name)
    filename = raw_input('filename? ')
    if filename:
        filename = open(filename,'w')
        for n in names:
            i = from_name[n]
            filename.write(i)
            filename.write('\t')
            filename.write(n)
            filename.write('\n')
        filename.close()


class CLV(object):
    """
    INPUT
        pmg_num (int) the product market group number, default = 1
        outfile1 (str) the filename indicating where to store the raw data before analysis, default = '../data/clvtrainingset01.csv'
        outfile2 (str) the filename containing the results, default = '../data/clv01.csv'
        date_range (list) the start date and end date of the years to analyze, default = ['2008-09-01','2016-09-01']
    attributes other than those listed above
        self.data (DataFrame) a pandas DataFrame object of the data to be used for analysis
        self.bgf (from lifetimes) a statistical model object from the lifetimes package
        self.ggf (from lifetimes) a statistical model object from the lifetimes package
        self.results (DataFrame) a pandas DataFrame object of the results of analysis
    """
    def __init__(self,pmg_num=1,outfile1='../data/clvtrainingset01.csv',outfile2='../data/clv01.csv',date_range=['2008-09-01','2016-09-01']):
        self.pmg_num = pmg_num
        # outfile1 stores a clean version of the raw data used for analysis; this is important for reproducibility
        self.outfile1 = outfile1
        # outfile2 stores the clv estimation results
        self.outfile2 = outfile2
        self.date_range = date_range
        self.data = None
        self.bgf = None
        self.ggf = None
        self.results = None

    def get_data_from_server(self,cmd=None):
        """
        Gets data from sales_db and stores the query results in self.data
        INPUT
            cmd (str) the default sql query is below

            The default query has been replaced. The original query was an 8 line select command.
        """
        # server name
        dsn = "THE SERVER NAME"
        cnxn_name = "DSN=%s" % dsn
        connection = odbc.connect(cnxn_name) # use to access the database
        c = connection.cursor() # generate cursor object
        
        # Grab transaction data from Postgres
        if not cmd:
            cmd = """SQL DEFAULT COMMAND GOES HERE""" % (self.pmg_num,self.date_range[0],self.date_range[1])
        
        c.execute(cmd) # execute the sql command
        
        # list to store the query data
        transaction_data = []
        
        # create a dictionary to convert customer ids to name
        to_name = dict(np.genfromtxt('../data/names.csv',dtype=str,delimiter='\t'))
        
        for row in c:
            cust, rsv_date, sales = row # pull data from each row of the query data
            cust_id = str(int(cust))
            name = to_name[cust_id]
            # check to see if customer is inactive
            if use(name):
                rsv_date1_readable = rsv_date.strftime('%Y-%m-%d') # date formatting
                sales_float = float(sales) # convert to float; represents the transaction amount
                transaction_data.append({"id":cust, "date":rsv_date, "sales":sales_float}) # add dictionary of data to list
        
        # convert to dataframe
        df = pd.DataFrame(transaction_data, columns=['id', 'date', 'sales'])
        # store results
        df.to_csv(self.outfile1,index=False)
        # IMPORTANT: use correct observation_period_end date
        self.data = summary_data_from_transaction_data(df, 'id', 'date', 'sales', observation_period_end=self.date_range[1], freq='M')

    def get_data_from_file(self,filename,**kwargs):
        df = pd.read_csv(filename,**kwargs)
        self.data = summary_data_from_transaction_data(df, 'id', 'date', 'sales', observation_period_end=self.date_range[1], freq='M')

    def fit(self,months=96):
        """
        Computes CLV estimates for the next n months and stores results in self.results
        INPUT
            months (int) number of months to predict, default = 96 (8 years)
        """
        ### PREDICT NUMBER OF PURCHASES
        self.bgf = BetaGeoFitter() # see lifetimes module documentation for details
        self.bgf.fit(self.data['frequency'], self.data['recency'], self.data['T'])
        # 8 years = 96 months
        self.data['predicted_purchases'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(
                months,
                self.data['frequency'],
                self.data['recency'],
                self.data['T'])

        ### PREDICT FUTURE PURCHASE AMOUNT
        self.ggf = GammaGammaFitter(penalizer_coef = 0)
        self.ggf.fit(self.data['frequency'], self.data['monetary_value'])
        # predict next transaction
        self.data['predicted_trans_profit'] = self.ggf.conditional_expected_average_profit(
                frequency = self.data['frequency'],
                monetary_value = self.data['monetary_value'])
        
        ### ESTIMATE CLV
        self.data['clv_estimation'] = self.data['predicted_trans_profit'] * self.data['predicted_purchases']
        self.data['prob_alive'] = self.bgf.conditional_probability_alive(
                self.data['frequency'],
                self.data['recency'],
                self.data['T'])
        self.results = self.data.sort_values(by='clv_estimation',ascending=False)
        # store results
        self.results.to_csv(self.outfile2,index=False)

    def plot_matrices(self):
        """
        plots three matrices:
            probability alive matrix: displays the probability that a customer is active
            frequency recency matrix: displays frequency and recency with color corresponding
                                        to monetary value
            period transactions: displays predicted and actual transaction values over time
            (check documentation in lifetimes for more details)
        """
        plot_probability_alive_matrix(self.bgf,cmap='viridis')
        plot_frequency_recency_matrix(self.bgf,cmap='viridis')
        plot_period_transactions(self.bgf)


def gen_names(cust_id,to_name):
    """returns a list of customer names given their IDs
    IN
        cust_id (array) of customer ids
        to_name (dict) mapping id -> name
    OUT
        S (list) of customer names
        err (list) of customer ids that created an error
    """
    # S contains the customer names
    S = []
    # err contains the error customer ids
    err = []
    for i in cust_id:
        try:
            S.append(to_name[str(i)])
        except:
            S.append('ERROR: %s' % str(i))
            err.append(i)
    if len(err) > 0:
        print('PROBLEM: %s errors' % len(err))
        print(err)
    return S

def inactive(S):
    """
    Returns the names and IDs of customers that are not active.
    The inactive customer names have one or more of the following messages:
        INACTIVE
        DO NOT USE
        DO NOT SELL
        RESELL
    IN
        S (array-like) of customer names
    OUT
        (array-like) of names with problems
    """
    F = []
    for item in S:
        item = item.lower()
        inactive = re.search('inactive',item)
        donotuse = re.search('do not use',item)
        donotsell = re.search('do not sell',item)
        reseller = re.search('resell',item)
        if inactive or donotuse or donotsell or reseller:
            F.append(item)
    return F

def use(name):
    """
    Returns False if the customer should not be used, based on certain name attributes
    IN
        name (string) which is the customer's name
    OUT
        (boolean) indicating if the name and id should be included in analysis
    """
    name = name.lower()
    inactive = re.search('inactive',name)
    donotuse = re.search('do not use',name)
    donotsell = re.search('do not sell',name)
    reseller = re.search('resell',name)
    if inactive or donotuse or donotsell or reseller:
        return False
    else:
        return True


