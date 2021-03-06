# spec.py
"""Volume II Lab 8: Markov Chains
Derek Miller
Vol 2
29 Oct 2015
"""

import numpy as np

# Problem 1: implement this function.
def random_markov(n):
    """Create and return a transition matrix for a random
    Markov chain with 'n' states as an nxn NumPy array.
    """
    chain = np.random.rand(n,n)
    for x in xrange(n):
        S = sum(chain[:,x])
        chain[:,x] /= S
    return chain


# Problem 2: modify this function.
def forecast(num_days):
    """Run a simulation for the weather over 'num_days' days, with
    "hot" as the starting state. Return a list containing the day-by-day
    results, not including the starting day.

    Example:
        >>> forecast(5)
        [1, 0, 0, 1, 0]
        # Or, if you prefer,
        ['cold', 'hot', 'hot', 'cold', 'hot']
    """
    trans_matrix = np.array([[.7, .6],[.3, .4]])
    day = 0
    predict = []
    for n in xrange(num_days):
        outcome = np.random.rand()
        if day == 1: # baby it's cold outside...
            if outcome < trans_matrix[0,1]:
                day = 0
                predict.append(day)
            else:
                day = 1
                predict.append(day)
        else:
            if outcome < trans_matrix[1,0]:
                day = 1
                predict.append(day) # cold tomorrow
            else:
                day = 0
                predict.append(day)
    return predict


# Problem 3: implement this function.
def four_state_forecast(days=1):
    """Same as forecast(), but using the four-state transition matrix.
    [[.5 .3 .1 .0]
     [.3 .3 .3 .3]
     [.2 .3 .4 .5]
     [.0 .1 .2 .2]]
    """
    trans_matrix = np.array([
        [.5, .3, .1, 0.],
        [.3, .3, .3, .3],
        [.2, .3, .4, .5],
        [0., .1, .2, .2]])
    temp = 0
    predict = []
    for x in xrange(days):
        crystal_ball = np.random.multinomial(1, trans_matrix[:,temp])
        temp = np.argmax(crystal_ball)
        predict.append(temp)

    return predict
    
# Problem 4: implement this function.
def analyze_simulation():
    """Analyze the results of the previous two problems. What percentage
    of days are in each state? Print your results to the terminal.
    """
    first = forecast(10000)
    d0 = first.count(0) / 10000.
    d1 = first.count(1) / 10000.
    second = four_state_forecast(10000)
    e0 = second.count(0) / 10000.
    e1 = second.count(1) / 10000.
    e2 = second.count(2) / 10000.
    e3 = second.count(3) / 10000.

    print("\nI\n\nHot Days: %f4\nCold Days: %f4\n\n") % (d0,d1)
    print("II\n\nHot: %f4\nMild: %f4\nCold: %f4\nFreezing: %f4\n") % (e0,e1,e2,e3)


# Problems 5-6: define and implement the described functions.
def numberize(read_file='textfile.txt'):
    """Reads in a text file that maps each word to a number.
    Writes the number of a word to a file called 'num_text.txt'.
    Returns a list of unique words from the text file.

    INPUTS:
        read_file   : (string) the filepath of the training set.
                      defaults to 'textfile.txt'

    OUTPUTS:
        lexicon     : (list) unique words from read_file
                      with lexicon[0] = '::BEGIN LIST>>' and
                      lexicon[-1] = '<<END LIST::'
    """
    n = 1
    num_text = open('num_text.txt','w')
    train = open(read_file,'r').readlines()
    lexicon = ['::BEGIN LIST>>']
    for line in train:
        word_list = line.lower().split()
        for word in word_list:
            if lexicon.count(word) == 0:
                lexicon.append(word)
                num_text.write(str(lexicon.index(word)) + " ")
            else:
                num_text.write(str(lexicon.index(word)) + " ")
        num_text.write("\n")
    lexicon.append('<<END LIST::')
    num_text.close()
    return lexicon

# Problem 6
def adj_matrix(filepath, uwords):
    """Reads in the number of unique words plus two. Defaults to the length
       of the output of numberize().
       
       INPUT:
            uwords  : number of unique words plus 2 [defaults to len(numberize())]
            
       OUTPUT:
            M       : transition matrix for a markov chain of dim uwords x uwords
    """
    # the file in a list of sentences (strings)
    indices = open(filepath,'r').readlines()
    uwords += 2
    t_matrix = np.zeros((uwords,uwords))
    for a in xrange(len(indices)):
        indices[a] = indices[a].split() # list of 'numbers' (strings)
    for sentence in indices: # list of numbers in each line
        t_matrix[(int(sentence[0]),0)] += 1 # 
        for s in range(len(sentence)-1):
            t_matrix[(int(sentence[s+1]),int(sentence[s]))] += 1
        t_matrix[uwords-1,int(sentence[-1])] += 1
    t_matrix[uwords-1,uwords-1] += 1
    for col in range(len(t_matrix[:,])):
        t_matrix[:,col] /= np.sum(t_matrix[:,col])
    return t_matrix



# Problem 7: implement this function.
def sentences(infile, outfile, num_sentences=1):
    """Generate random sentences using the word list generated in
    Problem 5 and the transition matrix generated in Problem 6.
    Write the results to the specified outfile.

    Parameters:
        infile (str): The path to a filen containing a training set.
        outfile (str): The file to write the random sentences to.
        num_sentences (int): The number of random sentences to write.

    Returns:
        None
    """
    opus = open(outfile,'w')
    uniwords = numberize(infile)
    M = adj_matrix('num_text.txt',len(uniwords))
    for x in xrange(num_sentences):
        word = 0
        while word != len(M)-1:
            anon = np.random.multinomial(1,M[:,word])
            word = np.argmax(anon)
            if word != len(M)-1:
                opus.write(uniwords[word] + " ")
        opus.write("\n")
    opus.close()

sentences('train_drSwift.txt','out_drSwift.txt',num_sentences=25)
