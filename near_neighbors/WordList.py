# WordList.py
"""Volume II Lab 5: Data Structures II (Trees). Auxiliary file
Use this function to complete problem 4. Do not modify.
"""

import numpy as np

# Use this function in problem 4.
def create_word_list(filename='English.txt'):
    """Read in a list of words from the specified file.
    Randomize the ordering and return the list.
    """
    with open(filename, 'r') as myfile: # Open the file with read-only access
        contents = myfile.read()        # Read in the text from the file
                                        # The file is implicitly closed
    wordlist = contents.split('\n')     # Get each word, separated by '\n'
    if wordlist[-1] == '\n':            # Remove the last endline
        wordlist = wordlist[:-1]                
    # Randomize, convert to a list, and return.
    return list(np.random.permutation(wordlist))

# You do not need this function, but read it anyway.
def export_word_list(words, outfile='Test.txt'):
    """Write a list of words to the specified file. You are not required
    to use this function, but it may be useful in testing sort_words().
    Note that 'words' must be a Python list.
    
    These two functions are examples of how file input / output works in
    Python. This concept will resurface many times in later labs.
    """
    f = open(outfile, 'w')       # Open the file with write-only access
    for w in words:                 # Write each word to the file, appending
        f.write(w + '\n')        #   an endline character after each word
    f.close()                       # Close the file.
