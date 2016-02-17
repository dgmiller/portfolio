# spec.py
"""Volume II Lab 7: Breadth-First Search (Kevin Bacon)
<name>
<class>
<date>
"""

from collections import deque
import networkx as nx

# Problems 1-4: Implement the following class
class Graph(object):
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a list of the
    corresponding node's neighbors.

    Attributes:
        dictionary: the adjacency list of the graph.
    """

    def __init__(self, adjacency):
        """Store the adjacency dictionary as a class attribute."""
        self.dictionary = adjacency

    # Problem 1
    def __str__(self):
        """String representation: a sorted view of the adjacency dictionary.
        
        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> print(Graph(test))
            A: B
            B: A; C
            C: B
        """
        str_rep = ""
        keys = sorted(self.dictionary.keys())
        for k in keys:
            val = sorted(self.dictionary[k])
            str_rep += "%s : %s\n" % (k, val)
        return str_rep

    # Problem 2
    def traverse(self, start):
        """Begin at 'start' and perform a breadth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation).

        Raises:
            ValueError: if 'start' is not in the adjacency dictionary.

        Example:
            >>> test = {'A':['B'], 'B':['A', 'C',], 'C':['B']}
            >>> Graph(test).traverse('B')
            ['B', 'A', 'C']
        """
        if self.dictionary[start] is None:
            raise ValueError('<start node> is not in the graph')
        N = start # N is the current node; its neighbors are denoted by n
        visit_queue = deque()
        marked = set()
        visited = list()
        marked.add(N)
        visited.append(N)
        for n in self.dictionary[N]:
            visit_queue.append(n)
            marked.add(n)
        N = visit_queue.popleft()

    # Problem 3 (Optional)
    def DFS(self, start):
        """Begin at 'start' and perform a depth-first search until all
        nodes in the graph have been visited. Return a list of values,
        in the order that they were visited. If 'start' is not in the
        adjacency dictionary, raise a ValueError.

        Inputs:
            start: the node to start the search at.

        Returns:
            the list of visited nodes (in order of visitation)
        """
        raise NotImplementedError("Problem 3 Incomplete")

    # Problem 4
    def shortest_path(self, start, target):
        """Begin at the node containing 'start' and perform a breadth-first
        search until the node containing 'target' is found. Return a list
        containg the shortest path from 'start' to 'target'. If either of
        the inputs are not in the adjacency graph, raise a ValueError.

        Inputs:
            start: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from start to target,
                including the endpoints.

        Example:
            >>> test = {'A':['B', 'F'], 'B':['A', 'C'], 'C':['B', 'D'],
            ...         'D':['C', 'E'], 'E':['D', 'F'], 'F':['A', 'E', 'G'],
            ...         'G':['A', 'F']}
            >>> Graph(test).shortest_path('A', 'G')
            ['A', 'F', 'G']
        """
        raise NotImplementedError("Problem 4 Incomplete")


# Problem 5: Write the following function
def convert_to_networkx(dictionary):
    """Convert 'dictionary' to a networkX object and return it."""
    nx_graph = nx.Graph()
    for k in dictionary:
        for val in dictionary[k]:
            nx_graph.add_edge(k,val)
    return nx_graph



# Helper function for problem 6
def parse(filename="movieData.txt"):
    """Generate an adjacency dictionary where each key is
    a movie and each value is a list of actors in the movie.
    """

    # open the file, read it in, and split the text by '\n'
    with open(filename, 'r') as movieFile:
        moviesList = movieFile.read().split('\n')
    graph = dict()

    # for each movie in the file,
    for movie in moviesList:
        # get movie name and list of actors
        names = movie.split('/')
        title = names[0]
        graph[title] = []
        # add the actors to the dictionary
        for actor in names[1:]:
            graph[title].append(actor)
    
    return graph


# Problems 6-8: Implement the following class
class BaconSolver(object):
    """Class for solving the Kevin Bacon problem."""

    # Problem 6
    def __init__(self, filename="movieData.txt"):
        """Initialize the networkX graph and with data from the specified
        file. Store the graph as a class attribute. Also store the collection
        of actors in the file as an attribute.
        """
        adj_dict = parse(filename)
        self.people = set()
        for key in adj_dict:
            for val in adj_dict[key]:
                if val not in self.people:
                    self.people.add(val)
        self.network = convert_to_networkx(adj_dict)


    # Problem 6
    def path_to_bacon(self, start, target="Bacon, Kevin"):
        """Find the shortest path from 'start' to 'target'."""
        if start not in self.people or target not in self.people:
            raise ValueError('input must be an actor')
        try:
            path = nx.shortest_path(self.network,start,target)
            return path
        except:
            return 0

    # Problem 7
    def bacon_number(self, start, target="Bacon, Kevin"):
        """Return the Bacon number of 'start'."""
        try:
            if start not in self.people or target not in self.people:
                raise ValueError('invalid input')
            n = 0
            path = self.path_to_bacon(start,target)
            for step in path:
                if step in self.people:
                    n += 1
            return n-1
        except:
            return -1

    # Problem 7
    def average_bacon(self, target="Bacon, Kevin"):
        """Calculate the average Bacon number in the data set.
        Note that actors are not guaranteed to be connected to the target.

        Inputs:
            target (str): the node to search the graph for
        """
        BN = 0.
        N = 0.
        for actor in self.people:
            n = self.bacon_number(actor,target)
            if n >= 0:
                N += 1
                BN += n
        return BN/N



### TEST FUNCTION ###

def test(x):
    #problem 1
    if x == 1:
        D = {'A':['C','B'], 'C':['A'], 'B':['A']}
        graph = Graph(D)
        print graph
    elif x == 6:
        #problem 2
        find_kb = BaconSolver()
        print find_kb.path_to_bacon('Kahler, Wolf')
        print find_kb.path_to_bacon('Watson, Emily')
        print find_kb.path_to_bacon('Gosling, Ryan')
        print find_kb.path_to_bacon('Gunther, Kimball')
    elif x == 7:
        #problem 3
        find_kb = BaconSolver()
        print find_kb.bacon_number('Kahler, Wolf')
        print find_kb.bacon_number('Watson, Emily')
        print find_kb.bacon_number('Gosling, Ryan')
        print find_kb.bacon_number('Gunther, Kimball')
        print find_kb.average_bacon()
    else:
        print 'enter the problem number'



# =========================== END OF FILE =============================== #
