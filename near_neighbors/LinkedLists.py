# LinkedLists.py
"""Volume II Lab 4: Data Structures 1 (Linked Lists)
Auxiliary file. Modify this file for problems 1-5.
Derek Miller
Jarvis Lab
24 Sep 2015
"""


# Problem 1: Add the magic methods __str__, __lt__, __eq__, and __gt__.
class Node(object):
    """A Node class for storing data."""
    def __init__(self, data):
        """Construct a new node that stores some data."""
        self.data = data

    def __str__(self):
        return str(self.data)

    def __lt__(self, other):
        return self.data < other.data

    def __gt__(self, other):
        return self.data > other.data

    def __eq__(self, other):
        return self.data == other.data


class LinkedListNode(Node):
    """A Node class for linked lists. Inherits from the 'Node' class.
    Contains a reference to the next node in the list.
    """
    def __init__(self, data):
        """Construct a Node and initialize an attribute for
        the next node in the list.
        """
        Node.__init__(self, data)
        self.next = None

# Problems 2-4: Finish implementing this class.
class LinkedList(object):
    """Singly-linked list data structure class.
    The first node in the list is referenced to by 'head'.
    """
    def __init__(self):
        """Create a new empty linked list. Create the head
        attribute and set it to None since the list is empty.
        """
        self.head = None

    def add(self, data):
        """Create a new Node containing 'data' and add it to
        the end of the list.
        
        Example:
            >>> my_list = LinkedList()
            >>> my_list.add(1)
            >>> my_list.head.data
            1
            >>> my_list.add(2)
            >>> my_list.head.next.data
            2
        """
        new_node = LinkedListNode(data)
        if self.head is None:
            self.head = new_node
        else:
            current_node = self.head
            while current_node.next is not None:
                current_node = current_node.next
            current_node.next = new_node 
    
    # Problem 2: Implement the __str__ method so that a LinkedList instance can
    #   be printed out the same way that Python lists are printed.
    def __str__(self):
        """String representation: the same as a standard Python list.
        
        Example:
            >>> my_list = LinkedList()
            >>> my_list.add(1)
            >>> my_list.add(2)
            >>> my_list.add(3)
            >>> print(my_list)
            [1, 2, 3]
            >>> str(my_list) == str([1,2,3])
            True
        """
        L = [] 
        current_node = self.head
        while current_node is not None:
            L.append(current_node.data)
            current_node = current_node.next
        return str(L)

    # Problem 3: Finish implementing LinkedList.remove() so that if the node
    #   is not found, an exception is raised.
    def remove(self, data):
        """Remove the node containing 'data'. If the list is empty, or if the
        target node is not in the list, raise a ValueError with error message
        "<data> is not in the list."
        
        Example:
            >>> print(my_list)
            [1, 2, 3]
            >>> my_list.remove(2)
            >>> print(my_list)
            [1, 3]
            >>> my_list.remove(2)
            2 is not in the list.
            >>> print(my_list)
            [1, 3]
        """
        if self.head is None: # the list is empty
            raise ValueError(str(data) + " is not in the list.")
        elif self.head.data == data:
            self.head = self.head.next
        else:
            current_node = self.head
            while current_node.data != data:
                prev_node = current_node
                current_node = current_node.next
                if current_node is None:
                    raise ValueError(str(data) + " is not in the list.")
            prev_node.next = current_node.next
            current_node = None
            

    # Problem 4: Implement LinkedList.insert().
    def insert(self, data, place):
        """Create a new Node containing 'data'. Insert it into the list before
        the first Node in the list containing 'place'. If the list is empty, or
        if there is no node containing 'place' in the list, raise a ValueError
        with error message "<place> is not in the list."
        
        Example:
            >>> print(my_list)
            [1, 3]
            >>> my_list.insert(2,3)
            >>> print(my_list)
            [1, 2, 3]
            >>> my_list.insert(2,4)
            4 is not in the list.
        """
        A = LinkedListNode(data)
        if self.head is None:
            raise ValueError(str(place) + " is not in the list.")
        elif self.head.data == place:
            A.next = self.head
            self.head = A
        else:
            current_node = self.head
            while current_node.next.data != place:
                current_node = current_node.next
                if current_node is None:
                    raise ValueError(str(place) + " is not in the list")
            A.next = current_node.next
            current_node.next = A




class DoublyLinkedListNode(LinkedListNode):
    """A Node class for doubly-linked lists. Inherits from the 'Node' class.
    Contains references to the next and previous nodes in the list.
    """
    def __init__(self,data):
        """Initialize the next and prev attributes."""
        Node.__init__(self,data)
        self.next = None
        self.prev = None

# Problem 5: Implement this class.
class DoublyLinkedList(LinkedList):
    """Doubly-linked list data structure class. Inherits from the 'LinkedList'
    class. Has a 'head' for the front of the list and a 'tail' for the end.
    """
    def __init__(self):
        LinkedList.__init__(self)
        self.tail = None

    def add(self, data):
        new_node = DoublyLinkedListNode(data)
        if self.tail is None and self.head is None:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            new_node.prev = self.tail
            self.tail = new_node

    def remove(self, data):
        if self.head is None and self.tail is None:
            raise ValueError(str(data) + " is not in the list.")
        elif self.head.data == data:
            self.head = self.head.next
            self.head.prev = None
        elif self.tail.data == data:
            self.tail = self.tail.prev
            self.tail.next = None
        else:
            current_node = self.head
            while current_node.data != data:
                prev_node = current_node
                current_node = current_node.next
                if current_node.next is None:
                    raise ValueError(str(data) + " is not in the list.")
            next_node = current_node.next
            prev_node.next = next_node
            next_node.prev = prev_node
            current_node.next = None
            current_node.prev = None
            current_node = None

    def insert(self, data, place):
        A = DoublyLinkedListNode(data)
        if self.head is None and self.tail is None:
            raise ValueError(str(data) + " is not in the list.")
        elif self.head.data == place:
            self.head.prev = A
            A.next = self.head
            self.head = A
        else:
            current_node = self.head
            while current_node.next.data != place:
                current_node = current_node.next
                if current_node.next is None:
                    raise ValueError(str(data) + " is not in the list.")
            next_node = current_node.next
            A.prev = current_node
            A.next = next_node
            current_node.next = A
            next_node.prev = A



        
# Problem 6: Implement this class. Use an instance of your object to implement
# the sort_words() function in solutions.py.
class SortedLinkedList(DoublyLinkedList):
    """Sorted doubly-linked list data structure class."""

    # Overload add() and insert().
    def add(self, data):
        """Create a new Node containing 'data' and insert it at the
        appropriate location to preserve list sorting.
        
        Example:
            >>> print(my_list)
            [3, 5]
            >>> my_list.add(2)
            >>> my_list.add(4)
            >>> my_list.add(6)
            >>> print(my_list)
            [2, 3, 4, 5, 6]
        """
        new_node = DoublyLinkedListNode(data)
        if self.head is None and self.tail is None:
            self.head = new_node
            self.tail = new_node
        elif new_node.data <= self.head.data:
            new_node.next = self.head
            self.head.prev = new_node
            self.head = new_node
        else:
            current_node = self.head
            while new_node.data > current_node.data:
                if current_node.next is None:
                    break
                else:
                    prev_node = current_node
                    current_node = current_node.next
            if new_node.data > current_node.data:
                self.tail.next = new_node
                new_node.prev = self.tail
                self.tail = new_node
            else:
                prev_node.next = new_node
                current_node.prev = new_node
                new_node.next = current_node
                new_node.prev = prev_node

    def insert(self, *args):
        raise ValueError("insert() has been disabled for this class.")
# =============================== END OF FILE =============================== #

