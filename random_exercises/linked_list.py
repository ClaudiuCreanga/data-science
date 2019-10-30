class Node:
    def __init__(self, d, n = None):
        self.data = d
        self.next_node = n

    def get_next(self):
        return self.next_node

    def set_next(self, n):
        self.next_node = n

    def get_data(self):
        return self.data

    def set_data(self,d):
        self.data = d

    def has_next(self):
        if self.get_next() is None:
            return False
        return True

    def to_string(self):
        return "Node value: " + str(self.data)

class LinkedList:
    def __init__(self, r = None):
        self.root = r
        self.size = 0

    def get_size(self):
        return self.size

    def add(self, d):
        new_node = Node(d, self.root)
        self.root = new_node
        self.size += 1

    def remove(self, d):
        this_node = self.root
        prev_node = None

        while this_node:
            if this_node.get_data() == d:
                next_node = this_node.get_next()
                if prev_node:
                    prev_node.set_next(next_node)

                else:
                    self.root = next_node
                self.size -= 1
                return True
            else:
                prev_node = this_node
                this_node = this_node.get_next()

        return False

    def find (self, d):
        this_node = self.root
        while this_node:
            if this_node.get_data() == d:
                return d
            else:
                this_node = this_node.get_next()
        return None

    def print_list(self):
        if self.root is None:
            return
        current = self.root
        print(current.to_string())
        while current.has_next():
            current = current.get_next()
            print(current.to_string())

myList = LinkedList()
myList.add(5)
myList.add(6)
myList.add(7)
myList.add(8)
myList.print_list()

def sort(l: LinkedList):
    new = []
    current = l.root
    new.append(current)
    while current.has_next():
        current = current.get_next()
        new.append(current)

    new = sorted(new, key= lambda node: node.get_data(), reverse=True)
    newl = LinkedList()
    for node in new:
        newl.add(node.get_data())
    return newl

da = sort(myList)
da.print_list()
