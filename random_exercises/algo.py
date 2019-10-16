# stacks

s = []
s.append("a")
s.append("b")
s.append("c")

last = s.pop()
print(last)
print(s)

class Stack:

    def __init__(self):
        self.items = []

    def add(self, item):
        self.items.append(item)

    def get(self):
        return self.items.pop()

stack = Stack()

stack.add("a")
stack.add("b")
print(stack.get())
print(stack.items)
