from anytree import Node, RenderTree

udo = Node("Udo")
marc = Node("Marc", parent=udo)
lian = Node("Lian", parent=marc)
dan = Node("Dan", parent=udo)
jet = Node("Jet", parent=dan)
jan = Node("Jan", parent=dan)
joe = Node("Joe", parent=marc)
clau = Node("clau", parent=lian)

for pre, fill, node in RenderTree(udo):
    print("%s%s" % (pre, node.name))

def traversal(node):
    if node:
        print(node.name)
        for v in node.children:
            traversal(v)

traversal(udo)
