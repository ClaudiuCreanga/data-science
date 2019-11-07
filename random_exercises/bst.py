from typing import List

# DFS can be implemented with a while loop and a stack or with a recursion
# BFS can be implemented with a queue

def criticalConnections(n: int, connections: List[List[int]]) -> List[List[int]]:
    graph = {}
    for node_1, node_2 in connections:  # undirected, so connect in both direction
        graph.setdefault(node_1, []).append(node_2)
        graph.setdefault(node_2, []).append(node_1)

    low = [None] * n
    dist = [None] * n

    def dfs(node, distance):
        if low[node] is not None:
            return low[node]
        low[node] = dist[node] = distance
        for connected_node in graph[node]:
            if dist[connected_node] is not None and dist[connected_node] == distance - 1:  # means this is parent
                continue  # ignore direct parent
            low[node] = min(dfs(connected_node, distance + 1), low[node])
        return low[node]

    dfs(0, 0)

    return [[node_1, node_2]
            for node_1, node_2 in connections
            if dist[node_1] < low[node_2] or dist[node_2] < low[node_1]]

# print(criticalConnections(6, [[0,1],[1,2],[2,0],[1,3],[3,4],[4,5],[5,3]]))

class Node:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

    def __str__(self):
        return f'<{self.val}, {self.left}, {self.right}>'

root = Node(1)
root.left = Node(2)
root.right = Node(3)
root.left.left = Node(4)
root.left.right = Node(5)

def printPreorder(root):
    if root:
        print(root.val)

        printPreorder(root.left)
        printPreorder(root.right)

#printPreorder(root)

root3 = Node(1)
root3.left = None
root3.right = Node(2)
#printPreorder(root)


def isSameTree(p, q) -> bool:
    result = []

    def traverse(root):
        if root:
            result.append(root.val)

            traverse(root.left)
            traverse(root.right)

    traverse(p)
    left = result
    result = []
    traverse(q)
    right = result

    return left == right

root2 = Node(1)
root2.left = Node(2)

#print(isSameTree(root3,root2))


def printInorder(root):
    if root:
        # First recur on left child
        printInorder(root.left)

        # then print the data of node
        print(root.val),

        # now recur on right child
        printInorder(root.right)

#printInorder(root)


def isSameTree2(self, p, q) -> bool:
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    return isSameTree2(p.left, q.left) and isSameTree2(p.right, q.right)

#print(isSameTree2(root3,root2))

def binaryTreePaths(root) -> List[str]:
    result = []
    temp = []
    def traverse(root):
        nonlocal temp
        if root:
            temp.append(root.val)
            traverse(root.left)
            traverse(root.right)
        elif len(temp):
            result.append("->".join(map(str, temp)))
            temp = []
    traverse(root)

    return result

#print(binaryTreePaths(root))

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}

def dfs(graph, start):
    visited = set()
    stack = [start]
    while stack:
        temp = stack.pop()
        if temp not in visited:
            visited.add(temp)
            stack.extend(graph[temp] - visited)
    return visited

#print(dfs(graph, 'A')) # {'E', 'D', 'F', 'A', 'C', 'B'}

def dfs_recursive(graph, start, visited=None):
    if visited == None:
        visited = set()
    if start not in visited:
        visited.add(start)
        for item in graph[start] - visited:
            dfs_recursive(graph, item, visited)
    return visited

#print(dfs_recursive(graph, 'A'))

graph_find = Node(5)
graph_find.left = Node(2)
graph_find.right = Node(8)
graph_find.left.left = Node(1)
graph_find.left.right = Node(4)
graph_find.right.left = Node(6)

def dfs_find(root, item):
   stack = [root]
   while stack:
       temp = stack.pop()
       if temp.val == item:
           return True
       else:
           if temp.left:
               stack.append(temp.left)
           if temp.right:
               stack.append(temp.right)
   return False


#print(dfs_find(graph_find, 9))

def find_preorder(postorder):
    pass

#print(find_preorder(["D","E","B","F","C","A"]))


def construct_bst_preorder(preorder):
    root = Node(preorder[0])
    left = []
    right = []
    direction = "left"
    for item in preorder[1:]:
        if item > root:
            direction = "right"
        if direction == "left":
            left.append(item)
        if direction == "right":
            right.append(item)
    root.left = Node(left[0])
    root.right = Node(right[0])

from collections import deque

data = [3,5,2,1,4,6,7,8,9,10,11,12,13,14]
n = iter(data)
tree = Node(next(n))
fringe = deque([tree])
while True:
    head = fringe.popleft()
    try:
        head.left = Node(next(n))
        fringe.append(head.left)
        head.right = Node(next(n))
        fringe.append(head.right)
    except StopIteration:
        break

#print(tree)


def binaryTreePathsGood( root) -> List[str]:
    paths = []

    def dfs(root, path=""):
        if root:
            if path != "":
                path += "->"
            path += str(root.val)
            if not root.left and not root.right:
                paths.append(path)
            dfs(root.left, path)
            dfs(root.right, path)

    dfs(root)
    return paths

#print(binaryTreePathsGood(root))

def binaryTreePathsGood2( root) -> List[str]:
    paths = []

    def dfs(root, path=[]):
        if root:
            path.append(root.val)
            if not root.left and not root.right:
                paths.append("->".join(map(str, path)))
            dfs(root.left, path)
            dfs(root.right, path)
            path.pop()

    dfs(root)
    return paths

#print(binaryTreePathsGood2(root))

def main(root):
    result = []
    current_level = [root]
    while len(current_level): # sort of bfs
        next_level = []
        for x in current_level:
            if x.left:
                next_level.append(x.left)
            if x.right:
                next_level.append(x.right)
        result.append([x.val for x in current_level if x])
        current_level = next_level
    return result

# mda = main(graph_find)
# for x in mda:
#     print("new list")
#     for i in x:
#         print(i)

def getLevelUtil(node, data, level):
    if (node == None):
        return 0

    if (node.val == data):
        return level

    downlevel = getLevelUtil(node.left,
                             data, level + 1)
    if (downlevel != 0):
        return downlevel

    downlevel = getLevelUtil(node.right,
                             data, level + 1)
    return downlevel

# graph_find = Node(5)
# graph_find.left = Node(2)
# graph_find.right = Node(8)
# graph_find.left.left = Node(1)
# graph_find.left.right = Node(4)
# graph_find.right.left = Node(6)
#print(getLevelUtil(graph_find, 4, 1))

def find_level(node, t, l, found = 0):
    if node:
        if node.val == t:
            print(l)
        find_level(node.left, t, l + 1, found)
        find_level(node.right, t, l + 1, found)


#print(find_level(graph_find, 4, 1))
