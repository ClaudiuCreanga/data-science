from typing import List

def rotLeft(a, d):
    for _ in range(d):
        a.append(a[0])
        del a[0]

    return a

#print(rotLeft([1, 2, 3, 4, 5], 3))

def minimumBribes(q):
    result = 0
    flag = False
    for passing in range(len(q)-1, 0, -1):
        if flag:
            result = "Too chaotic"
            break
        bribed = 0
        for i in range(passing):
            if q[i] > q[i+1]:
                q[i], q[i+1] = q[i+1], q[i]
                result += 1
                bribed += 1
                if bribed > 2:
                    flag = True
                    break
            else:
                bribed = 0

    return result

#print(minimumBribes([5, 1, 2, 3, 7, 8, 6, 4]))


def bubblesort(q):
    for passing in range(len(q)-1, 0, -1):
        for i in range(passing):
            if q[i] > q[i+1]:
                q[i], q[i+1] = q[i+1], q[i]
    return q

#print(bubblesort([5,1,4,2,8]))

def two_strings(a, b):
    edits = 0
    if abs(len(b) - len(a)) > 1:
        return "the words don't match"
    for index, letter in enumerate(a):
        if index <= len(b) - 1:
            if b[index] != letter:
                edits += 1
                if len(b) - len(a) == 1:
                    b = b[:index] + b[index+1:]
                elif len(b) - len(a) == -1:
                    b = b[:index] + letter + b[index:]
        else:
            edits += 1
    if edits > 1:
        return "the words don't match"

    return "the words match"

# print(two_strings("abc", "abc"))
# print(two_strings("adc", "abc"))
# print(two_strings("abc", "adc"))
# print(two_strings("bbc", "adc"))
# print(two_strings("abc", "ddc"))
# print(two_strings("bc", "dbc"))
# print(two_strings("abc", "bc"))
# print(two_strings("abc", "c"))
# print(two_strings("c", "c"))
# print(two_strings("ac", "aac"))
# print(two_strings("ac", "aaac"))
# print(two_strings("aaaaac", "aaac"))
# print(two_strings("aaaaac", "aaaac"))
# print(two_strings("aaac", "aaaac"))
# print(two_strings("abcde", "accde"))
# print(two_strings("abcde", "abcdf"))
# print(two_strings("abcdef", "abcde"))
# print(two_strings("geek", "geeks"))
# print(two_strings("m", ""))
# print(two_strings("", "m"))

def find_two_strings(a, b):
    if abs(len(a) - len(b)) > 1:
        return False
    i = 0
    j = 0
    edits = 0
    while (i < len(a) and j < len(b)):
        if a[i] != b[j]:
            edits += 1
            if len(a) > len(b):
                i+=1
            elif len(b) < len(a):
                j+=1
            else:
                i += 1
                j += 1
        else:
            i += 1
            j += 1
    if i < len(a) or j < len(b):
        edits += 1

    return not edits > 1
#
# print(find_two_strings("abc", "abc"))
# print(find_two_strings("adc", "abc"))
# print(find_two_strings("abc", "adc"))
# print(find_two_strings("bbc", "adc"))
# print(find_two_strings("abc", "ddc"))
# print(find_two_strings("bc", "dbc"))
# print(find_two_strings("abc", "bc"))
# print(find_two_strings("abc", "c"))
# print(find_two_strings("c", "c"))
# print(find_two_strings("ac", "aac"))
# print(find_two_strings("ac", "aaac"))
# print(find_two_strings("aaaaac", "aaac"))
# print(find_two_strings("aaaaac", "aaaac"))
# print(find_two_strings("aaac", "aaaac"))
# print(find_two_strings("abcde", "accde"))
# print(find_two_strings("abcde", "abcdf"))
# print(find_two_strings("abcdef", "abcde"))
# print(find_two_strings("geek", "geeks"))
# print(find_two_strings("m", ""))
# print(find_two_strings("", "m"))

def build_spiral_array(a):
    result = []
    rows = len(a)
    columns = len(a[0])
    row_index = 0
    column_index = 0
    while row_index < rows and column_index < columns:

        for i in range(column_index, columns):
            result.append(a[row_index][i])

        row_index += 1

        for i in range(row_index, rows):
            result.append(a[i][columns - 1])

        columns -= 1

        if row_index < rows:
            for i in range(columns - 1, column_index - 1, -1):
                result.append(a[rows - 1][i])

            rows -= 1

        if column_index < columns:
            for i in range(rows - 1, row_index - 1, -1):
                result.append(a[i][column_index])

            column_index += 1

    return " ".join(map(str, result))

#
# print(build_spiral_array([[1,    2,   3,   4],[5,    6,   7,   8],[9,   10,  11,  12],[13,  14,  15,  16]]))
# print(build_spiral_array([ [1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12],[13, 14, 15, 16, 17, 18]]))
# print(build_spiral_array([ [1,2,3], [8,9,4], [7,6,5]]))


def look_and_say(n):
    a = {
        "1": "11",
        "11": "21",
        "2": "12",
        "111": "3",
        "22": "22",
        "3": "13"

    }
    result = []
    count = 0

    def rec(x, count):
        if count > n:
            return
        count += 1
        r = ""
        i = 0
        while i < len(x):
            if i + 1 < len(x):
                if x[i] == x[i+1]:
                    r += a[x[i] + x[i+1]]
                    i += 1
                else:
                    r += (a[x[i]])
            else:
                r += a[x[i]]

            i += 1

        result.append(r)
        rec(r, count)

    rec(["1"], count)
    return result

#print(look_and_say(9))


def countnndSay(n):
    if n == 1:
        return "1"
    if n == 2:
        return "11"

    s = "11"

    for i in range(3, n+1):
        count = 1
        s += "$"
        tmp = ""
        l = len(s)

        for j in range(1, l):
            current = s[j]
            previous = s[j-1]
            if current != previous:
                tmp += str(count) + previous
                count = 1
            else:
                count += 1

        s = tmp
    return s

#print(countnndSay(9))

def myregex(pattern, mystring):
    i = 0
    while i < len(mystring):
        if mystring[i] == pattern[i]:
            return myregex(pattern[i+1:], mystring[i+1:])
        elif pattern[i] == ".":
            return myregex(pattern[i+1:], mystring[i+1:])
        else:
            return False
    return True




#print(myregex("dasss","dasss"))

def bonetrousle(n, k, b):
    minValue = b * (b + 1) / 2
    maxValue = b * (2 * k - b + 1)/ 2
    if minValue > n or maxValue < n:
        return [-1]
    if b == 1:
        return [n]
    result = list(range(1, b+1))
    r = int((n-minValue) % b)
    q = int((n -minValue) // b)

    for index, item in enumerate(result):
        result[index] += q
        if index > len(result) - r - 1 and index <= len(result) - 1:
            result[index] += 1
    print(len(set(result)) == b)
    return result



# print(bonetrousle(12, 8, 3))
# print(bonetrousle(10, 3, 3))
# print(bonetrousle(22,7,6))
# print(bonetrousle(38, 10, 7))
# print(bonetrousle(809880023823305331, 906161411045895284, 52920))


def bonetrousle2(n, k, b):
    init = b * (b + 1) // 2
    extra = (n - init) // b + 1
    over = init + extra * b - n
    answer = list(range(1, b + 1))
    answer = [i + extra - 1 for i in answer[:over]] + [i + extra for i in answer[over:]]
    if answer[-1] <= k and answer[0] >= 1:
        return answer
    else:
        return [-1]



#print(bonetrousle2(12, 8, 3))


def reverseStr(s: str, k: int) -> str:
    result = ""
    reverse = True
    i = 0
    while i < len(s):
        temp = s[i:i + k]
        if reverse:
            result += temp[::-1]
        else:
            result += temp
        reverse = not reverse
        i += k

    return result


#print(reverseStr("abcdefg", 2))

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

print(find_preorder(["D","E","B","F","C","A"]))


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

print(binaryTreePathsGood(root))

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

print(binaryTreePathsGood2(root))

