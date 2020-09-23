from typing import List, Dict
from collections import defaultdict

class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def bstFromPreorder(preorder: List[int]) -> TreeNode:
    # O(n2), for every element you start from the root node again.
    root = TreeNode(preorder[0])

    def buildBST(item, cur):
        if item < cur.val:
            if not cur.left:
                cur.left = TreeNode(item)
            else:
                buildBST(item, cur.left)
        else:
            if not cur.right:
                cur.right = TreeNode(item)
            else:
                buildBST(item, cur.right)

    for item in preorder[1:]:
        buildBST(item, root)

    return root

def bstFromPreorder2(preorder: List[int]) -> TreeNode:
    root = TreeNode(preorder[0])

    def buildBST(item, cur):
        if not cur:
            return TreeNode(item)
        if item < cur.val:
            cur.left = buildBST(item, cur.left)
        else:
            cur.right = buildBST(item, cur.right)
        return cur
    for item in preorder[1:]:
        buildBST(item, root)

    return root


def bstFromPreorder3(preorder: List[int]) -> TreeNode:
    root = TreeNode(preorder[0])

    def buildBST(n, preorder, pos, cur, left, right):
        value = preorder[pos]
        if pos == n or value < left or value > right:
            return pos
        if value < cur.val:
            cur.left = TreeNode(value)
            pos += 1
            pos = buildBST(n, preorder, pos, cur.left, left, cur.val -1)

        if pos == n or value < left or value > right:
            return pos

        cur.right = TreeNode(value)
        pos += 1
        pos = buildBST(n, preorder, pos, cur.right, cur.val+1, right)
        return pos

    buildBST(len(preorder) -1, preorder, 1, root,  -float("Inf"), float("Inf"))

    return root


# print(bstFromPreorder3([8,5,1,7,10,12]))

def maxSubArray(nums: List[int]) -> int:
    # Kadane's algorithm
    # we basically want to disregard any negative numbers because they don't help with the sum
    # so we do a max(0, temp)
    result = temp = 0
    for x in nums:
        temp += x
        result = max(temp, result)
        temp = max(0, temp)
    return result

def maxSubArray2(nums: List[int]) -> int:
    # we can do it in another way
    # first we build a prefix sum
    pref = [nums[0]]
    for i in range(1, len(nums)):
        pref.append(nums[i] + nums[i-1])
    answer = -float("Inf")
    min_so_far = 0
    for i in range(len(nums)):
        x = pref[i]
        new = x - min_so_far
        answer = max(answer, new)
        min_so_far = min(min_so_far, pref[i])
    return answer

def maxSubArray3(nums: List[int]) -> int:
    # Binary search
    # you have 3 pos: max(left), max(right) and left + right. the biggest is the solution
    # this doesn't work currently
    def binary(nums, size):
        if size <= 1:
            return nums[0]

        m = size // 2
        left = binary(nums, m)
        right = binary(nums, size-m)
        left_sum = right_sum = -float("Inf")
        sum = 0
        for i in range(m, 0, -1):
            sum += nums[i]
            left_sum = max(sum, left_sum)

        sum = 0
        for i in range(m+1, size):
            sum += nums[i]
            right_sum = max(sum, right_sum)

        return max(left, right, left_sum+right_sum)

    return binary(nums, len(nums))

#print(maxSubArray3([-2,1,-3,4,-1,2,1,-5,4]))

def knightChessboard(src, dest) -> int:
    # it is an instance of finding the shortest path (geodesic path)
    # build the chessboard

    # matrix = []
    # for i in range(8):
    #     matrix.append([])
    #     for j in range(8):
    #         x = 0
    #         if i:
    #             x = matrix[i-1][-1] + 1
    #         matrix[i].append(j + x)
    # or better
    from _collections import deque
    matrix = []
    for i in range(64):
        if i % 8 == 0:
            matrix.append([])
        matrix[i // 8].append(i)

    def possibilities(x):
        r = x // 8
        c = x % 8
        result = []
        options = [(r - 2, c + 1), (r - 2, c - 1), (r - 1, c - 2), (r - 1, c + 2), (r + 2, c - 1), (r + 2, c + 1), (r + 1, c + 2), (r + 1, c - 2)]
        for i, j in options:
            if 0 <= i <= 7 and 0 <= j <= 7:
                result.append(matrix[i][j])

        return result

    seen = set()
    q = deque()
    q.append((src, 0))
    while q:
        item, moves = q.popleft()
        if item == dest:
            return moves
        seen.add(item)
        for x in possibilities(item):
            if x == dest:
                return moves + 1
            if x not in seen:
                q.append((x, moves+1))
    return -1



# print(knightChessboard(0, 1))
# print(knightChessboard(19, 36))
# print(knightChessboard(18, 1))
# print(knightChessboard(18, 22))

class Node:
    def __init__(self, data = None, next = None, prev = None):
        self.data = data
        self.next = next
        self.prev = prev

class DoublyLinkedList:

    def __init__(self, root = None, head = None):
        self.root = root
        self.head = head
        self.size = 0

    def push(self, item):
        new_node = Node(item)
        if self.head:
            new_node.prev = self.head
            self.head.next = new_node
            self.head = new_node
        else:
            self.head = new_node
            self.root = new_node
        self.size += 1

    def remove(self):
        if self.size:
            if self.size == 1:
                self.head = None
                self.root = None
            else:
                self.head = self.head.prev
                self.head.next = None
            self.size -= 1

    def push_left(self, item):
        new_node = Node(item)
        if self.root:
            new_node.next = self.root
            self.root.prev = new_node
            self.root = new_node
        else:
            self.root = new_node
            self.head = new_node
        self.size += 1

    def remove_left(self):
        if self.size:
            if self.size == 1:
                self.root = None
                self.head = None
            else:
                self.root = self.root.next
                self.root.prev = None
            self.size -= 1

    def listprint(self, node):
        while (node is not None):
            print(node.data),
            node = node.next


# dllist = DoublyLinkedList()
# dllist.push(12)
# dllist.push(8)
# dllist.push(62)
# dllist.push(52)
# dllist.remove()
# dllist.remove_left()
# dllist.listprint(dllist.root)

def subarraySum(nums: List[int], k: int) -> int:
    hashmap = {0: 1}
    cur = result = 0
    for x in nums:
        cur += x
        result += hashmap.get(cur - k, 0)
        hashmap[cur] = hashmap.get(cur, 0) + 1
    return result

# print(subarraySum([1,1,1], 2))

def LAMBCHOPsolution2(x, y):
    bottom = 0
    for i in range(1, x+1):
        bottom += i
    top = 0
    for i in range(x, x+y-1):
        top += i
    return bottom + top


def LAMBCHOPsolution(x, y):
    return str(((x + y - 1)*(x+y-2)) // 2 + x)


# print(LAMBCHOPsolution(1, 1))
# print(LAMBCHOPsolution(3, 2))
# print(LAMBCHOPsolution(2, 3))
# print(LAMBCHOPsolution(5, 10))


def canJump(nums: List[int]) -> bool:
    if len(nums) < 2:
        return True
    i = j = 0
    n = len(nums) - 1
    while i < n:
        if nums[i] == 0 and j == 0:
            return False
        j = max(nums[i], j)
        if j >= n - i :
            return True
        i += 1
        j -= 1
    return False

print(canJump([1,1,2,2,0,1,1]))
print(canJump([1,0,1,0]))
print(canJump([0,2,3]))
print(canJump([2,3,1,1,4]))
print(canJump([3,2,1,0,4]))