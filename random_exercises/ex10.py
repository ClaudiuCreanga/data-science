from typing import List, Dict
from collections import defaultdict
from heapq import heappush, heappop

def canJump(nums: List[int]) -> bool:
    # the intuition is to use 2 pointers
    j = 0
    t = len(nums) - 1
    for i in range(t + 1):
        j = max(nums[i], j)
        if nums[i] == 0 and j == 0:
            return False
        if j >= t - i:
            return True
        j -= 1

    return False


def canJump(nums: List[int]) -> bool:
    # here we can just use a curr value and while loop
    i = 0
    cur = nums[0]
    t = len(nums) - 1
    while cur:
        if cur >= t - i:
            return True
        i += 1
        cur -= 1
        cur = max(nums[i], cur)

    return False

# print(canJump([2,3,1,1,4])) # t
# print(canJump([3,2,1,0,4])) # f
# print(canJump([1,1,2,2,0,1,1])) # t


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def bstFromPreorder(preorder: List[int]) -> TreeNode:
    # 2 solutions
    def insert_bst(cur, item):
        if item < cur.val:
            if cur.left:
                insert_bst(cur.left, item)
            else:
                cur.left = TreeNode(item)
        else:
            if cur.right:
                insert_bst(cur.right, item)
            else:
                cur.right = TreeNode(item)

    root = TreeNode(preorder[0])
    for x in preorder[1:]:
        insert_bst(root, x)

    return root

def bstFromPreorder(preorder: List[int]) -> TreeNode:
    root = TreeNode(preorder[0])
    stack = [root]
    for item in preorder[1:]:
        if stack[-1].val > item:
            stack[-1].left = TreeNode(item)
            stack.append(stack[-1].left)
        else:
            while stack and stack[-1].val < item:
                last = stack.pop()
            last.right = TreeNode(item)
            stack.append(last.right)
    return root


#
# root = bstFromPreorder([8, 5, 1, 7, 10, 12])
# def preOrder(root):
#     if root:
#         print(root.val)
#         preOrder(root.left)
#         preOrder(root.right)
#
# preOrder(root)
# def inOrder(root):
#     if root:
#         preOrder(root.left)
#         print(root.val)
#         preOrder(root.right)
#
# inOrder(root)

def minSubArrayLen(s: int, nums: List[int]) -> int:
    # Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead
    ans = float("inf")
    i = j = cur = 0
    while i < len(nums):
        cur += nums[i]
        while cur >= s:
            ans = min(ans, i - j + 1)
            cur -= nums[j]
            j += 1
        i += 1

    return 0 if ans == float("inf") else ans

def minSubArrayLen(s: int, nums: List[int]) -> int:
    j = 0
    ans = float("inf")
    for i in range(len(nums)):
        s -= nums[i]
        while s <= 0:
            ans = min(ans, i - j + 1)
            s += nums[j]
            j += 1

    return 0 if ans == float("inf") else ans
#
# print(minSubArrayLen(2, [2])) # 1
# print(minSubArrayLen(7, [2,3,1,2,4,3])) # 2

def longestOnes(A: List[int], K: int) -> int:
    # Given an array A of 0s and 1s, we may change up to K values from 0 to 1.
    # Return the length of the longest (contiguous) subarray that contains only 1s.
    i = j = 0
    ans = 0
    while i < len(A):
        K += A[i] - 1
        while K < 0:
            K += 1 - A[j]
            j += 1
        ans = max(ans, i - j + 1)
        i += 1
    return ans

def longestOnes(A: List[int], K: int) -> int:
    # Given an array A of 0s and 1s, we may change up to K values from 0 to 1.
    # Return the length of the longest (contiguous) subarray that contains only 1s.
    j = 0
    for i in range(len(A)):
        K += A[i] - 1
        if K < 0:
            K += 1 - A[j]
            j += 1
    return i - j + 1
#
# print(longestOnes([0, 0], 2)) # 2
# print(longestOnes([1], 2)) # 1
# print(longestOnes([1,0,0, 0], 2)) # 3
# print(longestOnes([1,0,0], 2)) # 3
# print(longestOnes([1,1,1,0,0,0], 2)) # 5
# print(longestOnes([1,1,1,0,0,0,1,1,1,1,0], 2)) # 6
# print(longestOnes([0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], 3)) # 10

def totalFruit(tree: List[int]) -> int:
    # find the longest contiguos array that contains only 2 distinct numbers
    # TODO test that it works
    ans = j = 0
    count = {}
    for i, x in enumerate(tree):
        count[x] = count.get(x, 0) + 1
        if len(count.keys()) <= 2:
            ans = max(ans, i - j + 1)
        else:
            t = tree[j]
            if count[t] == 1:
                count.pop(t)
            else:
                count[t] = count.get(t) - 1
            j += 1

    return ans

class Window:
    def __init__(self):
        self.data = {}
        self.size = 0

    def add(self, item):
        if item in self.data:
            self.data[item] += 1
        else:
            self.data[item] = 1
            self.size += 1

    def remove(self, item):
        if item in self.data:
            if self.data[item] == 1:
                self.data.pop(item)
                self.size -= 1
            else:
                self.data[item] -= 1


def totalFruit(tree: List[int]) -> int:
    # find the longest contiguos array that contains only 2 distinct numbers
    # TODO check if it works
    ans = j = 0
    count = Window()
    for i, x in enumerate(tree):
        count.add(x)
        while count.size > 2:
            t = tree[j]
            count.remove(t)
            j += 1
        ans = max(ans, i - j + 1)

    return ans

def totalFruit(tree: List[int]) -> int:
    # find the longest contiguos array that contains only 2 distinct numbers
    j = 0
    count = Window()
    for i, x in enumerate(tree):
        count.add(x)
        if count.size > 2:
            t = tree[j]
            count.remove(t)
            j += 1

    return i - j + 1
#
# print(totalFruit([1,2,1])) # 3
# print(totalFruit([0,1,2,2])) # 3
# print(totalFruit([3,3,3,1,2,1,1,2,3,3,4])) # 5
# print(totalFruit([1,2,3,2,2])) # 4

class CountWindow:
    def __init__(self):
        self.data = {k: 0 for k in "QWER"}

    def add(self, item):
        self.data[item] = self.data.get(item, 0) + 1

    def remove(self, item):
        self.data[item] -= 1

    def is_balanced(self, threshold):
        return max([self.data[x] for x in "QWER"]) <= threshold

def balancedString(s: str) -> int:
    # You are given a string containing only 4 kinds of characters 'Q', 'W', 'E' and 'R'.
    # minimum length of string that replaces the substring to be balanced
    # TODO check it works
    count = CountWindow()
    right_balance = len(s) // 4
    for x in s:
        count.add(x)

    j = 0
    ans = float("inf")
    if count.is_balanced(right_balance):
        return 0
    for i, x in enumerate(s):
        count.remove(x)
        while count.is_balanced(right_balance):
            ans = min(i - j + 1, ans)
            count.add(s[j])
            j += 1

    return ans


# print(balancedString("WWEQERQWQWWRWWERQWEQ")) # 4
# print(balancedString("QWQQEEER")) #2
# print(balancedString("QWER")) #0
# print(balancedString("QQER")) #1
# print(balancedString("QQQR")) #2
# print(balancedString("QQQQ")) #3

def countAndSay(n: int) -> str:
    # TODO try if it works
    def add_item(i, cur, n):
        if i == n:
            return cur
        j = 0
        ans = ""
        last = cur[0]
        for x in cur:
            if x == last:
                j += 1
            else:
                ans += str(j)
                ans += last
                last = x
                j = 1
        ans += str(j)
        ans += last
        return add_item(i+1, ans, n)

    return add_item(1, "1", n)

# print(countAndSay(1)) # 1
# print(countAndSay(2)) # 11
# print(countAndSay(3)) # 21
# print(countAndSay(4)) # 1211

def numberOfSubarrays(nums: List[int], k: int) -> int:
    # Given an array of integers nums and an integer k. A continuous subarray is called nice if there are k odd numbers on it.
    # Return the number of nice sub-arrays.
    for i, x in enumerate(nums):
        nums[i] = x & 1

    pref = 0
    count = {0: 1}
    ans = 0
    for x in nums:
        pref += x
        ans += count.get(pref - k, 0)
        count[pref] = count.get(pref, 0) + 1

    return ans


def numberOfSubarrays(nums: List[int], k: int) -> int:
    for i, x in enumerate(nums):
        nums[i] = x & 1

    def atMost(nums, k):
        j = 0
        ans = 0
        for i, x in enumerate(nums):
            k -= x
            while k < 0:
                k += nums[j]
                j += 1
            ans += i - j + 1

        return ans

    return atMost(nums, k) - atMost(nums, k - 1)


def numberOfSubarrays(nums: List[int], k: int) -> int:
    # TODO what's the intuition for ans +=  cur ??
    j = cur = ans = 0
    for i, x in enumerate(nums):
        if x & 1: # if odd
            k -= 1
            cur = 0
        while k == 0:
            k += nums[j] & 1 # add 1 if odd
            j += 1
            cur += 1
        ans += cur

    return ans

# print(numberOfSubarrays([1,1,2,1,1], 3)) # 2
# print(numberOfSubarrays([2,4,6], 1)) # 0
#  print(numberOfSubarrays([2,2,2,1,2,2,1,2,2,2], 2)) # 16


def subarraysWithKDistinct(A: List[int], K: int) -> int:
    # Given an array A of positive integers, call a (contiguous, not necessarily distinct) subarray of A good if the number of different integers in that subarray is exactly K
    def atMost(A, K):
        ans = j = 0
        count = Window()
        for i in range(len(A)):
            count.add(A[i])
            while count.size > K:
                count.remove(A[j])
                j += 1
            ans += i - j + 1
        return ans

    return atMost(A, K) - atMost(A, K-1)

def shortestSubarray(A: List[int], K: int) -> int:
    # Return the length of the shortest, non-empty, contiguous subarray of A with sum at least K
    from collections import deque
    q = deque()
    pass

# print(shortestSubarray([-28,81,-20,28,-29], 89)) #3
# print(shortestSubarray([84,-37,32,40,95], 167)) #3
# print(shortestSubarray([1], 1)) #1
# print(shortestSubarray([1, 2], 4)) #-1
# print(shortestSubarray([15, 20, 7, 8, 50], 40)) #1
# print(shortestSubarray([15, 20, 7, 8, 50], 4)) #1
# print(shortestSubarray([1, 2, -1, 2, 3], 4)) #2
# print(shortestSubarray([2,-1,2], 3)) #3


class RecentCounter:

    def __init__(self):
        self.requests = []

    def binary_search(self, l, h, t):
        if l <= h:
            m = l + (h - l) // 2
            if self.requests[m] < t:
                return self.binary_search(m+1, h, t)
            elif self.requests[m] > t:
                return self.binary_search(l, m-1, t)
            else:
                return m
        return l

    def ping(self, t: int) -> int:
        self.requests.append(t)
        bound = self.binary_search(0, len(self.requests) - 1, t-3000)
        return len(self.requests) - bound

class RecentCounter:

    def __init__(self):
        from collections import deque
        self.requests = deque()

    def ping(self, t: int) -> int:
        self.requests.append(t)
        while self.requests[0] < t - 3000:
            self.requests.popleft()
        return len(self.requests)

# Your RecentCounter object will be instantiated and called as such:
obj = RecentCounter()
print(obj.ping(1))
print(obj.ping(100))
print(obj.ping(3001))
print(obj.ping(3002))
