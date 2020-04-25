from typing import List, Dict
from collections import defaultdict

def longestPalindrome(s: str) -> str:
    j = len(s)

    def look_left_right(l, r):
        while l >= 0 and r < j:
            if s[l] == s[r]:
                l -= 1
                r += 1
            else:
                return l+1, r-1, r-l-2
        return l+1, r-1, r-l-2

    result = (0,0,0)
    for i in range(len(s)):
        res = look_left_right(i-1,i+1)
        if res[2] > result[2]:
            result = res
        res = look_left_right(i, i+1)
        if res[2] > result[2]:
            result = res

    return s[result[0]:result[1]+1]

# print(longestPalindrome(""))
# print(longestPalindrome("acc"))
# print(longestPalindrome("babad"))
# print(longestPalindrome("adam"))
# print(longestPalindrome("addam"))

def isHappy(n: int) -> bool:
    seen = set()
    def calculate(n):
        n = str(n)
        res = 0
        for i in n:
            res += int(i)**2
        if res == 1:
            return True
        else:
            if res in seen:
                return False
            else:
                seen.add(res)
                return calculate(res)

    return calculate(n)

# print(isHappy(19))

def maxSubArray(nums: List[int]) -> int:
    maxC = res = nums[0]
    for i in range(1, len(nums)):
        maxC = max(nums[i], maxC + nums[i])
        res = max(maxC, res)

    return res

#print(maxSubArray([-2,1,-3,4,-1,2,1,-5,4]))


def countElements(arr: List[int]) -> int:
    count = 0
    seen = set()
    for x in arr:
        seen.add(x)
    for x in arr:
        if x+1 in seen:
            count += 1

    return count

# print(countElements([1,2,3]))
# print(countElements([1,1,3,3,5,5,7,7]))


def groupAnagrams(strs: List[str]) -> List[List[str]]:
    res = defaultdict(list)
    for item in strs:
        alpha = tuple(sorted(item))
        res[alpha].append(item)

    return res.values()


#print(groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))


def backspaceCompare(S: str, T: str) -> bool:
    i = len(S) - 1
    j = len(T) - 1
    count_j = 0
    count_i = 0
    while i >= 0 and j >= 0:
        while count_i and i >= 0 and S[i] != "#":
            i -= 1
            count_i -= 1
        while count_j and j >= 0 and T[j] != "#":
            j -= 1
            count_j -= 1
        if S[i] == "#":
            while i >= 0 and S[i] == "#":
                i -= 1
                count_i += 1
            continue
        if T[j] == "#":
            while j >= 0 and T[j] == "#":
                j -= 1
                count_j += 1
            continue
        if i >= 0 and j >= 0:
            if S[i] == T[j]:
                i -= 1
                j -= 1
                continue
            else:
                return False

    if i >=0:
        while i >= 0:
            while count_i and i >= 0 and S[i] != "#":
                i -= 1
                count_i -= 1
            if S[i] == "#":
                while i >= 0 and S[i] == "#":
                    i -= 1
                    count_i += 1
                continue
            return S[:i+1] == ""

    if j >=0:
        while j >= 0:
            while count_j and j >= 0 and T[j] != "#":
                j -= 1
                count_j -= 1
            if T[j] == "#":
                while j >= 0 and T[j] == "#":
                    j -= 1
                    count_j += 1
                continue
            return T[:j+1] == ""

    return True


def backspaceCompare2(S: str, T: str) -> bool:
    def build(s):
        stack = []
        for l in s:
            if l != "#":
                stack.append(l)
            elif stack:
                stack.pop()
        return stack

    return build(S) == build(T)


# print(backspaceCompare2("j##yc##bs#srqpfzantto###########i#mwb", "j##yc##bs#srqpf#zantto###########i#mwb"))
# print(backspaceCompare2("y#fo##f", "y#fx#o##f"))
# print(backspaceCompare2("bbbextm", "bbb#extm"))
# print(backspaceCompare2("ab#c", "ad#c"))
# print(backspaceCompare2("ab##", "ll"))
# print(backspaceCompare2("a#c", "b"))
# print(backspaceCompare2("baa##", "b"))
# print(backspaceCompare2("xywrrmp", "xywrrmu#p"))
# print(backspaceCompare2("bxj##tw", "bxo#j##tw"))


class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []
        self.data_min = []

    def push(self, x: int) -> None:
        self.data.append(x)
        if self.data_min:
            if x < self.data_min[-1]:
                self.data_min.append(x)
            else:
                self.data_min.append(self.data_min[-1])
        else:
            self.data_min.append(x)

    def pop(self) -> None:
        if self.data_min:
            self.data_min.pop()
        if self.data:
            return self.data.pop()

    def top(self) -> int:
        if self.data:
            return self.data[-1]

    def getMin(self) -> int:
        if self.data_min:
            return self.data_min[-1]

# minStack = MinStack()
# minStack.push(-2)
# minStack.push(0)
# minStack.push(-3)
# print(minStack.getMin())
# minStack.pop()
# print(minStack.top())
# print(minStack.getMin())


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None


def middleNode(head: ListNode) -> ListNode:
    length = 0
    original_head = head
    while head.next:
        length += 1
        head = head.next
    if length == 0:
        return head

    if length % 2:
        m = length // 2
    else:
        m = length // 2 + 1

    counter = 0
    head = original_head
    while head.next:
        counter += 1
        if counter == m:
            return head.next
        head = head.next


def middleNode2(head: ListNode) -> ListNode:
    slow = fast = head
    while fast and fast.next:
        fast = fast.next.next
        slow = slow.next
    return slow


def stringShift(s: str, shift: List[List[int]]) -> str:
    l = list(s)
    for dir, amount in shift:
        if dir == 0:
            while amount:
                l.append(l.pop(0))
                amount -= 1
        else:
            while amount:
                l.insert(0, l.pop())
                amount -= 1

    return "".join(l)


# print(stringShift("abc", [[0,1],[1,2]]))
# print(stringShift("abcdefg", [[1,1],[1,1],[0,2],[1,3]]))


def stringShift2(s: str, shift: List[List[int]]) -> str:
    count = 0
    for dir, amount in shift:
        if dir == 0:
            count += amount
        else:
            count -= amount

    l = list(s)
    if count > 0 :
        while count > 0:
            l.append(l.pop(0))
            count -= 1
    else:
        while count < 0:
            l.insert(0, l.pop())
            count += 1

    return "".join(l)


# print(stringShift2("abc", [[0,1],[1,2]]))
# print(stringShift2("abcdefg", [[1,1],[1,1],[0,2],[1,3]]))


def findMaxLength(nums: List[int]) -> int:
    count = 0
    seen = {}
    result = 0
    for i, x in enumerate(nums):
        if x == 0:
            count -= 1
        else:
            count += 1
        if count == 0:
            result = max(result, i+1)
        if count in seen:
            result = max(result, i-seen[count])
        else:
            seen[count] = i

    return result


# print(findMaxLength([1,0,0,1,0,0,0]))
# print(findMaxLength([0,0,0,1,0,1]))
# print(findMaxLength([0,0,0,1,1,1]))
# print(findMaxLength([0,1,0]))
# print(findMaxLength([0,1,1]))


def lastStoneWeight(stones: List[int]) -> int:
    def smash(stones):
        y = max(stones)
        stones.remove(y)
        x = max(stones)
        stones.remove(x)
        change = y-x
        if change:
            stones.append(change)
        if len(stones) > 1:
            return smash(stones)
        return stones
    if len(stones) > 1:
        stones = smash(stones)
    if len(stones) == 1:
        return stones[0]
    return 0

# print(lastStoneWeight([2,7,4,1,8,1]))


def lastStoneWeight2(stones: List[int]) -> int:
    import heapq
    stones = [-x for x in stones]
    heapq.heapify(stones)
    def smash(stones):
        y = heapq.heappop(stones)
        x = heapq.heappop(stones)
        change = y-x
        if change:
            heapq.heappush(stones, change)
        if len(stones) > 1:
            return smash(stones)
        return stones
    if len(stones) > 1:
        stones = smash(stones)
    if len(stones) == 1:
        return -stones[0]
    return 0

# print(lastStoneWeight2([2,7,4,1,8,1]))


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def maxDepthBinaryTree(root) -> int:
    if not root:
        return -1
    else:
        left_height = maxDepthBinaryTree(root.left)
        right_height = maxDepthBinaryTree(root.right)

        return max(left_height, right_height) + 1


def diameterOfBinaryTree(root: TreeNode) -> int:
    if not root:
        return -1
    else:
        rootDiameter = maxDepthBinaryTree(root.left) + maxDepthBinaryTree(root.right) + 1
        leftDiameter = diameterOfBinaryTree(root.left)
        rightDiameter = diameterOfBinaryTree(root.right)

        return max(rootDiameter, leftDiameter, rightDiameter)


def diameterOfBinaryTree2(root: TreeNode) -> int:
    ans = 0
    def depth(root: TreeNode) -> int:
        nonlocal ans
        if not root:
            return 0
        left, right = depth(root.left), depth(root.right)
        ans = max(ans, left+right)
        return 1 + max(left, right)

    depth(root)
    return ans


# root = TreeNode(1)
# root.left = TreeNode(2)
# root.right = TreeNode(3)
# root.left.left = TreeNode(4)
# root.left.right = TreeNode(5)
# print(diameterOfBinaryTree2(root))


def checkValidString(s: str) -> bool:
    possibilities = ["(", ")", ""]
    l = list(s)
    d = len(l)
    def check(l, start, count):
        for i in range(start, d):
            if count < 0 or count > d - i:
                return False
            if l[i] == "":
                continue
            elif l[i] == "(":
                count += 1
            elif l[i] == ")":
                count -= 1
            else:
                for j, p in enumerate(possibilities):
                    l[i] = p
                    if check(l, i, count):
                        return True
                l[i] = "*"  # Backtracking
        if count == 0:
            return True
        else:
            return False

    return check(l, 0, 0)


def checkValidString2(s: str) -> bool:
    cmin = cmax = 0
    for x in s:
        if x == "(":
            cmin += 1
            cmax += 1
        elif x == ")":
            cmin -= 1
            cmax -= 1
        else:
            cmin = max(cmin-1, 0)
            cmax += 1
        if cmax < 0:
            return False
    return cmin == 0


# print(checkValidString(""))
# print(checkValidString("()"))
# print(checkValidString("())"))
# print(checkValidString("(*))"))
# print(checkValidString("(*)"))
# print(checkValidString("(((***"))
# print(checkValidString2("(((((*(()((((*((**(((()()*)()()()*((((**)())*)*)))))))(())(()))())((*()()(((()((()*(())*(()**)()(())"))


def numIslands(grid: List[List[str]]) -> int:
    islands = 0
    visited = set()

    def dfs(r, c):
        if -1 < r < len(grid) and -1 < c < len(grid[r]):
            if (r, c) not in visited:
                visited.add((r, c))
                if grid[r][c] == "1":
                    dfs(r + 1, c)
                    dfs(r - 1, c)
                    dfs(r, c + 1)
                    dfs(r, c - 1)

    for r in range(len(grid)):
        for c in range(len(grid[r])):
            if (r, c) not in visited:
                if grid[r][c] == "1":
                    islands += 1
                    dfs(r, c)
                else:
                    visited.add((r, c))

    return islands

# print(numIslands([["1","1","1","1","0"],["1","1","0","1","0"],["1","1","0","0","0"],["0","0","0","0","0"]]))
# print(numIslands([["1","1","1"],["0","1","0"],["1","1","1"]]))


def minPathSum(grid: List[List[int]]) -> int:  #dijkstra, big O NlogN (from ElogV in a graph)
    import heapq
    pq = [(grid[0][0], 0, 0)]
    heapq.heapify(pq)
    m = len(grid) - 1
    n = len(grid[0]) - 1
    visited = set()
    while pq:
        w, r, c = heapq.heappop(pq)
        if r == m and c == n:
            return w
        visited.add((r,c))
        if r+1 <= m and (r+1, c) not in visited:
            heapq.heappush(pq, (w + grid[r+1][c], r + 1, c))
        if c+1 <= n and (r, c+1) not in visited:
            heapq.heappush(pq, (w + grid[r][c+1], r, c+1))
        if c-1>= 0 and (r, c-1) not in visited:  # if you want it to move left
            heapq.heappush(pq, (w + grid[r][c-1], r, c-1))


#print(minPathSum([[1,3,1],[1,1,1],[4,2,1]]))


def minPathSum2(grid: List[List[int]]) -> int:  #dp
    p = [grid[0][0]]
    for i, x in enumerate(grid[0][1:]):
        p.append(x + p[i])
    for r in grid[1:]:
        for c, v in enumerate(r):
            if c > 0:
                p[c] = min(p[c], p[c-1]) + v
            else:
                p[c] += v

    return p[-1]


#print(minPathSum2([[1,3,1],[1,5,1],[4,2,1]]))


def productExceptSelf(nums: List[int]) -> List[int]:
    output = [1 for x in nums]
    left = 1
    right = 1
    for i, x in enumerate(nums):
        output[~i] *= right
        right *= nums[~i]
        output[i] *= left
        left *= x

    return output

#print(productExceptSelf([1,2,3,4]))


class BinaryMatrix(object):
   def get(self, x: int, y: int) -> int:
       m = [[0,0],[1,1]]
       return m[x][y]

   def dimensions(self) -> list:
       return [2, 2]


def leftMostColumnWithOne(binaryMatrix: 'BinaryMatrix') -> int:
    m, n = binaryMatrix.dimensions()
    def binary_search(l, r, i, cur_min):
        if l <= r:
            mid = l + (r - l) // 2
            item = binaryMatrix.get(i, mid)
            if item == 1:
                return binary_search(l, mid - 1, i, mid)
            else:
                return binary_search(mid+1, r, i, cur_min)
        else:
            return cur_min

    result = []
    for i in range(m):
        result.append(binary_search(0, n-1, i, float("Inf")))

    m = min(result)
    if m == float("Inf"):
        return -1
    return m  # MlogN


# b = BinaryMatrix()
# print(leftMostColumnWithOne(b))


def leftMostColumnWithOne2(binaryMatrix: 'BinaryMatrix') -> int:
    m, n = binaryMatrix.dimensions()
    i = 0
    j = n - 1
    while i < m or j > 0:
        cur = binaryMatrix.get(i, j)
        if cur == 0:
            i += 1
            if i == m:
                break
        else:
            if j == 0:
                return 0
            j -= 1
    if j == n - 1:
        return -1
    return j + 1


# b = BinaryMatrix()
# print(leftMostColumnWithOne2(b))


def bstFromPreorder(preorder: List[int]) -> TreeNode:  # TO REDO
    stack = [TreeNode(preorder[0])]
    for value in preorder[1:]:
        if value < stack[-1].val:
            stack[-1].left = TreeNode(value)
            stack.append(stack[-1].left)
        else:
            while stack and stack[-1].val < value:
                last = stack.pop()
            last.right = TreeNode(value)
            stack.append(last.right)
    return stack[0]


# print(bstFromPreorder([8,5,1,7,10,12]))


def search(nums: List[int], target: int) -> int:
    def binary(l, r, n, t):
        if l <= r:
            m = l + (r - l) // 2
            v = n[m]
            if v == t:
                return m
            vl = n[l]
            vr = n[r]
            if v > vl:
                if t == vl:
                    return l
                if vl < t < v:
                    return binary(l+1, m-1, n, t)
                return binary(m+1, r, n, t)
            else:
                if t == vr:
                    return r
                if v < t < vr:
                    return binary(m+1, r-1, n, t)
                return binary(l, m-1, n, t)
        return -1

    if nums and nums[0] > target > nums[-1]:
        return -1
    return binary(0, len(nums) - 1, nums, target)

#print(search([4,5,6,7,0,1,2], 0))
# print(search([4,5,6,7,8,1,2,3], 8))


def subarraySum(nums: List[int], k: int) -> int:
    cur = res = 0
    cache = {0:1}
    for v in nums:
        cur += v
        res += cache.get(cur-k, 0)
        cache[cur] = cache.get(cur, 0) + 1

    return res


# print(subarraySum([1,1,1], 2))
# print(subarraySum([1,2,1], 3))
# print(subarraySum([1,2,1], 4))
# print(subarraySum([1,2,1,1], 4))
# print(subarraySum([1,2,1,-1], 4))
# print(subarraySum([1,-2,1,1], 4))
# print(subarraySum([1,-2,1,4], 4))

class DNode:
    def __init__(self, v, prev = None, next = None):
        self.val = v
        self.prev = prev
        self.next = next


class LRUCache:

    def __init__(self, capacity: int):
        self.c = capacity
        self.data: Dict[int, DNode] = {}
        self.head = DNode((0,0))
        self.tail = DNode((0,0))
        self.head.next = self.tail
        self.tail.prev = self.head

    def get(self, key: int) -> int:
        if key in self.data:
            n = self._remove(key)
            self._add(n)
            return self.data[key].val[1]
        return -1

    def put(self, key: int, value: int) -> None:
        if key in self.data:
            self._remove(key)
        n = DNode((key, value))
        self._add(n)
        if len(self.data) > self.c:
            k = self.head.next.val[0]
            self._remove(k)

    def _add(self, n: DNode) -> None:
        self.data[n.val[0]] = n
        prevNode = self.tail.prev
        prevNode.next = n
        n.next = self.tail
        n.prev = prevNode
        self.tail.prev = n

    def _remove(self, key: int) -> DNode:
        n = self.data.pop(key)
        previousNode = n.prev
        nextNode = n.next
        previousNode.next = nextNode
        nextNode.prev = previousNode

        return n


# obj = LRUCache(2)
# obj.put(2,1)
# obj.put(2,2)
# param_1 = obj.get(2)
# print(param_1)
# obj.put(1,1)