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


