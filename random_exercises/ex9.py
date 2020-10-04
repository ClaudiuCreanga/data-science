from typing import List, Dict
from collections import defaultdict
from heapq import heappush, heappop

def subarraysWithKDistinct(A: List[int], K: int) -> int:
    # Given an array A of positive integers, call a (contiguous, not necessarily distinct) subarray of A good if the number of different integers in that subarray is exactly K
    # exactly k elements = at most k elements - at most k -1 elements

    def at_most_k_elements(A, K):
        j = ans = 0
        count = {}
        for i in range(len(A)):
            if A[i] not in count: K -= 1
            count[A[i]] = count.get(A[i], 0) + 1
            while K < 0:
                if count[A[j]] == 1:
                    count.pop(A[j])
                    K += 1
                else:
                    count[A[j]] = count.get(A[j], 0) - 1
                j += 1
            ans += i - j + 1

        return ans

    return at_most_k_elements(A, K) - at_most_k_elements(A, K-1)


class Window:
    def __init__(self):
        self.count = {}
        self.size = 0

    def add(self, x):
        self.count[x] = self.count.get(x, 0) + 1
        if self.count[x] == 1: self.size += 1

    def remove(self, x):
        self.count[x] = self.count.get(x) - 1
        if self.count[x] == 0: self.size -= 1


def subarraysWithKDistinct2(A: List[int], K: int) -> int:
    window1 = Window()
    window2 = Window()
    left = left2 = ans = 0
    for right, x in enumerate(A):
        window1.add(x)
        window2.add(x)

        while window1.size > K:
            window1.remove(A[left])
            left += 1

        while window2.size >= K:
            window2.remove(A[left2])
            left2 += 1

        ans += left2 - left

    return ans

#
# print(subarraysWithKDistinct2([1,2,1,2,3], 2)) # 7
# print(subarraysWithKDistinct2([1,2,1,3,4], 3)) # 3


def guards_solution(banana_list: List[int]):
    # to revisit from google foobar
    list_len = len(banana_list)
    graph = list([0] * list_len for i in range(list_len))

    def gcd(x, y):
       while(y):
           x, y = y, x % y
       return x

    def infinite_loop(x,y):
        if x == y:
            return 0

        l = gcd(x,y)

        if (x+y) % 2 == 1:
            return 1

        x,y = x/l,y/l
        x,y = max(x,y), min(x,y)
        return infinite_loop(x-y,2*y)

    for i in range(list_len):
        for j in range(list_len):
            if i < j:
                graph[i][j] = infinite_loop(banana_list[i], banana_list[j])
                graph[j][i] = graph[i][j]

    # A DFS based recursive function that returns true if a
    # matching for vertex u is possible
    def bpm(u, matchR, seen):
        for v in range(list_len):
            if graph[u][v] and seen[v] == False:
                seen[v] = True  # Mark v as visited

                if matchR[v] == -1 or bpm(matchR[v], matchR, seen):
                    matchR[v] = u
                    return True
        return False

    # Returns maximum number of matching
    matchR = [-1] * list_len
    result = 0  # Count of guards match
    for i in range(list_len):
        seen = [False] * list_len
        if bpm(i, matchR, seen):
            result += 1
    return int(list_len- 2*(result/2))

# print(guards_solution([1, 1]))
# print(guards_solution([1, 7, 3, 21, 13, 19]))


def numSubarraysWithSum(A: List[int], S: int) -> int:
    # In an array A of 0s and 1s, how many non-empty subarrays have sum S?
    def at_most(A, S):
        j = ans = 0
        for i in range(len(A)):
            S -= A[i]
            while S < 0 and j <= i:
                S += A[j]
                j += 1
            ans += i - j + 1
        return ans

    return at_most(A, S) - at_most(A, S-1)

# print(numSubarraysWithSum([0,0,0,0,0], 0)) # 15
# print(numSubarraysWithSum([1,0,1,0,1], 2)) # 4


def numSubarraysWithSum2(A: List[int], S: int) -> int:
    # In an array A of 0s and 1s, how many non-empty subarrays have sum S?
    ans = pref = 0
    count = {0: 1}
    for i in range(len(A)):
        pref += A[i]
        ans += count.get(pref - S, 0)
        count[pref] = count.get(pref, 0) + 1

    return ans
#
# print(numSubarraysWithSum2([0,0,0,0,0], 0)) # 15
# print(numSubarraysWithSum2([1,0,1,0,1], 2)) # 4

def shortestSubarray(A: List[int], K: int) -> int:
    # Return the length of the shortest, non-empty, contiguous subarray of A with sum at least K
    # TODO revisit
    from collections import deque
    q = deque()
    q.append([0,0])
    res, cur = float("inf"), 0
    for i, a in enumerate(A):
        cur += a
        while q and cur - q[0][1] >= K:
            res = min(res, i + 1 - q.popleft()[0])
        while q and cur <= q[-1][1]:
            q.pop()
        q.append([i + 1, cur])
    return res if res < float('inf') else -1

# print(shortestSubarray([-28,81,-20,28,-29], 89)) #3
# print(shortestSubarray([84,-37,32,40,95], 167)) #3
# print(shortestSubarray([1], 1)) #1
# print(shortestSubarray([1, 2], 4)) #-1
# print(shortestSubarray([15, 20, 7, 8, 50], 40)) #1
# print(shortestSubarray([15, 20, 7, 8, 50], 4)) #1
# print(shortestSubarray([1, 2, -1, 2, 3], 4)) #2
# print(shortestSubarray([2,-1,2], 3)) #3

def findMin(nums: List[int]) -> int:
    # Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. find the minimum element
    # TODO
    pass
    def binary_search(l, h):
        m = l + (h-l) // 2
        if l <= h:
            if m - 1 > 0 and nums[m] < nums[m-1]:
                return nums[m]


    return binary_search(0, len(nums))

# print(findMin([3,4,5,1,2])) #1
# print(findMin([3,4,5,6,7,0,1,2])) #0
# print(findMin([1,2,3,4,5,6,7,0])) #0
# print(findMin([7,8,1,2,3,4])) #1

def twoSum(numbers: List[int], target: int) -> List[int]:
    # Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
    count = {}
    for i, x in enumerate(numbers):
        if x in count and i != count[x]:
            return [min(count[x], i) +1, max(count[x], i) + 1]
        count[target - x] = i

    #
    # def binary_search(l, h):
    #     m = l + (h-l) // 2
    #     if l <= h:
    #         if numbers[m] in count and m != count[m]:
    #             return [min(count[m], m), max(count[m], m)]
    #         else:
    #             return

def twoSum(numbers: List[int], target: int) -> List[int]:
    # Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
    l, h = 0, len(numbers) - 1
    while l < h:
        s = numbers[l] + numbers[h]
        if s == target:
            return [l+1, h+1]
        elif s < target:
            l += 1
        else:
            h -= 1


def twoSum(numbers: List[int], target: int) -> List[int]:
    # Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
    # O(nlogn)
    for i in range(len(numbers)):
        dif = target - numbers[i]
        l, h = i+1, len(numbers) - 1
        while l <= h:
            m = l + (h-l) // 2
            if numbers[m] == dif:
                return [i+1, m+1]
            elif numbers[m] < dif:
                l = m+1
            else:
                h = m-1

# print(twoSum([2,7,11,15], 9)) # 1,2
# print(twoSum([2,3,4], 6)) # 1,3
# print(twoSum([-1,0], -1)) # 1,2

def mySqrt(x: int) -> int:
    # Implement int sqrt(int x).
    # the intuition here is that sqrt(x) is always smaller or equall to x/2
    # if it's not smaller than the high point becomes the middle - 1
    ans, l, h = 0, 1, x
    while l <= h:
        m = l + (h - l) // 2
        if m <= x / m:
            l = m + 1
            ans = m
        else:
            h = m - 1
    return ans

# print(mySqrt(0))
# print(mySqrt(1))
# print(mySqrt(64))

def searchRange(nums: List[int], target: int) -> List[int]:
    # Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.
    def binary_search(l, h, direction = "None"):
        if l <= h:
            m = l + (h - l) // 2
            if nums[m] == target:
                if direction == "left":
                    if m - 1 >= 0 and nums[m-1] == target:
                        return binary_search(l, m-1, direction)
                    return m
                elif direction == "right":
                    if m+1 < len(nums) and nums[m+1] == target:
                        return binary_search(m+1, len(nums) - 1, direction)
                return m
            elif nums[m] < target:
                return binary_search(m + 1, h, direction)
            else:
                return binary_search(l, m - 1, direction)

        return -1

    item = binary_search(0, len(nums) - 1)
    if item == -1:
        return [-1, -1]
    else:
        return [binary_search(0, item, "left"), binary_search(item, len(nums) - 1, "right")]

def searchRange(nums: List[int], target: int) -> List[int]:
    # Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.
    def binary_search_left(left, right):
        if left <= right:
            m = left + (right - left) // 2
            if target > nums[m]: left = m + 1
            else: right = m - 1
        return left

    def binary_search_right(left, right):
        if left <= right:
            m = left + (right - left) // 2
            if target >= nums[m]: left = m + 1
            else: right = m - 1
        return right

    left, right = binary_search_left(0, len(nums) - 1), binary_search_right(0, len(nums) - 1)
    return [left, right] if left <= right else [-1, -1]


# print(searchRange([1,3,3,3,3,3,3,3,3,4], 3)) #1, 8
# print(searchRange([3,3,3], 3)) #0, 2
# print(searchRange([0], 1)) #-1, -1
# print(searchRange([5,7,7,8,8,10], 8)) #3, 4


def myPow(x: float, n: int) -> float:
    # Implement pow(x, n), which calculates x raised to the power n (i.e. xn).
    if n == 0: return 1
    if n < 0: x, n = 1.0 / x, -n
    if n % 2 == 0: return myPow(x, n // 2) * myPow(x, n // 2)
    else: return x * myPow(x, n // 2) * myPow(x, n // 2)

    temp = myPow(x, int(n / 2))

    if (n % 2 == 0):
        return temp * temp
    else:
        return x * temp * temp

        #
    # result = 1.0
    # if n < 0:
    #     x, n = 1.0 / x, -n
    # while n:
    #     if n & 1:
    #         result *= x
    #     x, n = x * x, n >> 1
    # return result


# print(myPow(2, -2)) # 0.25
# print(myPow(2, 10)) # 1024


def hIndex(citations: List[int]) -> int:
    # find the h index of a researcher
    # TODO
    if not citations:
        return 0
    n = len(citations) - 1
    for i in range(n, -1, -1):
        item = citations[i]
        passed = n - i + 1
        if item <= passed:
            return citations[i]

    return len(citations)

print(hIndex([0,0,4,4])) # 2
print(hIndex([11,15])) # 2
print(hIndex([1,1,2])) # 1
print(hIndex([99, 100])) # 1
print(hIndex([100])) # 1
print(hIndex([1])) # 1
print(hIndex([0,1,3,5,6])) # 3