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
        count = {}
        size = 0

    def add(self, x):
        count[x] = count.get(x, 0) + 1
        if count[x] == 1: size += 1

    def remove(self, x):
        count[x] = count.get(x) - 1
        if count[x] == 0: size -= 1


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
    pass
    def binary_search(l, h):
        m = l + (h-l) // 2
        if l <= h:
            if m - 1 > 0 and nums[m] < nums[m-1]:
                return nums[m]


    return binary_search(0, len(nums))

print(findMin([3,4,5,1,2])) #1
print(findMin([3,4,5,6,7,0,1,2])) #0
print(findMin([1,2,3,4,5,6,7,0])) #0
print(findMin([7,8,1,2,3,4])) #1

def twoSum(numbers: List[int], target: int) -> List[int]:
    # Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
    count = {}
    for i, x in enumerate(numbers):
        count[target - x] = i

    for i, x in enumerate(numbers):
        if x in count and i != count[x]:
            return [min(count[x], i) +1, max(count[x], i) + 1]

    #
    # def binary_search(l, h):
    #     m = l + (h-l) // 2
    #     if l <= h:
    #         if numbers[m] in count and m != count[m]:
    #             return [min(count[m], m), max(count[m], m)]
    #         else:
    #             return

print(twoSum([2,7,11,15], 9)) # 1,2
print(twoSum([2,3,4], 6)) # 1,3
print(twoSum([-1,0], -1)) # 1,2