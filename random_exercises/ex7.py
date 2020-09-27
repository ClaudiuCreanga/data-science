from typing import List, Dict
from collections import defaultdict
from heapq import heappush, heappop

def canJump(nums: List[int]) -> bool:
    i = j = 0
    n = len(nums)
    while i < n:
        j = max(nums[i], j)
        if j + i >= n - 1:
            return True
        if j == 0 and nums[i] == 0:
            return False
        i += 1
        j -= 1

    return False


# print(canJump([2,3,1,1,4]))
# print(canJump([3,2,1,0,4]))
# print(canJump([1,1,2,2,0,1,1])) # t


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


def bstFromPreorder(preorder: List[int]) -> TreeNode:
    root = TreeNode(preorder[0])
    stack = [root]
    for value in preorder[1:]:
        if value < stack[-1].val:
            stack[-1].left = TreeNode(value)
            stack.append(stack[-1].left)
        else:
            while stack and stack[-1].val < value:
                last = stack.pop()
            last.right = TreeNode(value)
            stack.append(last.right)
    return root

#print(bstFromPreorder([8, 5, 1, 7, 10, 12, 3]))  # [8,5,10,1,7,null,12]


def subarraySum(nums: List[int], k: int) -> int:
    # the trick here is to use a hashmap and store the prefix sum at different points of the list
    # we know that the sum(i, j) == sum(0, j) - sum(0, i)
    # so if we know the prefix sum(0, j) up until one point and we look in the table for a previous encountered sum
    # then we can compare it to our wanted k sum
    answer = 0
    values = {0:1}
    pref_sum = 0
    for i in range(len(nums)):
        pref_sum += nums[i]
        answer += values.get(pref_sum - k, 0)
        values[pref_sum] = values.get(pref_sum, 0) + 1

    return answer

# print(subarraySum([1,1,1], 2)) # 2
# print(subarraySum([1], 0)) # 0

def shortestSubarray(A: List[int], K: int) -> int:
    # Return the length of the shortest, non-empty, contiguous subarray of A with sum at least K
    answer = float("inf")
    start = max(A[0], 0)
    if start >= K:
        return 1
    pref = [start]
    for i in range(1, len(A)):
        if A[i] >= K:
            return 1
        pref.append(max(A[i] + pref[i-1], 0))
        if pref[i] >= K:
            answer = min(answer, i+1)
            j = i - 1
            while j >= 0 and i-j < answer and pref[i] - pref[j] < K:
                j -= 1
            answer = min(answer, i - j)

    return -1 if answer == float("inf") else answer
    # if we already have sum >= K, adding elements will not help us, only removing them.
    # maybe find all subarrays with sum at least k and then see the shortest
#
# print(shortestSubarray([-28,81,-20,28,-29], 89)) #3
# print(shortestSubarray([84,-37,32,40,95], 167)) #3
# print(shortestSubarray([1], 1)) #1
# print(shortestSubarray([1, 2], 4)) #-1
# print(shortestSubarray([15, 20, 7, 8, 50], 40)) #1
# print(shortestSubarray([15, 20, 7, 8, 50], 4)) #1
# print(shortestSubarray([1, 2, -1, 2, 3], 4)) #2
# print(shortestSubarray([2,-1,2], 3)) #3



def minSubArrayLen(s: int, nums: List[int]) -> int:
    # Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead
    ans = float("inf")
    if not nums:
        return 0
    cur = j = 0
    for i in range(len(nums)):
        if nums[i] >= s:
            return 1
        cur += nums[i]
        if cur >= s:
            ans = min(ans, i+1)
            while j < i:
                cur -= nums[j]
                if cur >= s:
                    ans = min(ans, i-j)
                    j += 1
                else:
                    j += 1
                    break

    return 0 if ans == float("inf") else ans

def minSubArrayLen2(s: int, A: List[int]) -> int:
    # another way to do sliding window
    i = 0
    res = len(A) + 1
    cur = 0
    for j in range(len(A)):
        cur += A[j]
        while cur >= s:
            res = min(res,  j - i + 1)
            cur -= A[i]
            i += 1

    return res % (len(A) + 1)




print(minSubArrayLen2(7, [2,3,1,2,4,3])) # 2
print(minSubArrayLen2(4, [1,4,4])) # 1