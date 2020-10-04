from typing import List, Dict
from collections import defaultdict
from heapq import heappush, heappop

def canJump(nums: List[int]) -> bool:
    pass

print(canJump([2,3,1,1,4])) # t
print(canJump([3,2,1,0,4])) # f
print(canJump([1,1,2,2,0,1,1])) # t


class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

def bstFromPreorder(preorder: List[int]) -> TreeNode:
    # 2 solutions
    pass

print(bstFromPreorder([8,5,1,7,10,12]))  #[8,5,10,1,7,null,12]


def subarraySum(nums: List[int], k: int) -> int:
    pass

print(subarraySum([1,1,1], 2)) # 2
print(subarraySum([1], 0)) # 0


def minSubArrayLen(s: int, nums: List[int]) -> int:
    # Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead
    pass

print(minSubArrayLen(7, [2,3,1,2,4,3])) # 2


def longestOnes(A: List[int], K: int) -> int:
    # Given an array A of 0s and 1s, we may change up to K values from 0 to 1.
    # Return the length of the longest (contiguous) subarray that contains only 1s.
    pass

print(longestOnes([1,1,1,0,0,0,1,1,1,1,0], 2)) # 6
print(longestOnes([0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], 3)) # 10


def totalFruit(tree: List[int]) -> int:
    # find the longest contiguos array that contains only 2 distinct numbers
    pass

print(totalFruit([1,2,1])) # 3
print(totalFruit([0,1,2,2])) # 3
print(totalFruit([3,3,3,1,2,1,1,2,3,3,4])) # 5
print(totalFruit([1,2,3,2,2])) # 4


def balancedString(s: str) -> int:
    #You are given a string containing only 4 kinds of characters 'Q', 'W', 'E' and 'R'.
    # minimum length of string that replaces the substring to be balanced
    pass

print(balancedString("WWEQERQWQWWRWWERQWEQ")) # 4
print(balancedString("QWQQEEER")) #2
print(balancedString("QWER")) #0
print(balancedString("QQER")) #1
print(balancedString("QQQR")) #2
print(balancedString("QQQQ")) #3


def countAndSay(n: int) -> str:
    #https://leetcode.com/problems/count-and-say/
    pass

print(countAndSay(1)) # 1
print(countAndSay(3)) # 21
print(countAndSay(4)) # 1211


def numberOfSubarrays(nums: List[int], k: int) -> int:
    # Given an array of integers nums and an integer k. A continuous subarray is called nice if there are k odd numbers on it.
    # Return the number of nice sub-arrays.
    pass

print(numberOfSubarrays([1,1,2,1,1], 3)) # 2
print(numberOfSubarrays([2,4,6], 1)) # 0
print(numberOfSubarrays([2,2,2,1,2,2,1,2,2,2], 2)) # 16


def subarraysWithKDistinct(A: List[int], K: int) -> int:
    # Given an array A of positive integers, call a (contiguous, not necessarily distinct) subarray of A good if the number of different integers in that subarray is exactly K
    pass


print(subarraysWithKDistinct([1,2,1,2,3], 2)) # 7
print(subarraysWithKDistinct([1,2,1,3,4], 3)) # 3


def shortestSubarray(A: List[int], K: int) -> int:
    # Return the length of the shortest, non-empty, contiguous subarray of A with sum at least K
    pass

print(shortestSubarray([-28,81,-20,28,-29], 89)) #3
print(shortestSubarray([84,-37,32,40,95], 167)) #3
print(shortestSubarray([1], 1)) #1
print(shortestSubarray([1, 2], 4)) #-1
print(shortestSubarray([15, 20, 7, 8, 50], 40)) #1
print(shortestSubarray([15, 20, 7, 8, 50], 4)) #1
print(shortestSubarray([1, 2, -1, 2, 3], 4)) #2
print(shortestSubarray([2,-1,2], 3)) #3


def numSubarraysWithSum(A: List[int], S: int) -> int:
    # In an array A of 0s and 1s, how many non-empty subarrays have sum S?
    pass

print(numSubarraysWithSum([1,0,1,0,1], 2)) # 4
print(numSubarraysWithSum([0,0,0,0,0], 0)) # 15


def findMin(nums: List[int]) -> int:
    # Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand. find the minimum element
    pass

print(findMin([3,4,5,1,2] )) #1
print(findMin([4,5,6,7,0,1,2])) #0


def twoSum(numbers: List[int], target: int) -> List[int]:
    # Given an array of integers that is already sorted in ascending order, find two numbers such that they add up to a specific target number.
    pass

print(twoSum([2,7,11,15], 9)) # 1,2


def mySqrt(x: int) -> int:
    # Implement int sqrt(int x).
    pass

print(mySqrt(8))


def searchRange(nums: List[int], target: int) -> List[int]:
    # Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.
    pass

print(searchRange([5,7,7,8,8,10], 8)) #3, 4


def myPow(x: float, n: int) -> float:
    # Implement pow(x, n), which calculates x raised to the power n (i.e. xn).
    pass

print(myPow(2, 10)) # 1024
print(myPow(2, -2)) # 0.25


def hIndex(citations: List[int]) -> int:
    # find the h index of a researcher
    pass

print([0,1,3,5,6]) # 3


class RecentCounter:
    def __init__(self):
        pass

    def ping(self, t: int) -> int:
        pass

obj = RecentCounter()
print(obj.ping(1))
print(obj.ping(100))
print(obj.ping(3001))
print(obj.ping(3002))