from typing import List

def maxsum(arr, k):  # O n*k
    max = 0
    for i in range(len(arr)):
        temp = arr[i]
        for j in range(k-1):
            if i+j+1 < len(arr) - 1:
                temp += arr[i+j+1]
        max = temp if temp > max else max
    return max

#print(maxsum([1, 4, 2, 10, 23, 3, 1, 0, 20], 4))

# the window sliding technique moves the panel to the right by substracting the first element of the panel and adding just the next element
# we move basically the calculation, not the physical numbers, see below

def maxsum_better(arr, k): # O n
    window_sum = sum([arr[i] for i in range(k)])
    max = window_sum
    for i in range(len(arr) - k):
        window_sum = window_sum - arr[i] + arr[i+k]
        max = window_sum if window_sum > max else max
    return max

#print(maxsum_better([100, 4, 2, 10, 23, 3, 1, 0, 20], 4))


def print_max(a, n, k):
    # max_upto array stores the index
    # upto which the maximum element is a[i]
    # i.e. max(a[i], a[i + 1], ... a[max_upto[i]]) = a[i]

    max_upto = [0 for i in range(n)]

    # Update max_upto array similar to
    # finding next greater element
    s = []
    s.append(0)

    for i in range(1, n):
        while (len(s) > 0 and a[s[-1]] < a[i]):
            max_upto[s[-1]] = i - 1
            del s[-1]

        s.append(i)

    while (len(s) > 0):
        max_upto[s[-1]] = n - 1
        del s[-1]

    j = 0
    result = []
    for i in range(n - k + 1):

        # j < i is to check whether the
        # jth element is outside the window
        while (j < i or max_upto[j] < i + k - 1):
            j += 1
        result.append(a[j])

    return result


# Driver code

a = [9, 7, 2, 4, 6, 8, 2, 1, 5]
n = len(a)
k = 3
#print(print_max(a, n, k))


def longestOnes(A, K):
        i=0
        j=0
        maxlen = 0
        while j < len(A):
            if A[j]==1:
                j += 1
                maxlen = max(maxlen, j-i)
            else:
                if K>0:
                    K -= 1
                    j += 1
                    maxlen = max(maxlen, j-i)
                else:
                    if A[i]==0:
                        K += 1
                    i += 1
        return maxlen

#print(longestOnes([1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0], 0))

def longestOnes2(A: List[int], K: int) -> int:
    # the longest subarra with num of 0s <= K
    # dic[i]: how many num of 0s in A[:i]
    # Time, space O(n)
    dic, cnt, res = {}, 0, float('-inf')
    for i, num in enumerate(A):
        if 0 == num:
            cnt += 1
            dic[cnt] = i
        if cnt <= K:
            res = i + 1
        elif cnt - K in dic:
            res = max(res, i - dic[cnt - K])
    return res

#print(longestOnes2([1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0], 2))


def longestOnes3(A: List[int], K: int) -> int:
    '''
    sliding window
    For each A[hi], try to find the longest subarray.
    If A[lo] ~ A[hi] has zeros <= K, we continue to increment j until we include up to K + 1 0s
    If A[lo] ~ A[hi] has zeros > K, we increment i in order to potentially(depend on A[lo] ==1 or 0) exclude redundant 0s.
    '''
    lo = 0
    for hi in range(len(A)):
        K -= 1 - A[hi]
        if K < 0:
            # num of 0s in A[lo:hi] > K
            K += 1 - A[lo]
            lo += 1
            # it doesn't matter if A[lo] == 1 and K < 0 after that, because we need max width of the window, cur length of window is the max by far, so it won't affect the answer
    return hi - lo + 1

print(longestOnes3([1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0], 2))

def longestOnes4(A: List[int], K) -> int:
    begin, end = 0, 0
    for i in range(len(A)):
        if A[i] == 1:
            end += 1
        else:
            if K > 0:
                end += 1
                K -= 1
            else:
                begin += 1
                K += 1
    return end - begin


#print(longestOnes4([1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0], 2))
