

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




