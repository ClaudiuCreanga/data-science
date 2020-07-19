
def merge_sort(n):
    # recursive algo
    # base case is when you have one single element, you return it
    if len(n) > 1:
        m = len(n) // 2
        left = n[:m]
        right = n[m:]
        merge_sort(left)
        merge_sort(right)

        i = j = k = 0

        while i < len(left) and j < len(right):
            if left[i] < right[j]:
                n[k] = left[i]
                i += 1
            else:
                n[k] = right[j]
                j += 1
            k += 1

        while i < len(left):
            n[k] = left[i]
            i += 1
            k += 1

        while j < len(right):
            n[k] = right[j]
            j += 1
            k += 1

    return n

print(merge_sort([2,1,5,4,3,12,13,6]))