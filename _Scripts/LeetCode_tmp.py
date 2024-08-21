import time


def func(nums):
    k = 1
    if len(nums) == 1:
        return 1
    for i in range(1, len(nums)):
        if nums[i] != nums[i-1]:
            nums[k] = nums[i]
            k += 1
        print(i, nums)
    return k


def func2(haystack, needle):
    answer = -1
    foundword = False
    if len(needle) > len(haystack):
        return answer
    if haystack == needle:
        return 0
    for i in range(len(haystack)-len(needle)+1):
        ii = i
        word = ''
        if foundword:
            #print('foundword = True 1')
            break
        #print(f'i = {i}, {haystack[i]}, {needle[0]}, {haystack[i+len(needle)-1]}, {needle[-1]}')
        if haystack[i] == needle[0] and haystack[i+len(needle)-1] == needle[-1]:
            for j in range(len(needle)):
                if foundword:
                    #print('foundword = True 2')
                    break
                if haystack[ii] == needle[j]:
                    word += haystack[ii]
                    if word == needle:
                        answer = i
                        foundword = True
                        break
                    else:
                        ii += 1
    return answer

'''
def func3(haystack, needle):
    answer = -1
    foundword = False
    for j in range(len(needle)):
        for i in range(len(haystack)):
            if haystack[i] == needle[j]:

        ii = i
            word = ''
            if foundword: break
                word += haystack[ii]
                if word == needle:
                    answer = i
                    foundword = True
                    break
                else:
                    ii += 1
    return answer
'''


def maxProfit(prices):
    """
    :type prices: List[int]
    :rtype: int
    """
    i_min = prices[0]
    i_max = 0
    profit = 0
    for i in prices:
        if i<=i_min:
            i_min = i
        else:
            i_max = i
            if (i_max - i_min) > profit:
                profit = i_max - i_min
        print(f"imin {i_min}, imax {i_max}, profit {profit}")
    return profit


def romanToInt(s):
    """
    :type s: str
    :rtype: int
    """
    num = 0
    for n in s:
        if n == 'M':
            num = 1000
    return num


def summaryRanges(nums):
    if len(nums) < 1:
        res = nums
    elif len(nums) == 1:
        res = [str(nums[0])]
    else:
        res = list()
        rng = [nums[0]]
        nums.append(10000000)
        for i in nums[1:]:
            rng.append(i)
            if abs(rng[-2] - i) > 1:
                if len(rng) > 2:
                    res.append(str(rng[0]) + "->" + str(rng[-2]))
                    del rng[:-1]
                else:
                    res.append(str(rng[0]))
                    del rng[:-1]
    return res




#haystack = "mississippi"
haystack = 'abcsds'
needle = "c"
#needle = "issipi"
nums = [0,0,1,1,1,2,2,3,3,4]
nums = [1]
s = "XCIV"
prices = [7,1,5,3,6,4]
prices = [7,6,4,3,1]
nums = [0,1,2,4,5,7]
#nums = [0,2,3,4,6,8,9]
# ["0","2->4","6","8->9"]
start = time.time()

#print(func2(haystack, needle))
#print(func(nums))
print(summaryRanges(nums))
#print(maxProfit(prices))
#print(romanToInt(s))

finish = time.time()
delta_time = finish - start
print(f'\nScript took {delta_time:.25f} sec')