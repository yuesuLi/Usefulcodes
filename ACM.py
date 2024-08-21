
# n = int(input().strip())
# a, b, c, d = map(int, input().strip().split())
# l = list(map(int, input().strip().split()))
# import sys
#
# for line in sys.stdin:
#     a = line.split()
#     print(int(a[0]) + int(a[1]))






import numpy as np

def translate(num):

    num = num.lstrip('0')
    if num == '':
        return ''
    if num in num_dic:
        return num_dic[num]

    n = len(num)
    if n == 2:
        return f"{num_dic[num[0]+'0']} {num_dic[num[1]]}"

    if n == 3:
        prefix = f"{num_dic[num[0]]} hundred"
        suffix = translate(num[1:])
        if suffix:
            return f"{prefix} and {suffix}"
        else:
            return prefix

    if n == 4 or n == 5 or n == 6:

        mididx = n-3
        prefix = f"{translate(num[0:mididx])} thousand"
        suffix = translate(num[mididx:])
        if suffix:
            return f"{prefix} {suffix}"
        else:
            return prefix
    if n == 7:
        prefix = f"{num_dic[num[0]]} million"
        suffix = translate(num[1:])
        if suffix:
            return f"{prefix} {suffix}"
        else:
            return prefix

    return "ERROR"


number = input().strip()
num_dic = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight",
    "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen",
    "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", "20": "twenty", "30": "thirty",
    "40": "forty", "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
}
result = translate(number)
print(result)
print('done')
