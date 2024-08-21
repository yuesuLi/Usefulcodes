
import queue


class Solution:
    def permuteUnique(self, nums):
        nums.sort()
        path = []
        results = []
        length = len(nums)
        mask = [False] * length
        def backtracking(path, mask):
            if len(path) == length:
                results.append(path[:])
                return
            for i in range(length):
                if mask[i] == True: continue
                if i > 0 and nums[i] == nums[i - 1] and mask[i-1] == False: continue
                mask[i] = True
                path.append(nums[i])
                backtracking(path, mask)
                mask[i] = False
                path.pop()

        backtracking(path, mask)
        return results

sol = Solution()
result = sol.permuteUnique([1,1,3])

print('result:', result)


