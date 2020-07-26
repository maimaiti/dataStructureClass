from typing import List

def maxSubArray(nums: List[int]) -> int:
    curr_sum = max_sum = nums[0]
    
    for i in range(1,len(nums)):
        curr_sum = max(nums[i], curr_sum + nums[i])
        max_sum = max(max_sum, curr_sum)
    return max_sum

# nums = [-2,1,-3,4,-1,2,1,-5,4]    # output 6
# print(maxSubArray(nums))

def rob(nums: List[int]) -> int:
    preMax = 0
    currMax = 0 # choose f(-1) = f(0) = 0 for simplicity
    for x in nums:
        preMax, currMax = currMax, max(preMax + x, currMax)
    return currMax

# nums = [2,7,9,3,1]
# print(rob(nums))

def coinChange(coins: List[int], amount: int) -> int:
    
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for coin in coins:
        for x in range(coin, amount+1):
            dp[x] = min(dp[x], dp[x - coin] +1)
            
    return dp[amount] if dp[amount] != float('inf') else -1

# print(coinChange([1,2,5], 11))