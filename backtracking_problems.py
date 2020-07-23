from typing import List

def permute(nums: List[int]) -> List[List[int]]:

    def backtrack(first = 0):
        if first == n:    # if all integers are used up
            output.append(nums[:])
        
        for i in range(first, n):
            nums[first], nums[i] = nums[i], nums[first]
            backtrack(first + 1)
            nums[first], nums[i] = nums[i], nums[first]
    n = len(nums)
    output = []
    backtrack()
    return output

# nums=[1,2,3]
# print(permute(nums))

def permuteUnique(nums: List[int]) -> List[List[int]]:
    def backtrack(nums, path, l):
        # if all integers are used up
        if len(path) == l:
            res.append(path[:])

        for i in range(len(nums)):
            if i == 0 or nums[i] != nums[i-1]:
                # append i-th integer first 
                path.append(nums[i])
                # use the left integers to complete the permutations
                backtrack(nums[:i] + nums[i+1:], path, l)
                # backtrack
                path.pop()
    res = []
    nums.sort()
    backtrack(nums, [], len(nums))
    return res

# nums = [1,1,2]
# print(permuteUnique(nums)) 

def subsets(nums: List[int]) -> List[List[int]]:

    n = len(nums)
    out = [[]]

    for num in nums:
        out += [curr + [num] for curr in out]
    return out 

# nums = [1, 2, 3]
# print(subsets(nums))

def subsetsWithDup(nums: List[int]) -> List[List[int]]:
    
    output = [[]]
    nums.sort()
    
    for num in nums:
        output += [curr + [num] for curr in output if curr + [num] not in output]
    return output

# nums = [1, 2, 2]
# print(subsetsWithDup(nums))

#Given two integers n and k, 
# return all possible combinations of k numbers out of 1 ... n.
def combine(n: int, k: int) -> List[List[int]]:
    def backtrack(first = 1, curr = []):
        # if the combination is done
        if len(curr) == k:  
            output.append(curr[:])
        for i in range(first, n + 1):
            # add i into the current combination
            curr.append(i)
            # use next integers to complete the combination
            backtrack(i + 1, curr)
            # backtrack
            curr.pop()
    
    output = []
    backtrack()
    return output

#print(combine(4,2))


def combinationSum(candidates: List[int], target: int) -> List[List[int]]:

    result = []
    candidates = sorted(candidates)
    
    def backtrack(remain, stack):
        if remain == 0:
            result.append(stack)
            return 

        for item in candidates:
            if item > remain: 
                break
            if stack and item < stack[-1]: 
                continue
            else:
                backtrack(remain - item, stack + [item])

    backtrack(target, [])
    return result

# candidates = [2,3,6,7]  # no duplications
# print(combinationSum(candidates, 7))

def combinationSum2(candidates: List[int], target: int) -> List[List[int]]:
    res = []
    candidates.sort()
    
    def backtrack(idx, path, cur):
        if cur > target: return
        if cur == target:
            res.append(path)
            return
        for i in range(idx, len(candidates)):
            if i > idx and candidates[i] == candidates[i-1]:
                continue
            backtrack(i+1, path+[candidates[i]], cur+candidates[i])
    backtrack(0, [], 0)
    return res

# candidates = [10,1,2,7,6,1,5]  # duplication
# target = 8
# print(combinationSum2(candidates, target))

# Given a string containing digits from 2-9 inclusive, 
# return all possible letter combinations that the number could represent.
def letterCombinations(digits: str) -> List[str]:
    
        phone = {'2': ['a', 'b', 'c'],
                    '3': ['d', 'e', 'f'],
                    '4': ['g', 'h', 'i'],
                    '5': ['j', 'k', 'l'],
                    '6': ['m', 'n', 'o'],
                    '7': ['p', 'q', 'r', 's'],
                    '8': ['t', 'u', 'v'],
                    '9': ['w', 'x', 'y', 'z']}
        
        def backtrack(combination, next_digits):
            
            if len(next_digits) == 0:
                output.append(combination)
                
            else:
                for letter in phone[next_digits[0]]:
                    backtrack(combination + letter, next_digits[1:])
            
        output = []
        
        if digits:
            backtrack("", digits)
        return output

#Given n pairs of parentheses, write a function to generate 
# all combinations of well-formed parentheses.
def generateParenthesis(N):
    ans = []
    def backtrack(S = '', left = 0, right = 0):
        if len(S) == 2 * N:
            ans.append(S)
            return
        if left < N:
            backtrack(S+'(', left+1, right)
        if right < left:
            backtrack(S+')', left, right+1)

    backtrack()
    return ans

# print(generateParenthesis(3))

def restoreIpAddresses(s: str) -> List[str]:

    def backtrack(s, current, start):
        if len(current) == 4:
            if start == len(s):
                res.append(".".join(current))
            return
        for i in range(start, min(start+3, len(s))):
            if s[start] == '0' and i > start:
                continue
            if 0 <= int(s[start:i+1]) <= 255:
                backtrack(s, current + [s[start:i+1]], i + 1)

    res = []
    backtrack(s, [], 0)
    return res

#print(restoreIpAddresses("25525511135"))

class CourseSchedule(object):

    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """
        from collections import defaultdict
        courseDict = defaultdict(list)

        for relation in prerequisites:
            nextCourse, prevCourse = relation[0], relation[1]
            courseDict[prevCourse].append(nextCourse)

        path = [False] * numCourses
        for currCourse in range(numCourses):
            if self.isCyclic(currCourse, courseDict, path):
                return False
        return True


    def isCyclic(self, currCourse, courseDict, path):
        """
        backtracking method to check that no cycle would be formed starting from currCourse
        """
        if path[currCourse]:
            # come across a previously visited node, i.e. detect the cycle
            return True

        # before backtracking, mark the node in the path
        path[currCourse] = True

        # backtracking
        ret = False
        for child in courseDict[currCourse]:
            ret = self.isCyclic(child, courseDict, path)
            if ret: break

        # after backtracking, remove the node from the path
        path[currCourse] = False
        return ret

# numCourses = 2
# prerequisites = [[1,0],[0,1]]
# print(CourseSchedule().canFinish(numCourses, prerequisites))


