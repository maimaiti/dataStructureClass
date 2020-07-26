from typing import List 

def numIslands(grid: List[List[str]]) -> int:
    if not grid:
        return 0
    x_max, y_max = len(grid), len(grid[0]) # rows, cols
    res = 0

    def helper(x, y):
        if 0 <= x < x_max and 0 <= y < y_max and grid[x][y] == '1':
            grid[x][y] = '0'
            helper(x, y-1)  # left
            helper(x, y+1)  # right
            helper(x+1, y)  # down
            helper(x-1, y)  # up
    
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            if val == '1':
                res += 1
                helper(i, j)
    return res

grid = [
  ["1","1","0","0","0"],
  ["1","1","0","0","0"],
  ["0","0","1","0","0"],
  ["0","0","0","1","1"]]
print(numIslands(grid))

