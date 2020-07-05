import collections

# Binary Tree
class TreeNode:

    def __init__(self, val=0, left=0, right=0):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:

    def invertTree(self, root):
        if not root:
            return None
        else:
            root.left, root.right=self.invertTree(root.right), self.invertTree(root.left)
        return root


    def maxDepth(self, root):
        """return the max depth of the binary tree"""
        if not root:
            return 0
        else:
            left_root = self.maxDepth(root.left)
            right_root = self.maxDepth(root.right)
        return max(left_root, right_root) + 1


    def minDepth(self, root):
        """return the min depth of binary tree"""
        if not root:
            return 0
        children = [root.left, root.right]
        if not any(children):
            return 1

        min_depth=float('inf')
        for c in children:
            if c:
                min_depth=min(self.minDepth(c), min_depth)
        return min_depth + 1

    def inorderTraversal(self, root):

        def helper(root, res):
            if root:
                helper(root.left, res)
                res.append(root.val)
                helper(root.right, res)
        
        res=[]
        helper(root, res)
        return res

    def preOrderTraversal(self, root):

        def helper(root, res):
            if root:
                res.append(root.val)
                helper(root.left, res)
                helper(root.right, res)
        
        res=[]
        helper(root, res)
        return res


    def postOrderTraversal(self, root):
        def helper(root, res):
            if root:
                helper(root.left, res)
                helper(root.right, res)
                res.append(root.val)
        
        res=[]
        helper(root, res)
        return res

    def isSymmetric(self, root):

        def isMirror(t1, t2):
            if not t1 and not t2:
                return True
            if not t1 or not t2:
                return False
            return t1.val == t2.val and isMirror(t1.right, t2.left) \
            and isMirror(t1.left, t2.right)

        return isMirror(root, root)


    def binaryTreePaths(self, root):

        def construct_paths(root, path):
            if root:
                path += str(root.val)
                if not root.left and not root.right:
                    paths.append(path)
                else:
                    path += '->'
                    construct_paths(root.left, path)
                    construct_paths(root.right, path)
        paths = []
        construct_paths(root, "")
        return paths

    def diameterOfBinaryTree(self, root):
        self.ans = 1
        def dfs(node):
            if not node:
                return 0

            L=dfs(node.left)
            R=dfs(node.right)
            self.ans=max(self.ans, L+R+1)
            return max(L, R) + 1

        dfs(root)
        return self.ans - 1

    def isSubtree(self, s, t):

        def dfs(root):
            if root is None:
                return None

            tmp = str(root.val) + ',' + str(dfs(root.left)) + str(dfs(root.right))
            return ',' + tmp
        return dfs(s).find(dfs(s)) >= 0

    
    def maxPathSum(self, root):
        def max_gain(node):
            nonlocal max_sum
            if not root:
                return 0
            left_gain = max(max_gain(node.left), 0)
            right_gain = max(max_gain(node.right), 0)

            price_newpath = node.val + left_gain + right_gain
            max_sum = max(max_sum, price_newpath)

            return node.val + max(left_gain, right_gain)
        max_sum = float('-inf')
        max_gain(root)
        return max_sum

    def levelOrder(self, root: TreeNode):
        levels = []
        if not root:
            return levels
        
        def helper(node, level):
            if len(levels) == level:
                levels.append([])
            
            levels[level].append(node.val)

            if node.left:
                helper(node.left, level + 1)
            if node.right:
                helper(node.right, level + 1)
        helper(root, 0)
        return levels

    
    def zigzagLevelOrder(self, root: TreeNode):
        if not root:
            return None
        q = collections.deque()
        q.append(root)
        res = []
        i = 1
        while q:
            level = []
            for x in range(len(q)):
                node=q.popleft()
                level.append(node.val)
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
            
            if i%2 != 0:
                res.append(level)
            else:
                res.append(level[::-1])
            i += 1
        return res

    def hasPathSum(self, root, sum):
        if not root:
            return False

        sum -= root.val
        if not root.left and not root.right:
            return sum == 0
        return self.hasPathSum(root.left, sum) or self.hasPathSum(root.right, sum)

    def pathSum(self, root: TreeNode, target: int):
        result = []
        current_path = []
        
        def helper(node):
            if not node:
                return None
            
            current_path.append( node.val )
   
            if not node.left and not node.right and sum(current_path) == target:
                result.append(list(current_path))

            helper(node.left)
            helper(node.right)

            current_path.pop()

        helper(root)
        return result

    def verticalOrder(self, root: TreeNode):
        columnTable = collections.defaultdict(list)
        queue = collections.deque([(root, 0)])
        
        while queue:
            node, column = queue.popleft()
            
            if node is not None:
                columnTable[column].append(node.val)
                
                queue.append((node.left, column - 1))
                queue.append((node.right, column + 1))
                
        return [columnTable[x] for x in sorted(columnTable.keys())]

    def sumRootToLeaf(self, root, val=0):
        if not root:
            return 0
        val = val * 2 + root.val
        if root.left == root.right:
            return val
        return self.sumRootToLeaf(root.left, val) + self.sumRootToLeaf(root.right, val)

        



class CBTInserter:
# Complete Binary Tree Inserter. A complete binary tree is a binary tree in which every level, 
# except possibly the last, is completely filled, and all nodes are as far left as possible.

    def __init__(self, root: TreeNode):
        self.deque = collections.deque()
        self.root = root
        q = collections.deque([root])
        while q:
            node = q.popleft()
            if not node.left or not node.right:
                self.deque.append(node)
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
        

    def insert(self, v: int) -> int:
        node = self.deque[0]
        self.deque.append(TreeNode(v))
        
        if not node.left:
            node.left = self.deque[-1]
        else:
            node.right = self.deque[-1]
            self.deque.popleft()
        return node.val





a=TreeNode(1)
b=TreeNode(0)
c=TreeNode(1)
d=TreeNode(0)
e=TreeNode(1)
f=TreeNode(0)

a.left = b 
a.right= c
b.left = d
c.left = e
c.right = f

# a1=TreeNode(5)
# b1=TreeNode(2)
# c1=TreeNode(4)
# a1.left=b1
# a1.right=c1

bt = BinaryTree()
print(bt.sumRootToLeaf(a, val=0))


