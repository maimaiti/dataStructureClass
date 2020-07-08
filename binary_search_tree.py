import collections

# binary search Tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right=right

class binarySearchTree:

    def inorder(self, root):
        '''Inorder traversal of BST is an array sorted in the ascending order. '''
        return self.inorder(root.left) + [root.val] + self.inorder(root.right) if root else []

    def successor(self, root):
        '''One step right and then always left '''
        root=root.right
        while root.left:
            root=root.left
        return root.val

    def predecessor(self, root):
        '''One step left and then always right '''
        root = root.left
        while root.right:
            root = root.right
        return root.val

    def deleteNode(self, root, key):
        if not root:
            return None

            # delete from the right subtree
        if key > root.val:
            root.right = self.deleteNode(root.right, key)
        # delete from the left subtree
        elif key < root.val:
            root.left = self.deleteNode(root.left, key)
        
        else:
            # the node is a leaf
            if not (root.left or root.right):
                root = None

            # the root is not a leaf and has a right child
            elif root.right:
                root.val = self.successor(root)
                root.right = self.deleteNode(root.right, root.val)
            else:
                root.val = self.predecessor(root)
                root.left = self.deleteNode(root.left, root.val)
        return root

    def insertIntoBST(self, root, val):
        if not root:
            return TreeNode(val)

        if val > root.val:
            root.right=self.insertIntoBST(root.right, val)
        else:
            root.left=self.insertIntoBST(root.left, val)
        return root
    
    def serialize(self, root: TreeNode)->str:
        '''Encode a tree to a single string'''
        def postorder(root):
            return postorder(root.left) + postorder(root.right) + [root.val] if root else []
        return ' '.join(map(str, postorder(root)))
    
    def deserialize(self, data: str)-> TreeNode:
        '''decodes encoded data to tree'''
        def helper(lower=float('-inf'), upper=float('inf')):
            if not data or data[-1] < lower or data[-1] > upper:
                return None
            val = data.pop()
            root=TreeNode(val)
            root.right=helper(val, upper)
            root.right=helper(lower, val)
            return root
        
        data = [int(x) for x in data.split(' ') if x]
        return helper()

    def inorderSuccessor(self, root, p):
        # successor: one step right and then left till you can
        if p.right:
            p=p.right
            while p.left:
                p=p.left
            return p

        #the successor is somewhere upper in the tree
        stack, inorder=[], float('-inf')

        # inorder traversal: left->node->right 
        while stack or root:
            while root:
                stack.append(root)
                root=root.left

            # 2. all logica around the node
            root=stack.pop()
            if inorder == p.val:
                return root
            inorder = root.val

            # 3. go one step right
            root = root.right
        return None

    def findMode(self, root: TreeNode)-> list:
        if root:
            q=[]
            q.append(root)

            l = []
            while len(q) > 0:
                node=q.pop(0)
                l.append(node.val)
                if node.left != None:
                    q.append(node.left)
                if node.right != None:
                    q.append(node.right)
            
            c=collections.Counter(l)
            m=max(c.values())
            return [key for key, value in c.items() if value == m]

    
    def isValidBST(self, root):

        def helper(node, lower=float('-inf'), upper=float('inf')):
            if not node:
                return True

            val = node.val
            if val <= lower or val >= upper:
                return False
            
            if not helper(node.right, val, upper):
                return False
            if not helper(node.left, lower, val):
                return False
            return True
        
        return helper(root)


    def closestValue(self, root: TreeNode, target: float) -> int:

        def inorder(r: TreeNode):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
        
        return min(inorder(root), key = lambda x: abs(target - x))

    
    def sortedArrayToBST(self, nums: list) -> TreeNode:

        def helper(left, right):
            if left > right:
                return None
            
            p = (left + right)//2

            root=TreeNode(nums[p])
            root.left=helper(left, p-1)
            root.right=helper(p+1, right)
            return root
        
        return helper(0, len(nums)-1)


    def lowestCommonAncestor(self, root, p, q):

        parent_val = root.val
        p_val = p.val
        q_val = q.val
        if p_val > parent_val and q_val > parent_val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p_val < parent_val and q_val < parent_val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root

    
    def searchBST(self, root, val):
        if root is None or val == root.val:
            return root
        
        return self.searchBST(root.left, val) if val < root.val \
            else self.searchBST(root.right, val)

    
    def trimBST(self, root, L, R):
        """L and R are the lowest and highest boundaries"""
        def trim(node):
            if not node:
                return None
            elif node.val > R:
                return trim(node.left)
            elif node.val < L:
                return trim(node.right)
            else:
                node.left=trim(node.left)
                node.right=trim(node.right)
                return node
        return trim(root)

    def rangeSumBST(self, root, L, R):
        '''sum of values of all nodes between L and R inclusive'''
        def dfs(node):
            if node:
                if L <= node.val <= R:
                    self.ans += node.val
                if L < node.val:
                    dfs(node.left)
                if node.val < R:
                    dfs(node.right)
        self.ans = 0
        dfs(root)
        return self.ans 

    
    def getMinDiff(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        vals = []
        def dfs(root):
            if root:
                vals.append(root.val)
                dfs(root.left)
                dfs(root.right)

        dfs(root)
        vals.sort()
        return abs(min(vals[i+1] - vals[i] for i in range(len(vals)-1)))

    def __init__(self): # global tate so each recursive call can access and modifiy 
        self.total = 0     # the total sum

    def BST2greaterTree(self, root):
        if root:
            self.BST2greaterTree(root.right)
            self.total += root.val
            root.val = self.total
            self.BST2greaterTree(root.left)
        return root

    def twoSumIV(self, root, k):

        def find(root, k, setVals):
            if root is None:
                return False
            if k-root.val in setVals:
                return True
            setVals.add(root.val)
            return find(root.left, k, setVals) or find(root.right, k, setVals)
        
        setVals = set()
        return find(root, k, setVals)

    def twoSumBSTs(self, root1: TreeNode, root2: TreeNode, target: int)-> bool:

        def dfs(root, res):
            if not root:
                return None
            dfs(root.left, res)
            res.append(root.val)
            dfs(root.right, res)
        
        tree1, tree2 = [], []
        dfs(root1, tree1)
        dfs(root2, tree2)

        d = {}
        for i in range(len(tree1)):
            val = target - tree1[i]
            d[val] = True
        
        for i in range(len(tree2)):
            if tree2[i] in d:
                return True
        return False

    def splitBST(self, root: TreeNode, V: int) -> List[TreeNode]:
        if not root:
            return None, None
        elif root.val <= V:
            bns = self.splitBST(root.right, V)
            root.right = bns[0]
            return root, bns[1]
        else:
            bns = self.splitBST(root.left, V)
            root.left = bns[1]
            return bns[0], root

    def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
        # Return a list containing all the integers from both trees sorted in ascending order.
        def inorder(r):
            return inorder(r.left) + [r.val] + inorder(r.right) if r else []
        return sorted(inorder(root1) + inorder(root2))


    def balanceBST(self, root: TreeNode) -> TreeNode:
        
        def dfs(r, res):
            if r:
                dfs(r.left, res)
                res.append(r.val)
                dfs(r.right, res)
                
        value = []
        dfs(root, value)
        
        def buildTree(low, high):
            if low > high:
                return None
            
            mid = (low + high)//2
            ans = TreeNode(value[mid])
            ans.left = buildTree(low, mid-1)
            ans.right = buildTree(mid+1, high)
            
            return ans
        return buildTree(0, len(value)-1)



          
class BSTIterator:
    def __init__(self, root):
        self.nodes_sorted = []
        self.index = -1
        self.inorder(root)
    
    def inorder(self, root):
        if not root:
            return None
        self.inorder(root.left)
        self.nodes_sorted.append(root.val)
        self.inorder(root.right)

    def next(self)-> int:  # O(1)
        self.index += 1
        return self.nodes_sorted[self.index]

    def hasNext(self)-> bool:  # O(1)
        return self.index + 1 < len(self.nodes_sorted)






