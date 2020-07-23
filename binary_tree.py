import collections
from typing import List

# Binary Tree
class TreeNode:

    def __init__(self, val=0, left=0, right=0):
        self.val = val
        self.left = left
        self.right = right

class ListNode:

    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

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

    def invertTree(self, root: TreeNode) -> TreeNode:
        #  226. Invert Binary Tree
        if root is None:
            return None
        
        else:
            root.left, root.right = self.invertTree(root.right), self.invertTree(root.left)
        return root

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

    def widthOfBinaryTree(self, root: TreeNode) -> int:
        # Maximum Width of Binary Tree 
        if not root:
            return 0

        max_width = 0
        # queue of elements [(node, col_index)]
        queue = deque()
        queue.append((root, 0))

        while queue:
            level_length = len(queue)
            _, level_head_index = queue[0]
            # iterate through the current level
            for _ in range(level_length):
                node, col_index = queue.popleft()
                # preparing for the next level
                if node.left:
                    queue.append((node.left, 2 * col_index))
                if node.right:
                    queue.append((node.right, 2 * col_index + 1))

            # calculate the length of the current level,
            #   by comparing the first and last col_index.
            max_width = max(max_width, col_index - level_head_index + 1)

        return max_width

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

    def levelOrderBottom(self, root: TreeNode) -> List[List[int]]:
        # Iteration: BFS Traversal
        levels = []
        next_level = deque([root])
        
        while root and next_level:
            curr_level = next_level
            next_level = deque()
            levels.append([])
            
            for node in curr_level:
                # append the current node value
                levels[-1].append(node.val)
                # process child nodes for the next level
                if node.left:
                    next_level.append(node.left)
                if node.right:
                    next_level.append(node.right)
                    
        return levels[::-1]

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

    def sumOfLeftLeaves(self, root: TreeNode) -> int:
        # 404. Sum of Left Leaves
        if root is None: 
            return 0

        def is_leaf(node):
            return node is not None and node.left is None and node.right is None

        stack = [root]
        total = 0
        while stack:
            sub_root = stack.pop()
            # Check if the left node is a leaf node.
            if is_leaf(sub_root.left):
                total += sub_root.left.val
            # If the right node exists, put it on the stack.
            if sub_root.right is not None:
                stack.append(sub_root.right)
            # If the left node exists, put it on the stack.
            if sub_root.left is not None:
                stack.append(sub_root.left)

        return total

    def flatten(self, root: TreeNode)-> None:
        '''Do not return anything, modify root in place'''   

        def flattenTree(node: TreeNode):
            if not node:
                return None
            
            if not node.left and not node.right:
                return node

            leftTail = flattenTree(node.left)
            rightTail=flattenTree(node.right)

            if leftTail:
                leftTail.right = node.right
                node.right = node.left
                node.left = None
            return rightTail if rightTail else leftTail

        flattenTree(root)

    def findLeaves(self, root: TreeNode) -> list:
        
        def bottom_up_depth(root):
        
            if not root:
                return 0
            
            # get maximum bottom-up depth then add 1 for current level
            depth = max(bottom_up_depth(root.left), bottom_up_depth(root.right)) + 1
            
            # record in the depth_dict
            depth_dict[depth] = depth_dict.get(depth, []) + [root.val]
            return depth
        
        depth_dict = {}
        bottom_up_depth(root)
        # group nodes by depth
        return [depth_dict[i] for i in depth_dict]

    def rightSideView(self, root):
        if root is None:
            return []

        rightside = []
        
        def helper(node: TreeNode, level: int) -> None:
            if level == len(rightside):
                rightside.append(node.val)
            for child in (node.right, node.left):
                if child:
                    helper(child, level+1)
        helper(root, 0)
        return rightside

    def sumEvenGrandparent(self, root: TreeNode) -> int:
        
        def DFS(node, parent, grandparent):
            if not node:
                return None
            
            nonlocal answer 
            if parent and grandparent and grandparent.val % 2 == 0:
                answer += node.val
                
            DFS(node.left, node, parent)
            DFS(node.right, node, parent)
            
        answer = 0
        DFS(root, None, None)
        return answer

    def flipEquiv(self, root1: TreeNode, root2: TreeNode) -> bool:
        # A binary tree X is flip equivalent to a binary tree Y if and only if 
        # we can make X equal to Y after some number of flip operations.
        if root1 is root2:
            return True
        
        if not root1 or not root2 or root1.val != root2.val:
            return False
        
        return (self.flipEquiv(root1.left, root2.left) and
                self.flipEquiv(root1.right, root2.right) or 
                self.flipEquiv(root1.left, root2.right) and
                self.flipEquiv(root1.right, root2.left))

    def str2tree(self, s):
        # construct a binary tree from a string consisting of parenthesis and integers.
        def t(val, left=None, right=None):
            node, node.left, node.right = TreeNode(val), left, right
            return node

        # '4(2(3)(1))(6(5))' to 't(4,t(2,t(3),t(1)),t(6,t(5)))'
        return eval('t(' + s.replace('(', ',t(')  +')') if s else None 

    def longestUnivaluePath(self, root: TreeNode) -> int:
        # find the length of the longest path where each node in the path has the same value. 
        self.ans = 0
        
        def arrow_length(node):
            if not node: return 0
            left_length = arrow_length(node.left)
            right_length= arrow_length(node.right)
            left_arrow = right_arrow = 0
            if node.left and node.left.val == node.val:
                left_arrow = left_length + 1
            if node.right and node.right.val == node.val:
                right_arrow = right_length + 1
            self.ans = max(self.ans, left_arrow + right_arrow)
            return max(left_arrow, right_arrow)
        
        arrow_length(root)
        return self.ans 

    def distanceK(self, root: TreeNode, target: TreeNode, K: int):
        # Return a list of the values of all nodes that have a distance K from the target node. 

        def dfs(node, par = None):
            if node:
                node.par = par
                dfs(node.left, node)
                dfs(node.right, node)
                
        dfs(root)
        
        # breadth first search to find all nodes a distance K from a target
        queue = collections.deque([(target, 0)])
        seen = {target}
        while queue:
            if queue[0][1] == K:
                return [node.val for node, d in queue]
            node, d = queue.popleft()
            for nei in (node.left, node.right, node.par):
                if nei and nei not in seen:
                    seen.add(nei)
                    queue.append((nei, d+1))
        
        return []

    def isCompleteTree(self, root):
        # 958. Check Completeness of a Binary Tree
        nodes = [(root, 1)]
        i = 0
        while i < len(nodes):
            node, v = nodes[i]
            i += 1
            if node:
                nodes.append((node.left, 2*v))
                nodes.append((node.right, 2*v+1))

        return  nodes[-1][1] == len(nodes)  

    def serialize(self, root):
        """Encodes a tree to a single string.
        
        :type root: TreeNode
        :rtype: str
        """
        def rserialize(root, string):
            """ a recursive helper function for the serialize() function."""
            # check base case
            if root is None:
                string += 'None,'
            else:
                string += str(root.val) + ','
                string = rserialize(root.left, string)
                string = rserialize(root.right, string)
            return string
        
        return rserialize(root, '')
    
    def deserialize(self, data):
        """Decodes your encoded data to tree.
        
        :type data: str
        :rtype: TreeNode
        """
        def rdeserialize(l):
            """ a recursive helper function for deserialization."""
            if l[0] == 'None':
                l.pop(0)
                return None
                
            root = TreeNode(l[0])
            l.pop(0)
            root.left = rdeserialize(l)
            root.right = rdeserialize(l)
            return root

        data_list = data.split(',')
        root = rdeserialize(data_list)
        return root

    def rob(self, root: TreeNode) -> int:
        # 337. House Robber III
        def with_without_rob(root):
            
            if root:
                with_l, without_l = with_without_rob(root.left)
                with_r, without_r = with_without_rob(root.right)
                return (root.val + without_l + without_r, max(with_l, without_l) + max(with_r, without_r))
            return (0, 0)
        return max(with_without_rob(root))     

    def upsideDownBinaryTree(self, root: TreeNode) -> TreeNode:
        # 156. Binary Tree Upside Down
        # edge case
        if not root:
            return None
        # locate new root of the binary tree
        new_root = root
        while new_root.left:
            new_root = new_root.left
            
        def search(node, new_left=None, new_right=None):
            if not node:
                return
            search(node.left, node.right, node)
            search(node.right, None, None)
            node.left = new_left
            node.right = new_right
            
        search(root)
        return new_root

    def findLeaves(self, root: TreeNode):
        
        def bottom_up_depth(root):
        
            if not root:
                return 0
            
            depth = max(bottom_up_depth(root.left), bottom_up_depth(root.right)) + 1
            
            depth_dict[depth] = depth_dict.get(depth, []) + [root.val]
            return depth
        
        depth_dict = {}
        bottom_up_depth(root)
        return [depth_dict[i] for i in depth_dict]

    def findSecondMinimumValue(self, root: TreeNode) -> int:
        # 671. Second Minimum Node In a Binary Tree
        def dfs(node):
            if node:
                uniques.add(node.val)
                dfs(node.right)
                dfs(node.left)
            
        uniques = set()
        dfs(root)
        
        #  The first minimum must be root.val.
        min1, ans = root.val, float('inf')
        for v in uniques:
            if min1 < v < ans:
                ans = v
        
        return ans if ans < float('inf') else -1

    def buildTreePreIn(self, preorder: List[int], inorder: List[int]) -> TreeNode:
        # 105. Construct Binary Tree from Preorder and Inorder Traversal

        def helper(in_left = 0, in_right = len(inorder)):
            nonlocal pre_idx
            # if there is no elements to construct subtrees
            if in_left == in_right:
                return None
            
            # pick up pre_idx element as a root
            root_val = preorder[pre_idx]
            root = TreeNode(root_val)

            # root splits inorder list
            # into left and right subtrees
            index = idx_map[root_val]

            # recursion 
            pre_idx += 1
            # build left subtree
            root.left = helper(in_left, index)
            # build right subtree
            root.right = helper(index + 1, in_right)
            return root
        
        # start from first preorder element
        pre_idx = 0
        # build a hashmap value -> its index
        idx_map = {val:idx for idx, val in enumerate(inorder)} 
        return helper()

    def buildTreeInPost(self, inorder: List[int], postorder: List[int]) -> TreeNode:
        # 106. Construct Binary Tree from Inorder and Postorder Traversal
        def helper(in_left, in_right):
            # if there is no elements to construct subtrees
            if in_left > in_right:
                return None
            
            # pick up the last element as a root
            val = postorder.pop()
            root = TreeNode(val)

            # root splits inorder list
            # into left and right subtrees
            index = idx_map[val]
 
            # build right subtree
            root.right = helper(index + 1, in_right)
            # build left subtree
            root.left = helper(in_left, index - 1)
            return root
        
        # build a hashmap value -> its index
        idx_map = {val:idx for idx, val in enumerate(inorder)} 
        return helper(0, len(inorder) - 1)

    def isSubPath(self, head: ListNode, root: TreeNode) -> bool:
        # 1367. Linked List in Binary Tree
        #  Given a binary tree root and a linked list with head as the first node. 
        # Return True if all the elements in the linked list starting from the head 
        # correspond to some downward path
        def dfs(head, root):
            if not head:
                return True

            if not root:
                return False

            if head.val == root.val:
                return dfs(head.next, root.left) or dfs(head.next, root.right) 

            return False
        
        if not head:
            return True
        if not root:
            return False
        if dfs(head, root):
            return True
        return self.isSubPath(head, root.left) or self.isSubPath(head, root.right)

    def removeLeafNodes(self, root: TreeNode, target: int) -> TreeNode:
        # 1325. Delete Leaves With a Given Value
        if root:
            root.left = self.removeLeafNodes(root.left, target)
            root.right = self.removeLeafNodes(root.right, target) 
            if root.val == target and not root.left and not root.right:
                root = None
        return root 

    def maxProductSplitTree(self, root: TreeNode) -> int:
        # 1339. Maximum Product of Splitted Binary Tree

        all_sums = []

        def tree_sum(subroot):
            if subroot is None: return 0
            left_sum = tree_sum(subroot.left)
            right_sum = tree_sum(subroot.right)
            total_sum = left_sum + right_sum + subroot.val
            all_sums.append(total_sum)
            return total_sum

        total = tree_sum(root)
        best = 0
        for s in all_sums:
            best = max(best, s * (total - s))   
        return best % (10 ** 9 + 7)

    def sumNumbers(self, root: TreeNode) -> int:
        # 129. Sum Root to Leaf Numbers  [1,2,3] 1-2, 1-3, sum = 25 
        def preorder(r, curr_number):
            nonlocal root_to_leaf
            if r:
                curr_number = curr_number * 10 + r.val
                # if it's a leaf, update root-to-leaf sum
                if not (r.left or r.right):
                    root_to_leaf += curr_number
                    
                preorder(r.left, curr_number)
                preorder(r.right, curr_number) 
        
        root_to_leaf = 0
        preorder(root, 0)
        return root_to_leaf

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




