# Singly Linked List
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class LinkedList:

    def makeList(self, v):
        # make a linked list from a vector v
        head = ListNode(v[0])
        for i in range(1, len(v)):
            ptr = head
            while ptr.next:
                ptr = ptr.next
            ptr.next = ListNode(v[i])
        return head

    def reverseList(self, head):
        prev, curr = None, head
        while curr:
            curr.next, prev, curr = prev, curr, curr.next
        return prev

    def deleteDuplicates(self, head: ListNode):
        current = head
        while current and current.next:
            if current.val == current.next.val:
                current.next = current.next.next
            else:
                current = current.next
        return head

    def mergeList(self, l1, l2):

        prehead = ListNode(-1)
        prev = prehead

        while l1 and l2:
            if l1.val <= l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 if l1 is not None else l2
        return prehead.next

    def mergeKLists(self, lists):
        '''
        :type lists: list[ListNode]
        :rtype: ListNode
        '''
        nodes=[]
        for l in lists:
            while l:
                nodes.append(l.val)
                l=l.next
                
        head = point = ListNode(0)
        for x in sorted(nodes):
            point.next = ListNode(x)
            point = point.next 
        return head.next

    def hasCycle(self, head):
        if head == None or head.next == None:
            return False

        slow = head
        fast = head.next

        while slow != fast:
            if fast == None or fast.next == None:
                return False
            slow = slow.next
            fast = fast.next.next
        return True  

    def detectCycle(self, head: ListNode)-> ListNode:
        visited = set()
        node = head
        while node is not None:
            if node in visited:
                return node
            else:
                visited.add(node)
                node = node.next

        return None

    def middleNode(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def removeElements(self, head, val):
        sentinel = ListNode(0)
        sentinel.next = head
        
        prev, curr = sentinel, head
        while curr:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next

        return sentinel.next

    def isPalindrome(self, head):
        vals = []
        current = head
        while current != None:
            vals.append(current.val)
            current = current.next
        return vals == vals[::-1]

    def getDecimalValue(self, head):
        result = 0
        while head:
            result = result*2 + head.val
            head = head.next
        return result

    def addTwoNumber(self, l1, l2):
        result = ListNode(0)
        result_tail = result
        carry = 0
                
        while l1 or l2 or carry:            
            val1  = (l1.val if l1 else 0)
            val2  = (l2.val if l2 else 0)
            carry, out = divmod(val1+val2 + carry, 10)    
                      
            result_tail.next = ListNode(out)
            result_tail = result_tail.next                      
            
            l1 = (l1.next if l1 else None)
            l2 = (l2.next if l2 else None)
               
        return result.next

    def addTwoNumberII(self, l1: ListNode, l2: ListNode)-> ListNode:

        if not l1 and not l2:
            return None
        l1_num = 0
        while l1:
            l1_num = l1_num * 10 + l1.val
            l1 = l1.next 
        
        l2_num = 0
        while l2:
            l2_num = l2_num * 10 + l2.val
            l2 = l2.next 
        
        lsum = l1_num + l2_num
        
        head = ListNode(None)
        curr = head
        for istr in str(lsum):
            curr.next = ListNode(int(istr))
            curr = curr.next 
        
        return head.next 

    def rotateRight(self, head: ListNode, k: int)->ListNode:
        # base case
        if not head:
            return None
        if not head.next:
            return head

        # close the linked list into the ring
        old_tail = head
        n = 1
        while old_tail.next:
            old_tail = old_tail.next
            n += 1
        old_tail.next = head
        
        # find new tail (n-k%n-1)th node
        # and new head (n-k%n)th node
        new_tail = head
        for i in range(n-k % n-1):
            new_tail = new_tail.next
        new_head = new_tail.next
        
        # break the ring
        new_tail.next = None
        return new_head

    def deleteNodes(self, head, m, n):

        dummy = ListNode(None)
        dummy.next = head
        i = 0
        while head:
            if i < m-1:
                i += 1
            else:
                j = 0
                while j < n and head.next:
                    head.next = head.next.next
                    j += 1
                i = 0
            head = head.next
        return dummy.next

    def reverseBetween(self, head: ListNode, m: int, n: int) -> ListNode: 
        if not head:
            return None
        prev, curr = None, head

        while m > 1:
            prev, curr = curr, curr.next 
            m, n = m-1, n-1

        tail, con = curr, prev
        
        while n:
            third = curr.next 
            curr.next = prev
            prev = curr
            curr = third
            n -= 1
        
        if con:
            con.next = prev
        else:
            head = prev
        tail.next = curr
        return head

    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        #  remove the n-th node from the end of list 

        dummy = ListNode(0)
        dummy.next = head
        length=0
        first = head
        while first != None:
            length += 1
            first = first.next 
        
        length -= n
        first = dummy
        while length > 0:
            length -= 1
            first = first.next
        first.next = first.next.next
        return dummy.next 

    def oddEvenList(self, head: ListNode) -> ListNode:
        if head == None:
            return None

        odd, even = head, head.next
        evenHead = even 
        while even and even.next:
            odd.next = even.next
            odd = odd.next 
            even.next = odd.next 
            even = even.next
        odd.next = evenHead
        return head 

    def partition(self, head: ListNode, x: int) -> ListNode:
        before = before_head = ListNode(0)
        after = after_head = ListNode(0)

        while head:
            if head.val < x:
                before.next = head
                before = before.next
            else:
                after.next = head
                after = after.next
            head = head.next

        after.next = None
        before.next = after_head.next
        return before_head.next 

    def findMiddle(self, head):
        # When traversing the list with a pointer slow, 
        # make another pointer fast that traverses twice as fast. 
        # When fast reaches the end of the list, slow must be in the middle.
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    def findMiddleDisconnect(self, head):
        prev = None
        slow = fast = head
        
        while fast and fast.next:
            prev = slow 
            slow = slow.next
            fast = fast.next.next

        if prev:
            prev.next = None  # disconnect left half

        return slow 

    def sortedListToBST(self, head):
        if not head:
            return None

        mid = self.findMiddleDisconnect(head)
        node = TreeNode(mid.val)

        # base case when there is just one element in linked list
        if head == mid:
            return node

        # recursively from balanced BSTs using the left and right
        # halves of the original list
        node.left = self.sortedListToBST(head)
        node.right = self.sortedListToBST(mid.next)
        return node

    def insertionSortList(self, head: ListNode) -> ListNode:
        
        dummyHead = ListNode(0)
        dummyHead.next = nodeToInsert = head
        
        while head and head.next:
            if head.val > head.next.val:
                nodeToInsert = head.next  # locate nodeToInsert 
                nodeToInsertPre = dummyHead  # locate nodeToInsertPre
                
                while nodeToInsertPre.next.val < nodeToInsert.val:
                    nodeToInsertPre = nodeToInsertPre.next 
        # Insert nodeToInsert between nodeToInsertPre and nodeToInsertPre.next     
                head.next = nodeToInsert.next 
                nodeToInsert.next = nodeToInsertPre.next 
                nodeToInsertPre.next = nodeToInsert
            else:
                head = head.next 
                
        return dummyHead.next 

    def PlusOne(self, head: ListNode)-> ListNode:
        sentinel = ListNode(0)
        sentinel.next = head
        not_nine = sentinel

        # find the rightmost not-nine digit
        while head:
            if head.val != 9:
                not_nine = head
            head = head.next 

        # increase this rightmost not-nine digit by 1
        not_nine.val += 1
        not_nine = not_nine.next 

        # set all the following nines to zeros
        while not_nine:
            not_nine.val = 0
            not_nine = not_nine.next 
        # Return sentinel node if it was set to 1, and head sentinel.next otherwise
        return sentinel if sentinel.val else sentinel.next 

    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> ListNode:

        if headA == None or headB == None:
            return None

        A_pointer = headA
        B_pointer = headB

        while A_pointer != B_pointer:
            A_pointer = headB if A_pointer == None else A_pointer.next
            B_pointer = headA if B_pointer == None else B_pointer.next

        return A_pointer



class MyLinkedList:
    # design linked list. 
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.size = 0
        self.head = ListNode(0) # sentinel node
        

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        if index < 0 or index >= self.size:
            return -1
        
        # index steps move from sentinel node to wanted index
        curr = self.head
        for i in range(index + 1):
            curr = curr.next 
        return curr.val 
        

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        self.addAtIndex(0, val)

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        self.addAtIndex(self.size, val)
        

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        if index > self.size:
            return 
        
        if index < 0:
            index = 0
            
        self.size += 1
        # find the predecessor of the node to be added
        pred = self.head 
        for i in range(index):
            pred = pred.next 
        
        # node to be added
        to_add = ListNode(val)
        # insertion itself
        to_add.next = pred.next 
        pred.next = to_add 
        
    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        if index < 0 or index >= self.size:
            return 
        
        self.size -= 1
        
        # find predecessor of the node to be deleted
        pred = self.head
        for i in range(index):
            pred = pred.next 
        # pred.next 
        pred.next = pred.next.next 




class DListNode:
    def __init__(self, x):
        self.val = x
        self.next, self.prev = None, None

class MyDLinkedList:
    def __init__(self):
        self.size = 0
        # sentinel nodes as pseudo-head and pseudo-tail
        self.head, self.tail = DListNode(0), DListNode(0) 
        self.head.next = self.tail
        self.tail.prev = self.head
        

    def get(self, index: int) -> int:
        """
        Get the value of the index-th node in the linked list. If the index is invalid, return -1.
        """
        # if index is invalid
        if index < 0 or index >= self.size:
            return -1
        
        # choose the fastest way: to move from the head
        # or to move from the tail
        if index + 1 < self.size - index:
            curr = self.head
            for _ in range(index + 1):
                curr = curr.next
        else:
            curr = self.tail
            for _ in range(self.size - index):
                curr = curr.prev
                
        return curr.val
            

    def addAtHead(self, val: int) -> None:
        """
        Add a node of value val before the first element of the linked list. After the insertion, the new node will be the first node of the linked list.
        """
        pred, succ = self.head, self.head.next
        
        self.size += 1
        to_add = ListNode(val)
        to_add.prev = pred
        to_add.next = succ
        pred.next = to_add
        succ.prev = to_add
        

    def addAtTail(self, val: int) -> None:
        """
        Append a node of value val to the last element of the linked list.
        """
        succ, pred = self.tail, self.tail.prev
        
        self.size += 1
        to_add = ListNode(val)
        to_add.prev = pred
        to_add.next = succ
        pred.next = to_add
        succ.prev = to_add
        

    def addAtIndex(self, index: int, val: int) -> None:
        """
        Add a node of value val before the index-th node in the linked list. If index equals to the length of linked list, the node will be appended to the end of linked list. If index is greater than the length, the node will not be inserted.
        """
        # If index is greater than the length, 
        # the node will not be inserted.
        if index > self.size:
            return
        
        # [so weird] If index is negative, 
        # the node will be inserted at the head of the list.
        if index < 0:
            index = 0
        
        # find predecessor and successor of the node to be added
        if index < self.size - index:
            pred = self.head
            for _ in range(index):
                pred = pred.next
            succ = pred.next
        else:
            succ = self.tail
            for _ in range(self.size - index):
                succ = succ.prev
            pred = succ.prev
        
        # insertion itself
        self.size += 1
        to_add = ListNode(val)
        to_add.prev = pred
        to_add.next = succ
        pred.next = to_add
        succ.prev = to_add
        

    def deleteAtIndex(self, index: int) -> None:
        """
        Delete the index-th node in the linked list, if the index is valid.
        """
        # if the index is invalid, do nothing
        if index < 0 or index >= self.size:
            return
        
        # find predecessor and successor of the node to be deleted
        if index < self.size - index:
            pred = self.head
            for _ in range(index):
                pred = pred.next
            succ = pred.next.next
        else:
            succ = self.tail
            for _ in range(self.size - index - 1):
                succ = succ.prev
            pred = succ.prev.prev
            
        # delete pred.next 
        self.size -= 1
        pred.next = succ
        succ.prev = pred