# Singly Linked List
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class LinkedList:

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

    def reverseKgroup(self, head, k):

        def reverseLinkedList(head, k):

            new_head, ptr = None, head

            while k:
                next_node=ptr.next
                ptr.next = new_head
                new_head = ptr
                ptr = next_node
                k -= 1
            return new_head

        ptr = head
        ktail = None
        new_head = None
        while ptr:
            count = 0
            ptr = head
            while count < k and ptr:
                ptr = ptr.next
                count += 1

            if count == k:
                revHead = reverseLinkedList(head, k)
                if not new_head:
                    new_head=revHead
                
                if ktail:
                    ktail.next = revHead
                
                ktail=head
                head=ptr
        
        if ktail:
            ktail.next = head

        return new_head if new_head else head


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


a=ListNode(1)
b=ListNode(2)
c=ListNode(3)
d=ListNode(4)

a.next = b
b.next = c
c.next = d


ll = LinkedList()
dd=ll.deleteNodes(a, 1, 1)
print(dd.val, dd.next.val) 


