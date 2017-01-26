def oddManOut(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        i=0
        if (len(nums) == 1):
            return nums
        while i< len(nums):
            if (nums[i] == nums[i+1]):
                i = i+2
            if i == len(nums):
                return nums[len(nums)]
            else:
                return nums[i]