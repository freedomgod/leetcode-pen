1. 2022年3月27日——[找出缺失的观测数据](https://leetcode-cn.com/problems/find-missing-observations/)

   ```python
   class Solution:
       def missingRolls(self, rolls: List[int], mean: int, n: int) -> List[int]:
           m = len(rolls)
           sum_rolls = sum(rolls)
           total = mean * (m + n)  # m+n次点数总和
           round_rolls = total - sum_rolls  # 缺失的n次点数总和
           if (round_rolls < n) or (round_rolls > 6 * n):  # 如果剩余的点数不足以分成n次或者一定会超过6则不存在答案
               return []
           avg = round_rolls // n  # 一种答案就是把剩余点数均分到n次
           rnd = round_rolls % n   # 然后把多余的点数加到前rnd次
           ans = [avg] * n
           for i in range(rnd):
               ans[i] += 1
           return ans
   # 时间复杂度为O(n+m)
   # 空间复杂度O(1)
   ```




2. 2022年3月28日——[交替位二进制数](https://leetcode-cn.com/problems/binary-number-with-alternating-bits/)

   ```python
   class Solution:
       def hasAlternatingBits(self, n: int) -> bool:
           # bin_str = bin(n)[2:]  # 用内置函数将整数n转换为二进制
           def to_binary(x):  # 或者直接写一个求二进制的函数
               bin_s = ""
               while True:
                   bin_s = str(x % 2) + bin_s
                   x //= 2
                   if x == 0:
                       return bin_s
           bin_str = to_binary(n)
           for i in range(len(bin_str)-1):  # 循环二进制字符串
               if bin_str[i] == bin_str[i+1]:  # 判断是否交替出现
                   return False
           return True
   # 时间复杂度O(log n)
   # 空间复杂度O(log n)
       
   # 还有其他更好的解法
   # 官方题解，不需要完全求出整数的二进制再判断是否交替，在求二进制的过程中就进行判断
   class Solution:
       def hasAlternatingBits(self, n: int) -> bool:
           prev = 2  # 记录前一位的值，初始时可以随便设置一个非二进制的值
           while n:
               cur = n % 2  # 当前的二进制值
               if cur == prev:
                   return False
               prev = cur
               n //= 2
           return True
   # 时间复杂度O(log n)
   # 空间复杂度O(1)
   
   # 位运算 涉及到位运算的我都比较头疼，看不太懂
   # 对输入 n 的二进制表示右移一位后，得到的数字再与 n 按位异或得到 a。当且仅当输入 n 为交替位二进制数时，a 的二进制表示全为 1（不包括前导 0）。这里进行简单证明：当 a 的某一位为 1 时，当且仅当 n 的对应位和其前一位相异。当 a 的每一位为 1 时，当且仅当 n 的所有相邻位相异，即 n 为交替位二进制数。
   # 将 a 与 a + 1 按位与，当且仅当 a 的二进制表示全为 1 时，结果为 0。这里进行简单证明：当且仅当 a 的二进制表示全为 1 时，a + 1 可以进位，并将原最高位置为 0，按位与的结果为 0。否则，不会产生进位，两个最高位都为 1，相与结果不为 0。
   # 结合上述两步，可以判断输入是否为交替位二进制数。
   class Solution:
       def hasAlternatingBits(self, n: int) -> bool:
           a = n ^ (n >> 1)
           return a & (a + 1) == 0
   # 时间复杂度O(1)
   # 空间复杂度O(1)
   
   ```

   

3. 2022年3月29日——[考试的最大困扰度](https://leetcode-cn.com/problems/maximize-the-confusion-of-an-exam/)

   ```python
   class Solution:
       def subs(self, st, t, f, k):  # 计算字符串在ix位置以后，修改k次的最大连续相同题数
           """计算一个字符串中T和F的个数，如果有其中一个小于等于k，则说明可以通过k次修改把字符串改为全部连续，依次从字符串左边和右边去除字符串让计数靠近k，递归求解取长度最长的"""
           if t <= k or f <= k:  # 可以变为连续相同的字符串
               return t + f
           left = self.subs(st[1:], t-1 if st[0]=='T' else t, f-1 if st[0]=='F' else f, k)
           right = self.subs(st[:-1], t-1 if st[-1]=='T' else t, f-1 if st[-1]=='F' else f, k)
           return max(left, right)
           
       def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
           n = len(answerKey)  # 长度
           if n == 1 or k == n:  # 特殊情况判断
               return n
           
           cnt = Counter(answerKey)  # 记录T和F的个数
           max_num = self.subs(answerKey, cnt['T'], cnt['F'], k)  # 最大值
           return max_num
   # 这样容易理解，但是当字符串长度很长时，使用递归效率很低，会超时。
   # 用滑动窗口的思想，从左向右遍历，维护一个区间内某个字符(T或F)的数量不超过k，超过了则左端点右移。
   
   class Solution:
       def maxConsecutiveAnswers(self, answerKey: str, k: int) -> int:
           n = len(answerKey)  # 长度
           if n == 1 or k == n:  # 特殊情况判断
               return n
           def consecutive_ch(ch):  # 判断字符串中ch字符连续的最大长度，k相当于容错
               left, right = 0, 0  # 维护区间的左右序号
               max_num = 0  # 记录最长长度
               other_ch_num = 0  # 导致非连续的字符个数
               while right < n:
                   other_ch_num += answerKey[right] != ch
                   while other_ch_num > k:
                       other_ch_num -= answerKey[left] != ch
                       left += 1
                   
                   max_num = max(max_num, right - left + 1)  #返回最长长度
                   right += 1
               return max_num
           return max(consecutive_ch("T"), consecutive_ch("F"))
   # 时间复杂度O(n)
   # 空间复杂度O(1)
   
   ```

   









