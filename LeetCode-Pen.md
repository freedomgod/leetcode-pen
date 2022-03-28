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

   











