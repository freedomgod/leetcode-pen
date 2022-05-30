# 2022年

## 3月

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

   

4. 2022年3月30日——[找到处理最多请求的服务器](https://leetcode-cn.com/problems/find-servers-that-handled-most-number-of-requests/submissions/)

   ```python
   class Solution:
       def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
           server_map = {i: [0, 0] for i in range(k)}  # 创建一个服务器的字典，保存每个服务器当前处理的任务数量和最后一个任务完成的时间点
           n = len(arrival)
           for i in range(n):
               pre_server = i % k  # 预选的服务器
               if server_map[pre_server][1] <= arrival[i]:  # 判断预选的服务器是否有空闲处理任务i
                   server_map[pre_server][0] += 1  # 能够处理任务i，预选服务器处理的任务数量加1
                   server_map[pre_server][1] = arrival[i] + load[i]  # 最后完成任务的时间点
               else:  # 不能处理当前任务，需要寻找下一个空闲服务器
                   for j in range(k-1):  
                       pre_server = (pre_server + 1) % k  # 下一个服务器
                       if server_map[pre_server][1] <= arrival[i]:  # 判断预选的服务器是否有空闲处理任务i
                           server_map[pre_server][0] += 1  # 能够处理任务i，预选服务器处理的任务数量加1
                           server_map[pre_server][1] = arrival[i] + load[i]  # 最后完成任务的时间点
                           break  # 当前任务得到处理，退出这层循环
           max_num = max(server_map.values())[0]  # 处理任务最多的数量
           return [i for i in server_map if server_map[i][0] == max_num] # 返回列表
   # 这算是一种直译吧，
   # 时间复杂度O(kn)
   # 空间复杂度O(k)
   # 但是会超时，需要优化优化或者换一种做法
   
   class Solution:
       def busiestServers(self, k: int, arrival: List[int], load: List[int]) -> List[int]:
           server_map = {i: 0 for i in range(k)}  # 创建一个服务器的字典，保存每个服务器当前处理的任务数量
           available = list(range(k))  # 可用的空闲的服务器，作为一个优先队列
           busy = []  # 忙碌的服务器队列，保存服务器完成任务的时间点和服务器id
           n = len(arrival)
           for i in range(n):
               while busy and busy[0][0] <= arrival[i]:  # 把busy中可以执行任务i的服务器移除
                   _, idx = heapq.heappop(busy)  # 弹出，表示idx服务器变为空闲
                   heapq.heappush(available, i + (idx - i) % k)  # 把idx服务器加入到空闲服务器队列，并且关键就在于服务器的编号是什么，这个编号要不小于i且同余于idx，这样才能在保证available中，编号小于i的空闲服务器能排到编号不小于i mod k的空闲服务器后面
               if available:  # 表示有空闲的服务器
                   idx = heapq.heappop(available) % k   # 空闲服务器的编号
                   server_map[idx] += 1  # 服务器处理任务的数量加1
                   heapq.heappush(busy, (arrival[i] + load[i], idx))  # 加入到busy队列中
           max_num = max(server_map.values())  # 处理任务最多的数量
           return [i for i in server_map if server_map[i] == max_num] # 返回列表
   # 这里busy和available使用heapq来维护的优先队列，难点在available中服务器的编号，和前面相比大循环都是任务的数量n，内循环有一个堆维护的log k
   # 时间复杂度O((k+n)log k)
   # 空间复杂度O(k)
   ```

   

5. 2022年3月31日——[自除数](https://leetcode-cn.com/problems/self-dividing-numbers/)

   ```python
   class Solution:
       def selfDividingNumbers(self, left: int, right: int) -> List[int]:
           ans = []
           for a in range(left, right+1):
               str_a = str(a)  # 转换为字符串
               p = 1  # 状态，用于判断是否是自除数
               for sa in str_a:
                   if sa == '0' or (a % int(sa)):  # 为0或不能整除则不是自除数
                       p = 0
                       break
               if p:
                   ans.append(a)
           return ans
   # 直译即可
   # 时间复杂度O(nk)，n表示从left到right的区间长度，k为数值平均长度
   # 空间复杂度O(1)
   ```

   

## 4月

6. 2022年4月1日——[二倍数对数组](https://leetcode-cn.com/problems/array-of-doubled-pairs/)

   ```python
   class Solution:
       def canReorderDoubled(self, arr: List[int]) -> bool:
           if not arr:  # arr为空返回false
               return False
           a = arr.copy()
           a.sort(key=lambda x: abs(x))  # 数组按绝对值排序
           while a:
               a0 = a.pop(0)  # 弹出一个最小的值
               if 2 * a0 in a:  # 判断2倍的值是否在数组中
                   a.remove(2 * a0)  # 移除，在数组中移除一个元素要O(n)时间复杂度，需要优化
               else:
                   break  # 说明有值不能找到两倍的值
           if a:
               return False
           else:
               return True
   # 这种做法可能在判断2 * a0是否在a数组中和移除2 * a0元素时要花费较多时间，所以会超时，需要换一种操作
   
   class Solution:
       def canReorderDoubled(self, arr: List[int]) -> bool:
           if not arr:  # arr为空返回false
               return False
           cnt = Counter(arr)  # 对数组arr计数
           if cnt[0] % 2:  # 0的个数为奇数说明0无法匹配
               return False
           for k in sorted(cnt, key=abs):  # cnt按照键的绝对值排序
               if cnt[k] > cnt[2 * k]:  # k的数量大于2*k说明有部分的k没有足够的2*k匹配
                   return False
               cnt[2 * k] -= cnt[k]  # 减去已匹配的数量
           return True
   # 时间复杂度：O(nlogn)，其中 n 是数组 arr 的长度。最坏情况下哈希表中有 n 个元素，对其排序需要 O(nlogn) 的时间。
   # 空间复杂度：O(n)。最坏情况下哈希表中有 n 个元素，需要 O(n) 的空间。
   
   
   
   ```

   


7. 2022年4月2日——[强密码检验器](https://leetcode-cn.com/problems/strong-password-checker/)

   ```python
   class Solution:
       def get_continuous(self, ss):
           """
           从ss中统计连续出现三次的字符数量，并返回需要修改的次数，以及3减去包含字符种类数
           """
           contains, continuous = [0] * 3, []
           i = 0
           while not all(contains):
               if i >= len(ss):
                   break
               c = ss[i]
               if contains[0] == 0 and c in string.ascii_lowercase:
                   contains[0] = 1
               if contains[1] == 0 and c in string.ascii_uppercase:
                   contains[1] = 1
               if contains[2] == 0 and c in string.digits:
                   contains[2] = 1
               i += 1
           if len(ss) < 3:
               return (3 - sum(contains), [0])
           i, j = 0, 1
           while True:
               if j >= len(ss):
                   if j - i >= 3:
                       continuous.append(j - i)
                   break
               if ss[i] == ss[j]:
                   j += 1
               else:
                   if j - i >= 3:
                       continuous.append(j - i)
                   i, j = j, j + 1
           return (3 - sum(contains), continuous)
       def strongPasswordChecker(self, password: str) -> int:
           L = len(password)
           contains, continues = self.get_continuous(password)
           continuous = sum([a // 3 for a in continues])
           if 6 <= L <= 20:
               if contains <= 0 and continuous == 0:
                   return 0
               elif contains >= continuous:
                   return contains
               else:
                   return continuous
           elif L > 20:
               diff_l = L - 20
               cnt3 = sum([1 for a in continues if a % 3 == 0])
               if cnt3 <= diff_l:
                   continues = [a - 1 for a in continues if a % 3 == 0 and a - 1 >= 3]
                   rem = diff_l - cnt3
                   continues2 = []
                   for j, a in enumerate(continues):
                       if a > rem:
                           continues2.append(a - rem)
                           continues2 += continues[j+1:]
                           break
                       elif a == rem:
                           continues2 += continues[j+1:]
                           break
                       else:
                           rem = rem - a
                   continues = continues2
                   return diff_l + max(contains, sum([a // 3 for a in continues]))
               else:
                   continues = [a - 1 for a in continues.sort()[diff_l+1] if a % 3 == 0 and a - 1 >= 3] + continues[diff_l+1:]
                   return diff_l + max(contains, sum([a // 3 for a in continues]))
               # continues = [a - 1 for a in continues if a % 3 == 0 and a - 1 >= 3]
               # if diff_l >= continuous:
               #     continues = [a - (diff_l - cnt3) if a >= (diff_l - cnt3) for a in continues]
               #     return cnt3 + max(contains, sum([a // 3 for a in continues]))
               #     # return diff_l + contains if contains > 0 else 0
               # else:
               #     if cnt3 >= diff_l:
               #         return sum([a // 3 for a in continues])
               #     return diff_l + max(continuous - diff_l, contains)
           else:
               diff_l = 6 - L
               return max(diff_l, contains, continuous)
   # 分情况模拟就行，但是情况有些许复杂，上面代码没有完善，后面有时间完成一下，先抄一抄别的大佬的代码
   
   class Solution:
       def strongPasswordChecker(self, password: str) -> int:
           n = len(password)
           has_lower = has_upper = has_digit = False
           for ch in password:
               if ch.islower():
                   has_lower = True
               elif ch.isupper():
                   has_upper = True
               elif ch.isdigit():
                   has_digit = True
           
           categories = has_lower + has_upper + has_digit
   
           if n < 6:
               return max(6 - n, 3 - categories)
           elif n <= 20:
               replace = cnt = 0
               cur = "#"
   
               for ch in password:
                   if ch == cur:
                       cnt += 1
                   else:
                       replace += cnt // 3
                       cnt = 1
                       cur = ch
               
               replace += cnt // 3
               return max(replace, 3 - categories)
           else:
               # 替换次数和删除次数
               replace, remove = 0, n - 20
               # k mod 3 = 1 的组数，即删除 2 个字符可以减少 1 次替换操作
               rm2 = cnt = 0
               cur = "#"
   
               for ch in password:
                   if ch == cur:
                       cnt += 1
                   else:
                       if remove > 0 and cnt >= 3:
                           if cnt % 3 == 0:
                               # 如果是 k % 3 = 0 的组，那么优先删除 1 个字符，减少 1 次替换操作
                               remove -= 1
                               replace -= 1
                           elif cnt % 3 == 1:
                               # 如果是 k % 3 = 1 的组，那么存下来备用
                               rm2 += 1
                           # k % 3 = 2 的组无需显式考虑
                       replace += cnt // 3
                       cnt = 1
                       cur = ch
               
               if remove > 0 and cnt >= 3:
                   if cnt % 3 == 0:
                       remove -= 1
                       replace -= 1
                   elif cnt % 3 == 1:
                       rm2 += 1
               
               replace += cnt // 3
   
               # 使用 k % 3 = 1 的组的数量，由剩余的替换次数、组数和剩余的删除次数共同决定
               use2 = min(replace, rm2, remove // 2)
               replace -= use2
               remove -= use2 * 2
               # 由于每有一次替换次数就一定有 3 个连续相同的字符（k / 3 决定），因此这里可以直接计算出使用 k % 3 = 2 的组的数量
               use3 = min(replace, remove // 3)
               replace -= use3
               remove -= use3 * 3
               return (n - 20) + max(replace, 3 - categories)
   
   ```

   

8. 2022年4月3日——[寻找比目标字母大的最小字母](https://leetcode-cn.com/problems/find-smallest-letter-greater-than-target/)

   ```python
   class Solution:
       def nextGreatestLetter(self, letters: List[str], target: str) -> str:
           """
           看到要在排序后的字符列表里找目标字符，第一想到的就是二分查找了，只需要处理好细节方面
           """
           left, right = 0, len(letters) - 1  # 初始时定义左边和右边的指针
           mid = (left + right) // 2  # 中间位置
           while left < right:  # 循环条件
               if target > letters[mid]:  # 在右边部分则把left指针指向中间位置
                   left = mid + 1
               elif target == letters[mid]:  # 如果相等则直接退出
                   break
               else:
                   right = mid - 1
               mid = abs(left + right) // 2
           n = len(letters)
           while True:   # 要判断mid位置的情况
               if mid < n:
                   if target > letters[mid]:  # 如果目标字符更大，则返回下一个位置的字符，因为下一个位置的字符一定比目标字符大，但是要判断位置是否合法，不合法则返回第一个位置的字符
                       return letters[mid+1] if (mid+1) < n else letters[0]
                   elif target < letters[mid]:  # 目标字符更小，则该位置的字符就是第一个比目标字符大的字母
                       return letters[mid]
                   else:
                       mid += 1  # 和目标字符相等的情况下要过滤掉
               else:
                   return letters[0]
   # 要注意自己写的二分查要做好细节方面
   # 时间复杂度：O(log n)
   # 空间复杂度：O(1)
   
   class Solution:
       def nextGreatestLetter(self, letters: List[str], target: str) -> str:
           return letters[bisect_right(letters, target)] if target < letters[-1] else letters[0]
   # 这是大佬的利用已有的二分查找库函数的代码，太简单了
   ```




9. 2022年4月4日——[区域和检索](https://leetcode-cn.com/problems/range-sum-query-mutable/)

   ```python
   class NumArray:
   
       def __init__(self, nums: List[int]):
           self.nums = nums
   
       def update(self, index: int, val: int) -> None:
           self.nums[index] = val  # 更新值
   
       def sumRange(self, left: int, right: int) -> int:
           return sum(self.nums[left:right+1])  # 每次求和都会做很多重复工作，时间复杂度较大
   
   # 假设调用 update 和 sumRange 方法次数分别为m、n次，left到right区间的平均长度是k
   # 时间复杂度：O(m + nk)
   # 空间复杂度：O(1)
           
   
   # 改进
   class NumArray:
   
       def __init__(self, nums: List[int]):
           self.nums = nums
           self.sums = list(accumulate(nums))  # 累加，每个位置对应的是nums的前i个数的和
   
       def update(self, index: int, val: int) -> None:
           diff_value = val - self.nums[index]
           self.nums[index] = val  # 更新值
           self.sums[index:] = map(lambda x: x + diff_value, self.sums[index:])  # 更新sums
   
       def sumRange(self, left: int, right: int) -> int:
           if left == 0:
               return self.sums[right]
           else:
               return self.sums[right] - self.sums[left-1]
   
   # 假设调用 update 和 sumRange 方法次数分别为m、n次，index平均值为k
   # 时间复杂度：O(mk)
   # 空间复杂度：O(L)
   
   class NumArray:
   
       def __init__(self, nums: List[int]):
           self.nums = nums
           self.L = len(nums)
           self.size = int(self.L ** .5)  # 取每个块的大小为根号n
           self.sums = [0] * ((self.L + self.size - 1) // self.size)  # 块的数量，每个块记录其和
           for i in range(self.L):
               self.sums[i // self.size] += self.nums[i]
   
       def update(self, index: int, val: int) -> None:
           diff_value = val - self.nums[index]
           self.nums[index] = val  # 更新值
           self.sums[index // self.size] += diff_value  # 更新sums
   
       def sumRange(self, left: int, right: int) -> int:
           ls = left // self.size
           rs = right // self.size
           if ls == rs:
               return sum(self.nums[left:right+1])
           else:
               return sum(self.nums[left:(ls + 1) * self.size]) + sum(self.sums[ls+1:rs]) + sum(self.nums[rs * self.size:right+1])
   
   # 时间复杂度：构造函数为 O(n)，update 函数为 O(1)，sumRange 函数为 O(sqrt(n))，其中 n 为数组 nums 的大小。对于sumRange 函数，我们最多遍历两个块以及 sum 数组，因此时间复杂度为 O(sqrt(n))。
   # 空间复杂度：O(sqrt(n))
   # 这是分块的思想，相比我前面写的两种，本质还是暴力，没有节省多少时间，而这里更新索引的值时，不用更新所有的和，而是只需更新块的和，效率更高
   
   ```

   

10. 2022年4月5日——[二进制表示中质数个计算置位](https://leetcode-cn.com/problems/prime-number-of-set-bits-in-binary-representation/)

    ```python
    class Solution:
        def countPrimeSetBits(self, left: int, right: int) -> int:
            def isprime(x):  # 判断是否是素数
                if x < 2:
                    return False
                i = 2
                while i ** 2 <= x:
                    if x % i == 0:
                        return False
                    i += 1
                return True
            
            def int2bin(x):  # 也可以尝试手写一个二进制转换
                if x == 0:
                    return '0'
                ans = ''
                while x:
                    ans = str(x % 2) + ans
                    x = x >> 1
                return ans
            
            cnt = 0  # 计数
            for a in range(left, right + 1):
                b = bin(a)[2:]  # 整数转二进制
                if isprime(sum(map(lambda x: int(x), b))):  # 判断1的个数是否为素数
                    cnt += 1
            return cnt
    # 时间复杂度：数n的二进制位数是int(log2n) + 1，判断一个数是否为素数的时间复杂度是sqrt(x)，所以判断一个区间L的时间复杂度大概为O(L*sqrt(log2n))
    # 空间复杂度：O(1)
    # 这个直接做差点超时，需要优化
    
    
    class Solution:
        def countPrimeSetBits(self, left: int, right: int) -> int:
            def isprime(x):  # 判断是否是素数
                if x < 2:
                    return False
                i = 2
                while i ** 2 <= x:
                    if x % i == 0:
                        return False
                    i += 1
                return True
            
            def int2bin(x):  # 也可以尝试手写一个二进制转换
                if x == 0:
                    return '0'
                ans = ''
                while x:
                    ans = str(x % 2) + ans
                    x = x >> 1
                return ans
            cnt = 0  # 计数
            dic = {}  # 创建一个字典，记录是否是素数，减少判断素数的时间
            for a in range(left, right + 1):
                b = bin(a)[2:]  # 整数转二进制
                num = sum(map(lambda x: int(x), b))  # 1的个数
                try:
                    if dic[num]:  # num在字典中，则直接取值判断是否是素数
                        cnt += 1
                except KeyError:
                    dic[num] = isprime(num)  # num不在字典中，则判断是否素数并保存结果
                    if dic[num]:
                        cnt += 1
            return cnt
    # 这样改了一点，结果优化的效果很有限
    
    class Solution:
        def countPrimeSetBits(self, left: int, right: int) -> int:
            def isprime(x):  # 判断是否是素数
                if x < 2:
                    return False
                i = 2
                while i ** 2 <= x:
                    if x % i == 0:
                        return False
                    i += 1
                return True
    
            return sum(isprime(bin(x).count('1')) for x in range(left, right + 1))  # 不啰嗦，直接一个列表推导式求和就是结果
    # 这个本质上是一样的解法，不过用时居然比我前面写的快了很多，感觉原因可能是语句太啰嗦了，还多执行了sum求和函数
    # 还有一种利用位运算判断质数的方法，效率更高。
    ```

    

11. 2022年4月6日——[最小高度树](https://leetcode-cn.com/problems/minimum-height-trees/)

    ```python
    class Solution:
        def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
            if n == 1:
                return [0]
            dic = {}
            for a, b in edges:
                if a not in dic:
                    dic[a] = [b]
                else:
                    dic[a].append(b)
                if b not in dic:
                    dic[b] = [a]
                else:
                    dic[b].append(a)
            
            def dfs(start):
                ans = []
                q = deque([(start, 0)])  # 队列保存的是节点和根节点到该节点的距离或者说深度
                vis = {start}  # 记录遍历过的节点
                while q:  # 广度优先搜索
                    k = len(q)
                    for _ in range(k):
                        nd, d = q.popleft()
                        ans.append((nd, d))
                        for x in dic[nd]:
                            if x not in vis:
                                vis.add(x)
                                q.append((x, d + 1))
                return ans[-1][1]
            
            depth = []  # 难点在于如何把一种情况推广到所有节点为根节点的情况，找最短高度
            for i in range(n):
                depth.append(dfs(i))
            return [i for i, x in enumerate(depth) if x == min(depth)]
    # 时间复杂度：O(n^2)
    # 空间复杂度：O(n)
    
    
    class Solution:
        def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:
            if n == 1:
                return [0]
            dic = {}
            for a, b in edges:
                if a not in dic:
                    dic[a] = [b]
                else:
                    dic[a].append(b)
                if b not in dic:
                    dic[b] = [a]
                else:
                    dic[b].append(a)
            
            parents = [0] * n
            def dfs(start):
                ans = []
                q = deque([(start, 0)])  # 队列保存的是节点和根节点到该节点的距离或者说深度
                vis = {start}  # 记录遍历过的节点
                while q:  # 广度优先搜索
                    k = len(q)
                    for _ in range(k):
                        nd, d = q.popleft()
                        ans.append((nd, d))
                        for x in dic[nd]:
                            if x not in vis:
                                vis.add(x)
                                q.append((x, d + 1))
                                parents[x] = nd  # 记录的是x的父节点nd
                return ans[-1][0]
            
            x = dfs(0)  # 0为根节点，返回的x为最大深度的末尾的节点
            y = dfs(x)  # 再遍历x为根节点的情况，找最长的路径
            path = []   # 记录路径
            parents[x] = -1  # 赋值为-1是结束的条件，根节点无父节点
            while y != -1:
                path.append(y)
                y = parents[y]
            m = len(path)
            return [path[m // 2]] if m % 2 else [path[m // 2 - 1], path[m // 2]]
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

12. 2022年4月7日——[旋转字符串](https://leetcode-cn.com/problems/rotate-string/)

    ```python
    class Solution:
        def rotateString(self, s: str, goal: str) -> bool:
            n, m = len(s), len(goal)
            if n != m:
                return False
            for i in range(n):  # 直译即可
                if s[i+1:] + s[:i+1] == goal:   # 遍历字符串s把左边的部分旋转到右边和目标字符串比较
                    return True
            return False
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    
    class Solution:
        def rotateString(self, s: str, goal: str) -> bool:
            return len(s) == len(goal) and goal in s + s
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

13. 2022年4月8日——[N 叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/)

    ```python
    """
    # Definition for a Node.
    class Node:
        def __init__(self, val=None, children=None):
            self.val = val
            self.children = children
    """
    
    class Solution:
        def levelOrder(self, root: 'Node') -> List[List[int]]:
            if not root:  # 根节点为空
                return []
            q = deque([root])  # 队列，实现广度优先遍历，非递归实现
            res = []
            while q:
                n = len(q)
                cur = []  # 存储当前一层的节点值
                for _ in range(n):   # n代表了这一层的节点
                    nd = q.popleft()
                    cur.append(nd.val)
                    if nd.children:  # 孩子节点非空
                        q.extend(nd.children)  # 下一层的孩子节点加入队列
                res.append(cur)
            return res
    
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

14. 2022年4月9日——[到达终点](https://leetcode-cn.com/problems/reaching-points/)

    ```python
    class Solution:
        def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
            if sx > tx or sy > ty:
                return False
            q = deque([(sx, sy)])   # 广度优先搜索
            while q:
                n = len(q)
                for i in range(n):
                    x, y = q.popleft()
                    if x == tx and y == ty:  # 判断是否是终点
                        return True
                    elif x > tx or y > ty:   # 坐标过头了，不可能到达终点
                        continue
                    else:
                        q.extend([(x, x + y), (x + y, y)])  # 添加下一个可能的位置
            return False
    # 很明显广搜虽然能解，但是每个位置都有两个方向搜索，差不多是2^n，起点和终点距离远一点就会花费很长时间搜索
    
    
    class Solution:
        def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
            if sx > tx or sy > ty:  # 判断初始终点位置大于起点
                return False
            while sx <= tx and sy <= ty:  # 循环条件
                if sx == tx and sy == ty:  # 判断终点
                    return True
                if tx > ty:   # 判断是从哪种情况的转换，还原对应的坐标即可
                    tx, ty = tx - ty, ty
                elif tx < ty:
                    tx, ty = tx, ty - tx
                else:
                    return False
            return False
        
    # 这里是利用新转换后的坐标x,y不可能相等，直接从终点开始反推，看能不能得到起点，那么时间复杂度取决于起点和终点较大的区间距离
    # 但也有极端的情况，使得在反推坐标的时候，重复多做了无意义的循环，所以还要再改下
    
    class Solution:
        def reachingPoints(self, sx: int, sy: int, tx: int, ty: int) -> bool:
            if sx > tx or sy > ty:  # 判断初始终点位置大于起点
                return False
            while sx <= tx and sy <= ty:  # 循环条件
                if sx == tx and sy == ty:  # 判断终点
                    return True
                if tx > ty:   # 判断是从哪种情况的转换，还原对应的坐标即可
                    k = (tx - sx) // ty  # 求出一个k，让新的tx、ty仍然是大于等于起点坐标的
                    if k:
                        tx, ty = tx - ty * k, ty
                    else:  # 如果k为0，说明下一步的坐标必定会小于起点坐标
                        return False
                elif tx < ty:
                    k = (ty - sy) // tx
                    if k:
                        tx, ty = tx, ty - tx * k
                    else:
                        return False
                else:
                    return False
            return False
    
    # 时间复杂度：emmm，好像是O(log(max(tx, ty)))
    # 空间复杂度：O(1)
    ```

    

15. 2022年4月10日——[唯一摩尔斯密码词](https://leetcode-cn.com/problems/unique-morse-code-words/)

    ```python
    class Solution:
        def uniqueMorseRepresentations(self, words: List[str]) -> int:
            alph = string.ascii_lowercase
            mose = [".-","-...","-.-.","-..",".","..-.","--.","....","..",".---","-.-",".-..","--","-.","---",".--.","--.-",".-.","...","-","..-","...-",".--","-..-","-.--","--.."]
            d = dict(zip(alph, mose))  # 建立字母关于摩尔斯密码的字典
            vis = set()  # 保存单词翻译的结果
            for w in words:  # 遍历每个单词
                trans = ""
                for c in w:
                    trans += d[c]
                if trans not in vis:  # 不同的单词翻译
                    vis.add(trans)
            return len(vis)
    
    # 设words长度为n，每个单词平均长度为m，或者说words中单词总长度为S
    # 时间复杂度：O(n*m) / O(S)
    # 空间复杂度：O(n)   / O(S)
    ```

    

16. 2022年4月11日——[统计各位数字都不同的数字个数](https://leetcode-cn.com/problems/count-numbers-with-unique-digits/)

    ```python
    def countnum():
        d = [1]
        for i in range(1, 9):
            tmp = 0
            b = [10 ** (i - 1), 10 ** i]
            for j in range(b[0], b[1]):
                if len(set(str(j))) < i:
                    tmp += 1
            d.append(b[1] - b[0] - tmp + d[-1])
        return d
    
    class Solution:
        def countNumbersWithUniqueDigits(self, n: int) -> int:
            countnum = [1, 10, 91, 739, 5275, 32491, 168571, 712891, 2345851]  # 打表
            return countnum[n]
        
    # 直接暴力会超时，但是因为n的范围是0~8，所以可以在IDE或其他地方把所有情况求出来然后打表
    
    
    class Solution:
        def countNumbersWithUniqueDigits(self, n: int) -> int:
            # countnum = [1, 10, 91, 739, 5275, 32491, 168571, 712891, 2345851]  # 打表
            # return countnum[n]
            if n == 0:
                return 1
            ans = 10  # 记录只有一位数的情况为10种
            cur = 1  # 记录末位要乘的数
            for i in range(2, n+1):  # 找每个位数的情况相加
                cur *= (10 - i + 1)  # 9*9 + 9*9*8 + 9*9*8*7 + ……
                ans += 9 * cur
            return ans
        
    # 实际上这个问题是一个组合问题，第一个位数只能从9个数选择，剩下的位数可选的要依次减一
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    ```

    

17. 2022年4月12日——[写字符串需要的行数](https://leetcode-cn.com/problems/number-of-lines-to-write-string/)

    ```python
    class Solution:
        def numberOfLines(self, widths: List[int], s: str) -> List[int]:
            dic = dict(zip(string.ascii_lowercase, widths))  # 把字母所需单位映射到字典
            max_len = 100  # 每行最大单位
            cur_len = 0  # 当前行的长度
            i, n = 0, len(s)
            ans = [1, 0]  # 结果
            while i < n:
                if (cur_len + dic[s[i]]) <= max_len:  # 当前行还能写字母
                    cur_len += dic[s[i]]  # 增加长度
                    ans[1] = cur_len  # 更新结果
                else:
                    cur_len = dic[s[i]]  # 开启新的一行
                    ans[1] = cur_len
                    ans[0] += 1
                i += 1
            return ans
        
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    ```

    

18. 2022年4月13日——[O(1) 时间插入、删除和获取随机元素](https://leetcode-cn.com/problems/insert-delete-getrandom-o1/)

    ```python
    class RandomizedSet:
    
        def __init__(self):
            self.sets = set()  # 初始化集合
    
        def insert(self, val: int) -> bool:
            n = len(self.sets)  # 插入值之前集合的长度
            self.sets.add(val)  # 把val加入集合
            return len(self.sets) != n  # 如何插入后的长度和n不同同则返回True
    
        def remove(self, val: int) -> bool:
            try:
                self.sets.remove(val)  # 在集合中移除val
                return True
            except KeyError:
                return False  # 不存在则返回False
    
        def getRandom(self) -> int:
            return random.choice(list(self.sets))  # 使用随机函数choice随机选择集合中的一个值
    
    
    # Your RandomizedSet object will be instantiated and called as such:
    # obj = RandomizedSet()
    # param_1 = obj.insert(val)
    # param_2 = obj.remove(val)
    # param_3 = obj.getRandom()
    
    # 只是一般的实现方式，但我不确定是否是O(1)时间复杂度，因为在插入值的时候会求集合的长度。所以实际上要用哈希表+列表来实现
    
    class RandomizedSet:
        def __init__(self):
            self.nums = []
            self.indices = {}
    
        def insert(self, val: int) -> bool:
            if val in self.indices:
                return False
            self.indices[val] = len(self.nums)
            self.nums.append(val)
            return True
    
        def remove(self, val: int) -> bool:
            if val not in self.indices:
                return False
            id = self.indices[val]
            self.nums[id] = self.nums[-1]
            self.indices[self.nums[id]] = id
            self.nums.pop()
            del self.indices[val]
            return True
    
        def getRandom(self) -> int:
            return choice(self.nums)
        
    # 时间复杂度：O(1)
    # 空间复杂度：O(n)
    ```

    

19. 2022年4月14日——[最富有客户的资产总量](https://leetcode-cn.com/problems/richest-customer-wealth/)

    ```python
    class Solution:
        def maximumWealth(self, accounts: List[List[int]]) -> int:
            return max(map(lambda x: sum(x), accounts))
        
    # 时间复杂度：O(m*n)
    # 空间复杂度：O(1)
    ```

    

20. 2022年4月15日——[迷你语法分析器](https://leetcode-cn.com/problems/mini-parser/)

    ```python
    class Solution:
        def deserialize(self, s: str) -> NestedInteger:
            if s[0] != '[':
                return NestedInteger(int(s))
            stack, num, negative = [], 0, False
            for i, c in enumerate(s):
                if c == '-':
                    negative = True
                elif c.isdigit():
                    num = num * 10 + int(c)
                elif c == '[':
                    stack.append(NestedInteger())
                elif c in ',]':
                    if s[i-1].isdigit():
                        if negative:
                            num = -num
                        stack[-1].add(NestedInteger(num))
                    num, negative = 0, False
                    if c == ']' and len(stack) > 1:
                        stack[-2].add(stack.pop())
            return stack.pop()
    
    # 比较明显的类似于括号匹配一类的可以用栈解决的问题，可以用递归、栈来解决
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

21. 2022年4月16日——[最大回文数乘积](https://leetcode-cn.com/problems/largest-palindrome-product/)

    ```python
    class Solution:
        def largestPalindrome(self, n: int) -> int:
            if n == 1:
                return 9
            upper = 10 ** n - 1
            for left in range(upper, upper // 10, -1):  # 枚举回文数的左半部分
                p, x = left, left
                while x:
                    p = p * 10 + x % 10  # 翻转左半部分到其自身末尾，构造回文数 p
                    x //= 10
                x = upper
                while x * x >= p:
                    if p % x == 0:  # x 是 p 的因子
                        return p % 1337
                    x -= 1
    # 貌似只能枚举了，而枚举又要想办法是最大的，所以通过构造回文数，判断是否符合条件
    # 时间复杂度：O(10^2n)
    # 空间复杂度：O(1)
    ```

    

22. 2022年4月17日——[最常见的单词](https://leetcode-cn.com/problems/most-common-word/)

    ```python
    class Solution:
        def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
            i, j, n = 0, 0, len(paragraph)
            paragraph = paragraph.lower()  # 先转为小写字母
            words = {}  #  保存单词的计数
            while i < n:
                if paragraph[i].isalpha():   # 如果当前为字母，则向后找出这个单词
                    wd = ""
                    while j < n and paragraph[j].isalpha():
                        wd += paragraph[j]
                        j += 1
                    if wd not in banned:
                        if wd not in words:  # 加入到计数
                            words[wd] = 1
                        else:
                            words[wd] += 1
                    i = j
                i += 1
                j += 1
            return max(words, key=lambda x: words[x])
        
    # 主要就是把字符串中的单词分离出来计数，split方法就不够用了，因为不能按照空格和标点符号分割，也可以考虑用re正则表达式，但我还是直接遍历对单词计数
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    
    ```

    

23. 2022年4月18日——[字典序排数](https://leetcode-cn.com/problems/lexicographical-numbers/)

    ```python
    class Solution:
        def lexicalOrder(self, n: int) -> List[int]:
            return sorted(range(1, n + 1), key=lambda x: str(x))
    
        
    # 一行代码搞定，就是对[1, n]的数排序，key为数字的字典序
    # 时间复杂度：O(nlogn)
    # 空间复杂度：O(1)
    # 上面虽然简洁，但是不符合题目要求的O(n)复杂度
    
    
    class Solution:
        def lexicalOrder(self, n: int) -> List[int]:
            ans = [0] * n
            num = 1
            for i in range(n):
                ans[i] = num
                if num * 10 <= n:
                    num *= 10
                else:
                    while num % 10 == 9 or num + 1 > n:
                        num //= 10
                    num += 1
            return ans
    
        
    # 深度优先搜索
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    ```

    

24. 2022年4月19日——[字符的最短距离](https://leetcode-cn.com/problems/shortest-distance-to-a-character/)

    ```python
    class Solution:
        def shortestToChar(self, s: str, c: str) -> List[int]:
            n = len(s)
            ans = [n] * n
            que = []
            i = 0
            while i < n:  # 找出所有c字符的位置，作为搜索的起点
                ix = s.find(c, i)
                if ix != -1:
                    ans[ix] = 0
                    que.append((ix, 0))
                    i = ix + 1
                else:
                    break
            while que:              # 广度优先搜索
                k = len(que)
                for _ in range(k):
                    j, d = que.pop(0)
                    if j > 0 and (d + 1) < ans[j - 1]:
                        ans[j - 1] = d + 1
                        que.append((j - 1, d + 1))
                    if j < (n - 1) and (d + 1) < ans[j + 1]:
                        ans[j + 1] = d + 1
                        que.append((j + 1, d + 1))
            return ans 
    
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

25. 2022年4月20日——[文件的最长绝对路径](https://leetcode-cn.com/problems/longest-absolute-file-path/)

    ```python
    class Solution:
        def lengthLongestPath(self, input: str) -> int:
            ans = 0
            cur = ['']  # 存储每一级的目录
            sub = 0   # 记录子目录的级数
            i = 0
            n = len(input)
            while i < n:
                if input[i] == '\n':   # \n是\t开始的位置
                    i += 1
                    sub = 0
                    while i < n and input[i] == '\t':# 根据\t的个数判断下一个目录或文件的位置
                        sub += 1
                        i += 1
                else:
                    directory = ''
                    while i < n and (input[i].isdigit() or input[i].isalpha() or input[i] in ' .'):
                        directory += input[i]
                        i += 1
                    if sub == 0:   # 在一级目录
                        cur[0] = directory
                    else:     # 当前位置是更深一级的目录
                        cur = cur[:sub] + [directory]
                    if '.' in directory:   # 每加进一个目录判断是否是文件及长度是否更长
                        ans = max(ans, len('/'.join(cur)))
            return ans
    
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

26. 2022年4月21日——[山羊拉丁文](https://leetcode-cn.com/problems/goat-latin/)

    ```python
    class Solution:
        def toGoatLatin(self, sentence: str) -> str:
            ss = sentence.split(' ')
            ans = ''
            vowels = ['a', 'e', 'i', 'o', 'u']  # 元音字母
            for i, w in enumerate(ss, 1):
                if w[0].lower() in vowels:
                    w += 'ma'
                else:
                    w = w[1:] + w[0] + 'ma'
                w += 'a' * i
                ans += w + ' '
            return ans[:-1]
        
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    ```

    

27. 2022年4月22日——[旋转函数](https://leetcode-cn.com/problems/rotate-function/)

    ```python
    class Solution:
        def maxRotateFunction(self, nums: List[int]) -> int:
            ans = 0
            n = len(nums)
            if n == 1:
                return 0
            for i, a in enumerate(nums):
                F = sum(map(lambda x: nums[(x + i) % n] * x, range(n)))  # 旋转后的坐标，然后求和
                if F > ans:
                    ans = F
            return ans
        
    # 超时
    # 时间复杂度：O(n^2)
    # 空间复杂度：O(1)
    
    
    class Solution:
        def maxRotateFunction(self, nums: List[int]) -> int:
            f, n, numSum = 0, len(nums), sum(nums)
            for i, num in enumerate(nums):
                f += i * num
            res = f
            for i in range(n - 1, 0, -1):
                f = f + numSum - n * nums[i]  # 具有迭代公式
                res = max(res, f)
            return res
    
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    
    ```

    

28. 2022年4月23日——[安装栅栏](https://leetcode-cn.com/problems/erect-the-fence/)

    ```python
    class Line:
        def __init__(self, k, b):
            self.k = k
            self.b = b
        
        def get_y(self, z):
            return self.k * z + self.b
    
        def get_dist(self, pos):
            return abs(self.k * pos[0] + self.b - pos[1]) / sqrt(self.k ** 2 + 1)
    
    class Solution:
        def outerTrees(self, trees: List[List[int]]) -> List[List[int]]:
            n = len(trees)
            x = list(map(lambda z: z[0], trees))  # 把坐标的x和y分开
            y = list(map(lambda z: z[1], trees))
            bound = max(x), min(x), max(y), min(y)  # 找出树的x、y边界坐标
            if bound[0] == bound[1] or bound[2] == bound[3]:
                return trees
            left, right, up, bottom = [], [], [], []  # 存放上下左右边界的坐标
            ans, remainder = [], []  # 分别放结果边界的坐标和斜对角的边界坐标
            for i, t in enumerate(trees):
                isin = 0
                if t[0] == bound[0]:
                    right.append(trees[i])
                    isin = 1
                elif t[0] == bound[1]:
                    left.append(trees[i])
                    isin = 1
                if t[1] == bound[2]:
                    up.append(trees[i])
                    isin = 1
                elif t[1] == bound[3]:
                    bottom.append(trees[i])
                    isin = 1
                if isin:
                    ans.append(t)
                else:
                    remainder.append(t)
            left_up = [max(left, key=lambda z: z[1]), min(up, key=lambda z: z[0])]  # 用两点确定斜对角直线
            left_bottom = [min(left, key=lambda z: z[1]), min(bottom, key=lambda z: z[0])]
            right_up = [max(right, key=lambda z: z[1]), max(up, key=lambda z: z[0])]
            right_bottom = [min(right, key=lambda z: z[1]), max(bottom, key=lambda z: z[0])]
            line = []
            for c in [left_up, left_bottom, right_up, right_bottom]:
                try:
                    k1 = (c[1][1] - c[0][1]) / (c[1][0] - c[0][0])
                    line.append(Line(k1, c[0][1] - k1 * c[0][0]))
                except ZeroDivisionError:
                    line.append(None)
            point = [([], -1) for _ in range(len(line))]
            for pos in remainder:
            # def judge(pos):  # 判断坐标是否在斜对角线上或处在线外
                if line[0]:
                    if pos[1] >= line[0].get_y(pos[0]):
                        d = line[0].get_dist(pos)
                        if d > point[0][1]:
                            point[0] = ([pos[0], pos[1]], d)
                if line[1]:
                    if pos[1] <= line[1].get_y(pos[0]):
                        d = line[1].get_dist(pos)
                        if d > point[1][1]:
                            point[1] = ([pos[0], pos[1]], d)
                if line[2]:
                    if pos[1] >= line[2].get_y(pos[0]):
                        d = line[2].get_dist(pos)
                        if d > point[2][1]:
                            point[2] = ([pos[0], pos[1]], d)
                if line[3]:
                    if pos[1] <= line[3].get_y(pos[0]):
                        d = line[3].get_dist(pos)
                        if d > point[3][1]:
                            point[3] = ([pos[0], pos[1]], d)
            ans.extend(list(map(lambda z: z[0], filter(lambda q: q[0], point))))
            return ans
    # 答案错误😴
    class Solution:
        def outerTrees(self, trees: List[List[int]]) -> List[List[int]]:
            def cross(p: List[int], q: List[int], r: List[int]) -> int:
                return (q[0] - p[0]) * (r[1] - q[1]) - (q[1] - p[1]) * (r[0] - q[0])
    
            n = len(trees)
            if n < 4:
                return trees
    
            # 按照 x 从小到大排序，如果 x 相同，则按照 y 从小到大排序
            trees.sort()
    
            hull = [0]  # hull[0] 需要入栈两次，不标记
            used = [False] * n
            # 求凸包的下半部分
            for i in range(1, n):
                while len(hull) > 1 and cross(trees[hull[-2]], trees[hull[-1]], trees[i]) < 0:
                    used[hull.pop()] = False
                used[i] = True
                hull.append(i)
            # 求凸包的上半部分
            m = len(hull)
            for i in range(n - 2, -1, -1):
                if not used[i]:
                    while len(hull) > m and cross(trees[hull[-2]], trees[hull[-1]], trees[i]) < 0:
                        used[hull.pop()] = False
                    used[i] = True
                    hull.append(i)
            # hull[0] 同时参与凸包的上半部分检测，因此需去掉重复的 hull[0]
            hull.pop()
    
            return [trees[i] for i in hull]
    # 求凸包的算法，只懂大概的意思，但是不会写，只copy了一份代码
    # 时间复杂度：O(nlogn)
    # 空间复杂度：O(n)
    ```

    

29. 2022年4月24日——[二进制间距](https://leetcode-cn.com/problems/binary-gap/)

    ```python
    class Solution:
        def binaryGap(self, n: int) -> int:
            bn = bin(n)[2:]
            l = len(bn)
            ans = 0
            i = bn.find('1')
            if i == -1:
                return 0
            while i < l:
                ix = bn.find('1', i+1)
                if ix != -1:
                    ans = max(ix - i, ans)
                    i = ix
                else:
                    break
            return ans
    
    # 漏了一天
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    ```

    

30. 2022年4月25日——[随机数索引](https://leetcode-cn.com/problems/random-pick-index/)

    ```python
    class Solution:
        def __init__(self, nums: List[int]):
            self.pos = defaultdict(list)
            for i, num in enumerate(nums):
                self.pos[num].append(i)
    
        def pick(self, target: int) -> int:
            return choice(self.pos[target])
    
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

31. 2022年4月26日——[三维形体投影面积](https://leetcode-cn.com/problems/projection-area-of-3d-shapes/)

    ```python
    class Solution:
        def projectionArea(self, grid: List[List[int]]) -> int:
            m, n = len(grid), len(grid[0])
            cnt_z, cnt_x, cnt_y = 0, 0, 0
            for i in range(m):
                for j in range(n):
                    if grid[i][j]:
                        cnt_z += 1
                cnt_x += max(grid[i])
            for j in range(n):
                cur = 0
                for i in range(m):
                    if grid[i][j] > cur:
                        cur = grid[i][j]
                cnt_y += cur
            return cnt_x + cnt_y + cnt_z
        
    # 时间复杂度：O(mn)
    # 空间复杂度：O(mn)
    ```

    

32. 2022年4月27日——[太平洋大西洋水流问题](https://leetcode-cn.com/problems/pacific-atlantic-water-flow/)

    ```python
    class Solution:
        def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
            m, n = len(heights), len(heights[0])
    
            def search(starts: List[Tuple[int, int]]) -> Set[Tuple[int, int]]:
                visited = set()
                def dfs(x: int, y: int):
                    if (x, y) in visited:
                        return
                    visited.add((x, y))
                    for nx, ny in ((x, y + 1), (x, y - 1), (x - 1, y), (x + 1, y)):
                        if 0 <= nx < m and 0 <= ny < n and heights[nx][ny] >= heights[x][y]:
                            dfs(nx, ny)
                for x, y in starts:
                    dfs(x, y)
                return visited
    
            pacific = [(0, i) for i in range(n)] + [(i, 0) for i in range(1, m)]
            atlantic = [(m - 1, i) for i in range(n)] + [(i, n - 1) for i in range(m - 1)]
            return list(map(list, search(pacific) & search(atlantic)))
    
    # 深搜或广搜    
    # 时间复杂度：O(mn)
    # 空间复杂度：O(mn)
    ```

    

33. 2022年4月28日——[按奇偶排序数组](https://leetcode-cn.com/problems/sort-array-by-parity/)

    ```python
    class Solution:
        def sortArrayByParity(self, nums: List[int]) -> List[int]:
            i, j = 0, len(nums) - 1  # 双指针
            while i < j:
                p1 = nums[i] % 2
                p2 = nums[j] % 2
                if p1:   # 左边的为奇数
                    if p2 == 0:
                        nums[i], nums[j] = nums[j], nums[i]  # 右边的为偶数则交换位置
                        i += 1
                    j -= 1
                else:
                    if p2:
                        j -= 1
                    i += 1
            return nums
        
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    ```

    

34. 2022年4月29日——[建立四叉树](https://leetcode-cn.com/problems/construct-quad-tree/)

    ```python
    """
    # Definition for a QuadTree node.
    class Node:
        def __init__(self, val, isLeaf, topLeft, topRight, bottomLeft, bottomRight):
            self.val = val
            self.isLeaf = isLeaf
            self.topLeft = topLeft
            self.topRight = topRight
            self.bottomLeft = bottomLeft
            self.bottomRight = bottomRight
    """
    
    class Solution:
        def construct(self, grid: List[List[int]]) -> 'Node':
            def dfs(r0: int, c0: int, r1: int, c1: int) -> 'Node':
                if all(grid[i][j] == grid[r0][c0] for i in range(r0, r1) for j in range(c0, c1)):
                    return Node(grid[r0][c0] == 1, True)
                return Node(
                    True,
                    False,
                    dfs(r0, c0, (r0 + r1) // 2, (c0 + c1) // 2),
                    dfs(r0, (c0 + c1) // 2, (r0 + r1) // 2, c1),
                    dfs((r0 + r1) // 2, c0, r1, (c0 + c1) // 2),
                    dfs((r0 + r1) // 2, (c0 + c1) // 2, r1, c1),
                )
            return dfs(0, 0, len(grid), len(grid))
        
    # 直接递归
    # 时间复杂度：n^2log(n)
    # 空间复杂度：log(n)
    ```

    

35. 2022年4月30日——[最小差值 I](https://leetcode-cn.com/problems/smallest-range-i/)

    ```python
    class Solution:
        def smallestRangeI(self, nums: List[int], k: int) -> int:
            n = len(nums)
            small, large = min(nums) + k, max(nums) - k
            return 0 if small > large else large - small
        
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    ```

    


## 5月

36. 2022年5月1日——[两棵二叉搜索树中的所有元素](https://leetcode-cn.com/problems/all-elements-in-two-binary-search-trees/)

    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def getAllElements(self, root1: TreeNode, root2: TreeNode) -> List[int]:
            def get_lis(root):  # 把二叉搜索树转换为有序列表
                if not root:
                    return []
                return get_lis(root.left) + [root.val] + get_lis(root.right)
            lis1, lis2 = get_lis(root1), get_lis(root2)
            ans = []
            while lis1 and lis2:
                ans.append(lis1.pop(0) if lis1[0] < lis2[0] else lis2.pop(0))
            ans += lis1 + lis2
            return ans
        
    # 类似归并排序一样，转换成有序列表然后合并，当然没有优化所以效率低
    # 时间复杂度：O(n+m)
    # 空间复杂度：O(n+m)
    ```

    

37. 2022年5月2日——[标签验证器](https://leetcode-cn.com/problems/tag-validator/)

    ```python
    class Solution:
        def isValid(self, code: str) -> bool:
            tags = []
            i, n = 0, len(code)
            while i < n:
                if code[i] != "<":
                    if not tags:
                        return False
                    i += 1
                    continue
                if i == n - 1:
                    return False
                if code[i + 1] == "/":
                    j = code.find(">", i)
                    if j == -1:
                        return False
                    tagname = code[i + 2: j]
                    if not tags or tags[-1] != tagname:
                        return False
                    tags.pop()
                    i = j + 1
                    if not tags and i != n:
                        return False
                elif code[i + 1] == "!":
                    if not tags:
                        return False
                    cdata = code[i + 2: i + 9]
                    if cdata != "[CDATA[":
                        return False
                    j = code.find("]]>", i)
                    if j == -1:
                        return False
                    i = j + 1
                else:
                    j = code.find(">", i)
                    if j == -1:
                        return False
                    tagname = code[i + 1: j]
                    if not 1 <= len(tagname) <= 9 or not all(ch.isupper() for ch in tagname):
                        return False
                    tags.append(tagname)
                    i = j + 1
    
            return not tags
    # 困难题，没做
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

38. 2022年5月3日——[重新排列日志文件](https://leetcode-cn.com/problems/reorder-data-in-log-files/)

    ```python
    class Solution:
        def reorderLogFiles(self, logs: List[str]) -> List[str]:
            # 尝试用归并排序
            def merge(a1, a2):
                m = []
                while a1 and a2:
                    i1 = a1[0].find(' ')
                    i2 = a2[0].find(' ')
                    if a1[0][i1+1].isdigit():
                        if a2[0][i2+1].isdigit():
                            m.append(a1.pop(0))
                        else:
                            m.append(a2.pop(0))
                    else:
                        if a2[0][i2+1].isdigit():
                            m.append(a1.pop(0))
                        else:
                            if a1[0][i1+1:] == a2[0][i2+1:]:
                                m.append(a1.pop(0) if a1[0][:i1] < a2[0][:i2] else a2.pop(0))
                            else:
                                m.append(a1.pop(0) if a1[0][i1+1:] < a2[0][i2+1:] else a2.pop(0))
                m = m + a1 + a2
                return m
            def merge_sort(a):
                if len(a) == 1:
                    return a
                left = a[:len(a) // 2]
                right = a[len(a) // 2:]
                return merge(merge_sort(left), merge_sort(right))
            return merge_sort(logs)
        
    # 时间复杂度：O(nlogn)
    # 空间复杂度：O(logn)
    ```

    

39. 2022年5月4日——[找出游戏的获胜者](https://leetcode-cn.com/problems/find-the-winner-of-the-circular-game/)

    ```python
    class Solution:
        def findTheWinner(self, n: int, k: int) -> int:
            remains = list(range(1, n+1))   # 还剩下的编号
            cur = 0  # 当前计数的开始位置
            while remains:
                m = len(remains)
                nxt = (cur + k - 1) % m  # 下一个位置
                ans = remains.pop(nxt)
                cur = nxt if nxt < m else 0
            return ans
    
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    
    # 官方解答的更简洁
    class Solution:
        def findTheWinner(self, n: int, k: int) -> int:
            winner = 1
            for i in range(2, n + 1):
                winner = (k + winner - 1) % i + 1
            return winner
    
    ```

    

40. 2022年5月5日——[乘积小于 K 的子数组](https://leetcode-cn.com/problems/subarray-product-less-than-k/)

    ```python
    class Solution:
        def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
            # 类似贪心算法
            if k == 0:  # 为0直接返回
                return 0
            n = len(nums)
            pre, nxt = 0, 0   # 计数
            cur, cur_n = 1, 0   # 当前乘积
            for i, a in enumerate(nums):
                if a < k:  # 当前值小于k
                    nxt += 1
                    cur = cur * a
                    cur_n += 1
                    if cur < k:
                        nxt += pre + cur_n - 1  # i位置及以前的数目，状态转移方程
                    else:
                        while cur_n:
                            cur //= nums[i - cur_n + 1]
                            cur_n -= 1
                            if cur < k:
                                break
                        nxt += pre + cur_n - 1
                else:
                    cur_n = 0
                    cur = 1
                    nxt = pre
                pre, nxt = nxt, 0
            return pre
    
    # 时间复杂度：O(mn)  # m为最大连续的程度
    # 空间复杂度：O(1)
    
    
    class Solution:
        def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
            ans, prod, i = 0, 1, 0
            for j, num in enumerate(nums):
                prod *= num
                while i <= j and prod >= k:
                    prod //= nums[i]
                    i += 1
                ans += j - i + 1
            return ans
    
    # 滑动窗口
    # 时间复杂度：O(n)  # m为最大连续的程度
    # 空间复杂度：O(1)
    
    ```

    

41. 2022年5月6日——[最近的请求次数](https://leetcode-cn.com/problems/number-of-recent-calls/)

    ```python
    class RecentCounter:
    
        def __init__(self):
            self.left = 0  # 左端的时间节点
            self.lis = []  # 每次请求的时间列表
    
        def ping(self, t: int) -> int:
            self.lis.append(t)
            while (t - 3000) > self.lis[self.left]:  # 滑动窗口，保证left为在t3000毫秒内
                self.left += 1
            return len(self.lis) - self.left
    
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    
    # Your RecentCounter object will be instantiated and called as such:
    # obj = RecentCounter()
    # param_1 = obj.ping(t)
    
    
    class RecentCounter:
        def __init__(self):
            self.q = deque()
    
        def ping(self, t: int) -> int:
            self.q.append(t)
            while self.q[0] < t - 3000:
                self.q.popleft()
            return len(self.q)
    # 官方的更简洁的写法
    # 时间复杂度：均摊 O(1)，每个元素至多入队出队各一次。
    # 空间复杂度：O(L)，其中 L 为队列的最大元素个数。
    ```

    

42. 2022年5月7日——[最小基因变化](https://leetcode-cn.com/problems/minimum-genetic-mutation/)

    ```python
    class Solution:
        def minMutation(self, start: str, end: str, bank: List[str]) -> int:
            # 整体想的是bfs，从末尾往前判断一次操作，然后找最短的次数
            if end not in bank:
                return -1
            if start not in bank:
                bank.append(start)
            def bfs(st, ed, bk, step=0):
                nbk = bk[:]
                nbk.remove(ed)
                available = []
                for x in nbk:
                    if sum([y1 != y2 for y1, y2 in zip(x, ed)]) == 1:
                        if x == st:
                            return step + 1
                        else:
                            available.append(x)
                if not available:
                    return -1
                ans = -1
                for a in available:
                    res = bfs(st, a, nbk, step + 1)
                    if res != -1:
                        ans = min(ans, res) if ans != -1 else res
                return ans
            return bfs(start, end, bank)
    # 时间复杂度：O(n^2)
    # 空间复杂度：O(n)  # n为bank的长度
    ```

    

43. 2022年5月8日——[数组中重复的数据](https://leetcode.cn/problems/find-all-duplicates-in-an-array/)

    ```python
    class Solution:
        def findDuplicates(self, nums: List[int]) -> List[int]:
            ans = []
            i = 0
            se = set()
            while i < len(nums):
                se.add(nums[i])
                if len(se) == (i + 1):
                    i += 1
                else:
                    ans.append(nums.pop(i))
            return ans
    # 不符合题目要求，效率低
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    
    class Solution:
        def findDuplicates(self, nums: List[int]) -> List[int]:
            for i in range(len(nums)):
                while nums[i] != nums[nums[i] - 1]:
                    nums[nums[i] - 1], nums[i] = nums[i], nums[nums[i] - 1]
            return [num for i, num in enumerate(nums) if num - 1 != i]
    
    
    # 很巧妙的一种方法
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    ```

    

44. 2022年5月9日——[增减字符串匹配](https://leetcode.cn/problems/di-string-match/)

    ```python
    class Solution:
        def diStringMatch(self, s: str) -> List[int]:
            n = len(s)
            lis = []
            available = list(range(n + 1))
            for i in range(n):
                if s[i] == 'I':
                    lis.append(available.pop(0))
                else:
                    lis.append(available.pop(-1))
            return lis + available
    # 每次从可取值中弹出一大一小的值
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    
    
    class Solution:
        def diStringMatch(self, s: str) -> List[int]:
            lo = 0
            hi = n = len(s)
            perm = [0] * (n + 1)
            for i, ch in enumerate(s):
                if ch == 'I':
                    perm[i] = lo
                    lo += 1
                else:
                    perm[i] = hi
                    hi -= 1
            perm[n] = lo  # 最后剩下一个数，此时 lo == hi
            return perm
    
    # 双指针解法
    # 时间复杂度：O(n)
    # 空间复杂度：O(1)
    ```

    

45. 2022年5月10日——[猫和老鼠 II](https://leetcode.cn/problems/cat-and-mouse-ii/)

    ```python
    # 挺难的，题解也看的不明白
    ```

    

46. 2022年5月11日——[序列化和反序列化二叉搜索树](https://leetcode.cn/problems/serialize-and-deserialize-bst/)

    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    
    class Codec:
    
        def serialize(self, root: TreeNode) -> str:
            """Encodes a tree to a single string.
            """
            # 把树用层序遍历表示，空节点也填充进去，保证为满二叉树
            if root is None:
                return '[]'
            q = [root]
            lis = []
            while q:
                n = len(q)
                for _ in range(n):
                    nd = q.pop(0)
                    if nd:
                        lis.append(nd.val)
                        # if nd.left or nd.right:
                        q.append(nd.left)
                        q.append(nd.right)
                    else:
                        lis.append(None)
            return str(lis)
    
        def deserialize(self, data: str) -> TreeNode:
            """Decodes your encoded data to tree.
            """
            lis = eval(data)
            if not lis:
                return None
            i, n = 0, len(lis)
            root = TreeNode(lis[i])
            q = [root]
            while (2 * i + 2) < n:
                t = q.pop(0)
                if lis[2 * i + 1] is not None:
                    t.left = TreeNode(lis[2 * i + 1])
                    q.append(t.left)
                if lis[2 * i + 2] is not None:
                    t.right = TreeNode(lis[2 * i + 2])
                    q.append(t.right)
                i += 1
            return root
            
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    # Your Codec object will be instantiated and called as such:
    # Your Codec object will be instantiated and called as such:
    # ser = Codec()
    # deser = Codec()
    # tree = ser.serialize(root)
    # ans = deser.deserialize(tree)
    # return ans
    ```

    

47. 2022年5月12日——[删列造序](https://leetcode.cn/problems/delete-columns-to-make-sorted/)

    ```python
    class Solution:
        def minDeletionSize(self, strs: List[str]) -> int:
            m, n = len(strs), len(strs[0])
            cnt = 0
            for j in range(n):
                cur = strs[0][j]
                for i in range(1, m):
                    if strs[i][j] < cur:
                        cnt += 1
                        break
                    cur = strs[i][j]
            return cnt
    
    # 时间复杂度：O(mn)
    # 空间复杂度：O(1)
    ```

    

47. 2022年5月13日——[一次编辑](https://leetcode.cn/problems/one-away-lcci/)

    ```python
    class Solution:
        def oneEditAway(self, first: str, second: str) -> bool:
            m, n = len(first), len(second)
            if m < n:
                return self.oneEditAway(second, first)
            if m - n > 1:
                return False
            for i, (x, y) in enumerate(zip(first, second)):
                if x != y:
                    return first[i + 1:] == second[i + 1:] if m == n else first[i + 1:] == second[i:]  # 注：改用下标枚举可达到 O(1) 空间复杂度
            return True
    
    # 时间复杂度：O(m+n)
    # 空间复杂度：O(1)
    ```

    

49. 2022年5月14日——[贴纸拼词](https://leetcode.cn/problems/stickers-to-spell-word/)

    ```python
    class Solution:
        def minStickers(self, stickers: List[str], target: str) -> int:
            def trans(s):
                cnts = Counter()
                for c in s:
                    if c in target:
                        cnts[c] += 1
                return cnts
    
            availables = [c for st in stickers if (c:=trans(st))]
            queue = deque([(target, 0)])
            explored = {target}
            while queue:
                cur, step = queue.popleft()
                if not cur:
                    return step
                for avl in availables:
                    if cur[0] in avl:
                        nxt = cur
                        for k, v in avl.items():
                            nxt = nxt.replace(k, '', v)
                        if nxt not in explored:
                            explored.add(nxt)
                            queue.append((nxt, step + 1))
            return -1
    # 深搜或者广搜，但是写代码还得注意些，自己没写出来，抄代码
    ```

    

50. 2022年5月15日——[最大三角形面积](https://leetcode.cn/problems/largest-triangle-area/)

    ```python
    class Solution:
        def largestTriangleArea(self, points: List[List[int]]) -> float:
            # 直接暴力？
            area = 0
            n = len(points)
            for i in range(n - 2):
                for j in range(i + 1, n - 1):
                    for k in range(j + 1, n):
                        s = abs(points[i][0] * points[j][1]- points[i][0] * points[k][1] + points[j][0] * points[k][1] - points[j][0] * points[i][1] + points[k][0] * points[i][1] - points[k][0] * points[j][1]) / 2
                        if s > area:
                            area = s
            return area
    # 时间复杂度：O(n^3)
    # 空间复杂度：O(1)
    
    
    class Solution:
        def largestTriangleArea(self, points: List[List[int]]) -> float:
            def triangleArea(x1: int, y1: int, x2: int, y2: int, x3: int, y3: int) -> float:
                return abs(x1 * y2 + x2 * y3 + x3 * y1 - x1 * y3 - x2 * y1 - x3 * y2) / 2
            return max(triangleArea(x1, y1, x2, y2, x3, y3) for (x1, y1), (x2, y2), (x3, y3) in combinations(points, 3))
    # 时间复杂度：O(n^3)
    # 空间复杂度：O(1)
    ```

    

51. 2022年5月16日——[后继者](https://leetcode.cn/problems/successor-lcci/)

    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, x):
    #         self.val = x
    #         self.left = None
    #         self.right = None
    
    class Solution:
        def inorderSuccessor(self, root: TreeNode, p: TreeNode) -> TreeNode:
            st, pre, cur = [], None, root
            while st or cur:
                while cur:
                    st.append(cur)
                    cur = cur.left
                cur = st.pop()
                if pre == p:
                    return cur
                pre = cur
                cur = cur.right
            return None
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

52. 2022年5月17日——[验证外星语词典](https://leetcode.cn/problems/verifying-an-alien-dictionary/)

    ```python
    class Solution:
        def isAlienSorted(self, words: List[str], order: str) -> bool:
            n = len(words)
            if n == 1:
                return True
            mapped = {c: i for i, c in enumerate(order)}
            for i in range(n-1):
                for w1, w2 in zip(words[i], words[i+1]):
                    if mapped[w2] < mapped[w1]:
                        return False
                    elif mapped[w2] > mapped[w1]:
                        break
                else:
                    if len(words[i]) > len(words[i+1]):
                        return False
            return True
    
    # 时间复杂度：O(n)  # 所有字母的长度
    # 空间复杂度：O(1)
    ```

    
    ```c++
    class Solution {
    public:
        bool isAlienSorted(vector<string>& words, string order) {
            map<char, int> mapped;
            for (int i=0; i < order.length(); i++){
                mapped[order[i]] = i;
            }
    
            for (int i=0; i < words.size() - 1; i++){
                int m = words[i].length(), n = words[i+1].length();
                for (int j=0; j < min(m, n); j++){
                    if (mapped[words[i][j]] > mapped[words[i+1][j]]){
                        return false;
                    }
                    else if (mapped[words[i][j]] < mapped[words[i+1][j]]){
                        break;
                    }
                }
                if ((m > n) && (words[i].substr(0, n) == words[i+1].substr(0, n))) {
                    return false;
                }
            }
            return true;
        }
    };
    ```
    
    

53. 2022年5月18日——[乘法表中第k小的数](https://leetcode.cn/problems/kth-smallest-number-in-multiplication-table/)

    ```python
    class Solution:
        def findKthNumber(self, m: int, n: int, k: int) -> int:
            a = []
            for i in range(1, m+1):
                for j in range(1, n+1):
                    heapq.heappush(a, i * j)
            return heapq.nsmallest(k, a)[-1]
        # 想通过堆来做，但也会超时
        
    
    class Solution:
        def findKthNumber(self, m: int, n: int, k: int) -> int:
            left, right = 1, m * n
            while left < right:
                x = left + (right - left) // 2
                cnt = x // n * n
                for i in range(x // n + 1, m + 1):
                    cnt += x // i
                if cnt >= k:
                    right = x
                else:
                    left = x + 1
            return left
    
    # 使用二分查找的思想，很妙
    # 时间复杂度：O(mlog(mn))
    # 空间复杂度：O(1)
    ```

    ```c++
    class Solution {
    public:
        int findKthNumber(int m, int n, int k) {
            int left = 1, right = m * n;
            while (left < right) {
                int x = left + (right - left) / 2;
                int count = x / n * n;
                for (int i = x / n + 1; i <= m; ++i) {
                    count += x / i;
                }
                if (count >= k) {
                    right = x;
                } else {
                    left = x + 1;
                }
            }
            return left;
        }
    };
    // 使用二分查找的思想，很妙
    // 时间复杂度：O(mlog(mn))
    // 空间复杂度：O(1)
    ```

    

54. 2022年5月19日——[最少移动次数使数组元素相等 II](https://leetcode.cn/problems/minimum-moves-to-equal-array-elements-ii/)

    ```python
    class Solution:
        def minMoves2(self, nums: List[int]) -> int:
            nums.sort()
            return sum(abs(num - nums[len(nums) // 2]) for num in nums)
    # 排序，选中位数，求和
    # 时间复杂度：O(nlog(n))
    # 空间复杂度：O(1)
    ```

    ```c++
    class Solution {
    public:
        int minMoves2(vector<int>& nums) {
            sort(nums.begin(), nums.end());
            int n = nums.size(), ret = 0, x = nums[n / 2];
            for (int i = 0; i < n; i++) {
                ret += abs(nums[i] - x);
            }
            return ret;
        }
    };
    
    # 时间复杂度：O(nlog(n))
    # 空间复杂度：O(logn)
    ```

    

55. 2022年5月20日——[寻找右区间](https://leetcode.cn/problems/find-right-interval/)

    ```python
    ```

    ```c++
    // 创建一个类记录序号
    class interval_cls {
    public:
        vector<int> inter;
        int id;
        interval_cls() {};
        interval_cls(vector<int> vect_inter, int ids):inter(vect_inter),id(ids) {};
    };
    
    // c++中为sort函数提供比较方法
    bool interval_cmp(interval_cls c1, interval_cls c2){
        return c1.inter[0] < c2.inter[0];
    }
    
    // 二分查找
    int binary_search(vector<interval_cls> v_cls, int endi){
        int n = v_cls.size(), left = 0, right = n - 1;
        int mid;
        while (left < right){
            mid = (left + right) / 2;
            if (v_cls[mid].inter[0] < endi){
                left = mid + 1;
            }
            else if (v_cls[mid].inter[0] == endi){
                return v_cls[mid].id;
            }
            else {
                right = mid;
            }
        }
        if (v_cls[right].inter[0] < endi){
            return -1;
        } else {
            return v_cls[right].id;
        }
    }
    
    class Solution {
    public:
        vector<int> findRightInterval(vector<vector<int>>& intervals) {
            vector<interval_cls> vect_cls;
            int n = intervals.size();
            for (int i = 0; i < n; ++i){
                interval_cls tmp(intervals[i], i);
                vect_cls.push_back(tmp);
            }
            sort(vect_cls.begin(), vect_cls.end(), interval_cmp);
            vector<int> ans;
            for (int i = 0; i < n; ++i){
                ans.push_back(binary_search(vect_cls, intervals[i][1]));
            }
            return ans;
        }
    };
    // 会超时，但是思想就是二分查找，这里应该是实现的方式不对
    ```

    

56. 2022年5月21日——[在长度 2N 的数组中找出重复 N 次的元素](https://leetcode.cn/problems/n-repeated-element-in-size-2n-array/)

    ```python
    class Solution:
        def repeatedNTimes(self, nums: List[int]) -> int:
            cnt = Counter(nums)
            return [x for x in cnt if cnt[x] > 1][0]
    ```

    ```c++
    class Solution {
    public:
        int repeatedNTimes(vector<int>& nums) {
            // 按照题目意思，只需要找有重复的元素，可以使用map
            map<int, int> mapped;
            for (int i = 0; i < nums.size(); ++i){
                auto it = mapped.find(nums[i]);
                if (it == mapped.end()){
                    mapped[nums[i]] = 1;
                } else {
                    return nums[i];
                }
            }
            return 0;
        }
    };
    // 时间复杂度：O(n)
    // 空间复杂度：O(n)
    ```

    

57. 2022年5月22日——[我能赢吗](https://leetcode.cn/problems/can-i-win/)

    ```python
    class Solution:
        def canIWin(self, maxChoosableInteger: int, desiredTotal: int) -> bool:
            @cache
            def dfs(usedNumbers: int, currentTotal: int) -> bool:
                for i in range(maxChoosableInteger):
                    if (usedNumbers >> i) & 1 == 0:
                        if currentTotal + i + 1 >= desiredTotal or not dfs(usedNumbers | (1 << i), currentTotal + i + 1):
                            return True
                return False
    
            return (1 + maxChoosableInteger) * maxChoosableInteger // 2 >= desiredTotal and dfs(0, 0)
    
    s
    # 博弈论DP？不会做
    
    ```

    

58. 2022年5月23日——[为高尔夫比赛砍树](https://leetcode.cn/problems/cut-off-trees-for-golf-event/)

    ```python
    class Solution:
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        def dfs(self, pos, d2, area):
            new_area = deepcopy(area)
            new_area[pos[0]][pos[1]] = 0
            m, n = len(new_area), len(new_area[0])
            q = deque([[pos[0], pos[1], 0]])
            while q:
                k = len(q)
                for _ in range(k):
                    x, y, dis = q.popleft()
                    for dr in self.dirs:
                        nx, ny = x + dr[0], y + dr[1]
                        if (0 <= nx < m) and (0 <= ny < n) and (new_area[nx][ny] != 0):
                            if new_area[nx][ny] == d2:
                                return dis + 1
                            q.append([nx, ny, dis + 1])
                            new_area[nx][ny] = 0
            return -1
            
        def cutOffTree(self, forest: List[List[int]]) -> int:
            # 用广度优先搜索
            trees = sorted([[c, i, j] for i, r in enumerate(forest) for j, c in enumerate(r) if c not in [0, 1]])
            step = 0
            pos = [0, 0]
            for t, i, j in trees:
                if t == forest[pos[0]][pos[1]]:
                    dis = 0
                else:
                    dis = self.dfs(pos, t, forest)
                if dis == -1:
                    return -1
                step += dis
                pos = [i, j]
            return step
    ```

    ```c++
    class Solution {
    public:
        int dfs(int *pos, int d, vector<vector<int>> forest){
            int k, m = forest.size(), n = forest[0].size();
            int dirs[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
            queue<vector<int>> q;
            vector<int> tmp = {pos[0], pos[1], 0};
            q.push(tmp);
            forest[pos[0]][pos[1]] = 0;
            while (!q.empty()){
                k = q.size();
                for (int i = 0; i < k; ++i){
                    tmp = q.front();
                    q.pop();
                    for (int j = 0; j < 4; ++j){
                        int nx = tmp[0] + dirs[j][0], ny = tmp[1] + dirs[j][1];
                        if ((nx >= 0) && (nx < m) && (ny >= 0) && (ny < n) && (forest[nx][ny] != 0)){
                            if (forest[nx][ny] == d){
                                return tmp[2] + 1;
                            }
                            vector<int> tem = {nx, ny, tmp[2] + 1};
                            q.push(tem);
                            forest[tmp[0]][tmp[1]] = 0;
                        }
                    }
                }
            }
            return -1;
        }
    
        static bool cmp(vector<int> a, vector<int> b){
            return a[0] < b[0];
        }
    
        int cutOffTree(vector<vector<int>>& forest) {
            vector<vector<int>> trees;
            for (int i = 0; i < forest.size(); ++i){
                for (int j = 0; j < forest[i].size(); ++j){
                    if ((forest[i][j] != 0) && (forest[i][j] != 1)){
                        vector<int> tmp={forest[i][j], i, j};
                        trees.push_back(tmp);
                    }
                }
            }
            sort(trees.begin(), trees.end(), cmp);
            int dist, step = 0;
            int pos[2] = {0, 0};
            for (int i = 0; i < trees.size(); ++i){
                int d = trees[i][0];
                if (forest[pos[0]][pos[1]] == d){
                    dist = 0;
                } else {
                    dist = dfs(pos, d, forest);
                }
                if (dist == -1){
                    return -1;
                }
                step += dist;
                pos[0] = trees[i][1];
                pos[1] = trees[i][2];
            }
            return step;
        }
    };
    ```

    

59. 2022年5月24日——[单值二叉树](https://leetcode.cn/problems/univalued-binary-tree/)

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        bool isUnivalTree(TreeNode* root) {
            // 深搜
            int v = root->val;
            stack<TreeNode *> stk;
            TreeNode* cur = root;
            while ((!stk.empty()) || (cur)){
                while (cur){
                    if (cur->val != v){
                        return false;
                    }
                    stk.push(cur);
                    cur = cur->left;
                }
                cur = stk.top();
                stk.pop();
                cur = cur->right;
            }
            return true;
        }
    };
    
    class Solution {
    public:
        bool isUnivalTree(TreeNode* root) {
            // 广搜
            int n, v = root->val;
            queue<TreeNode *> q;
            TreeNode* cur;
            q.push(root);
            while (!q.empty()){
                n = q.size();
                for (int i = 0; i < n; ++i){
                    cur = q.front();
                    q.pop();
                    if (cur->val != v){
                        return false;
                    }
                    if (cur->left){
                        q.push(cur->left);
                    }
                    if (cur->right){
                        q.push(cur->right);
                    }
                }
            }
            return true;
        }
    };
    
    class Solution {
    public:
        bool isUnivalTree(TreeNode* root) {
            // 递归
            if (!root){
                return true;
            }
            int v = root->val;
            if (((root->left) && (root->left->val != v)) || ((root->right) && (root->right->val != v))){
                return false;
            }
            return isUnivalTree(root->left) && isUnivalTree(root->right);
        }
    };
    // 时间复杂度：O(n)
    // 空间复杂度：O(n)
    ```

    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def isUnivalTree(self, root: TreeNode) -> bool:
            if not root:
                return True
            v = root.val
            if (root.left and v != root.left.val) or (root.right and v != root.right.val):
                return False
            return self.isUnivalTree(root.left) and self.isUnivalTree(root.right)
    
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    ```

    

60. 2022年5月25日——[环绕字符串中唯一的子字符串](https://leetcode.cn/problems/unique-substrings-in-wraparound-string/)

    ```python
    class Solution:
        def findSubstringInWraproundString(self, p: str) -> int:
            dp = defaultdict(int)
            k = 0
            for i, ch in enumerate(p):
                if i > 0 and (ord(ch) - ord(p[i - 1])) % 26 == 1:  # 字符之差为 1 或 -25
                    k += 1
                else:
                    k = 1
                dp[ch] = max(dp[ch], k)
            return sum(dp.values())
    # 用动态规划做
    ```

    ```c++
    class Solution {
    public:
        int findSubstringInWraproundString(string p) {
            vector<int> dp(26);
            int k = 0;
            for (int i = 0; i < p.length(); ++i) {
                if (i && (p[i] - p[i - 1] + 26) % 26 == 1) { // 字符之差为 1 或 -25
                    ++k;
                } else {
                    k = 1;
                }
                dp[p[i] - 'a'] = max(dp[p[i] - 'a'], k);
            }
            return accumulate(dp.begin(), dp.end(), 0);
        }
    };
    
    // 时间复杂度：O(n)
    // 空间复杂度：O(1)
    ```

    

61. 2022年5月26日——[掉落的方块](https://leetcode.cn/problems/falling-squares/)

    ```python
    class Solution:
        def fallingSquares(self, positions: List[List[int]]) -> List[int]:
            n = len(positions)
            heights = [0] * n
            for i, (left1, side1) in enumerate(positions):
                right1 = left1 + side1 - 1
                heights[i] = side1
                for j in range(i):
                    left2, right2 = positions[j][0], positions[j][0] + positions[j][1] - 1
                    if right1 >= left2 and right2 >= left1:
                        heights[i] = max(heights[i], heights[j] + side1)
            for i in range(1, n):
                heights[i] = max(heights[i], heights[i - 1])
            return heights
    
    ```

    

62. 2022年5月27日——[单词距离](https://leetcode.cn/problems/find-closest-lcci/)

    ```python
    class Solution:
        def findClosest(self, words: List[str], word1: str, word2: str) -> int:
            i1, i2 = -1, -1
            ans = len(words)
            for i, w in enumerate(words):
                if w == word1:
                    i1 = i
                elif w == word2:
                    i2 = i
                if i1 >= 0 and i2 >= 0:
                    ans = min(ans, abs(i1 - i2))
            return ans
    # 直接暴力
    
    ```

    ```c++
    class Solution {
    public:
        int findClosest(vector<string>& words, string word1, string word2) {
            vector<int> v1, v2, v;
            for (int i = 0; i < words.size(); ++i){
                if (words[i] == word1){
                    v1.push_back(i);
                }
                else if (words[i] == word2){
                    v2.push_back(i);
                }
            }
            for (auto it1: v1){
                for (auto it2: v2){
                    v.push_back(abs(it1 - it2));
                }
            }
            return *min_element(v.begin(), v.end());
        }
    };
    ```

    

63. 2022年5月28日——[删除最外层的括号](https://leetcode.cn/problems/remove-outermost-parentheses/)

    ```python
    class Solution:
        def removeOuterParentheses(self, s: str) -> str:
            ans = ""
            primitive = []
            n = 0
            for i, c in enumerate(s):
                if not primitive and c == '(':  # 原语为空
                    primitive.append(c)
                    n += 1
                else:  # 原语非空
                    if c == '(':
                        primitive.append(c)
                        n += 1
                    else:
                        primitive.pop()
                        n -= 1
                if n > 1 or (n >= 1 and c == ')'):
                    ans += c
            return ans
    
    # 时间复杂度：O(n)
    # 空间复杂度：O(n)
    
    ```

    ```c++
    class Solution {
    public:
        string removeOuterParentheses(string s) {
            string ans;
            int n = 0;
            stack<char> stk;
            for (int i = 0; i < s.length(); ++i){
                if (stk.empty()) {
                    stk.push(s[i]);
                    n += 1;
                }
                else {
                    if (s[i] == '(') {
                        stk.push('(');
                        n += 1;
                    }
                    else {
                        stk.pop();
                        n -= 1;
                    }
                }
                if ((n > 1) || (n >= 1 && s[i] == ')')) {
                    ans += s[i];
                }
            }
            return ans;
        }
    };
    ```

    

64. 2022年5月29日——[验证IP地址](https://leetcode.cn/problems/validate-ip-address/)

    ```python
    class Solution:
        def validIPAddress(self, queryIP: str) -> str:
            st = queryIP.split('.')
            if len(st) == 4:
                for c in st:
                    if c.isdigit() and str(int(c)) == c and 0 <= int(c) <= 255:
                        continue
                    return 'Neither'
                return 'IPv4'
            else:
                st = queryIP.split(':')
                if len(st) == 8:
                    for c in st:
                        t = all(map(lambda y: 'a' <= y <= 'f' or 'A' <= y <= 'F', filter(lambda x: x.isalpha(), c)))
                        if 1 <= len(c) <= 4 and t:
                            continue
                        return 'Neither'
                    return 'IPv6'
                return 'Neither'
                
    ```

    ```c++
    class Solution {
    public:
        vector<string> split_s(string s, char delim){
            int n = s.size(), j = 0;
            vector<string> vs;
            for (int i = 0; i < n; ++i){
                if (s[i] == delim){
                    vs.push_back(s.substr(j, i - j));
                    j = i + 1;
                }
            }
            vs.push_back(s.substr(j));
            return vs;
        }
        bool is_num(string s){
            if (s.empty()){
                return false;
            }
            for (auto c: s){
                if (c >= '0' && c <= '9'){
                    continue;
                }
                return false;
            }
            return true;
        }
        bool to_int(string s){
            int ans = 0;
            for(int i = 0; i < s.size(); ++i){
                ans = ans * 10 + s[i] - '0';
                if (ans > 255){
                    return false;
                }
            }
            return true;
        }
        bool is_pre0(string s){
            int n = s.size();
            if (n <= 1){
                return false;
            }
            return s[0] == '0';
        }
        bool is_hex(string s){
            for (auto c: s){
                if (c >= '0' && c <= '9'){
                    continue;
                }
                if ((c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')){
                    continue;
                } else {
                    return false;
                }
            }
            return true;
        }
        string validIPAddress(string queryIP) {
            vector<string> vs = split_s(queryIP, '.');
            if (vs.size() == 4){
                for (auto ss: vs){
                    if (is_num(ss) && !is_pre0(ss) && to_int(ss)){
                        continue;
                    }
                    return "Neither";
                }
                return "IPv4";
            }
            else {
                vector<string> vs = this->split_s(queryIP, ':');
                if (vs.size() == 8){
                    for (auto ss: vs){
                        if ((ss.size() >= 1 && ss.size() <= 4) && is_hex(ss)){
                            continue;
                        }
                        return "Neither";
                    }
                    return "IPv6";
                }
                return "Neither";
            }
        }
    };
    ```

    

65. 2022年5月30日——[从根到叶的二进制数之和](https://leetcode.cn/problems/sum-of-root-to-leaf-binary-numbers/)

    ```python
    # Definition for a binary tree node.
    # class TreeNode:
    #     def __init__(self, val=0, left=None, right=None):
    #         self.val = val
    #         self.left = left
    #         self.right = right
    class Solution:
        def sumRootToLeaf(self, root: Optional[TreeNode]) -> int:
            ans = 0
            def dfs(nd, st=''):
                if not nd:
                    return 
                s = st + str(nd.val)
                if not (nd.left or nd.right):
                    nonlocal ans
                    ans += int(s, 2)
                    return 
                dfs(nd.left, s)
                dfs(nd.right, s)
            dfs(root)
            return ans
    d
    # 时间复杂度：O(n)
    # 空间复杂度：o(n)
    ```

    ```c++
    /**
     * Definition for a binary tree node.
     * struct TreeNode {
     *     int val;
     *     TreeNode *left;
     *     TreeNode *right;
     *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
     * };
     */
    class Solution {
    public:
        int ans = 0;
        int bin2int(string ss){
            int res = 0, n = ss.length(), b = 1;
            for (int i = n - 1; i >= 0; --i){
                res += (ss[i] - '0') * b;
                b *= 2;
            }
            return res;
        }
        void dfs(TreeNode* nd, string st){
            if (nd){
                string s = st + to_string(nd->val);
                if (!(nd->left || nd->right)){
                    ans += bin2int(s);
                }
                dfs(nd->left, s);
                dfs(nd->right, s);
            }
        }
        int sumRootToLeaf(TreeNode* root) {
            dfs(root, "");
            return ans;
        }
    };
    ```

    







## 6月

























