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

    

















