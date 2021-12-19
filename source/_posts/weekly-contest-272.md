---
title: LeetCode 周赛 272
date: 2021-12-19 13:56:42
tags: LeetCode 周赛
---

----------

# LeetCode 周赛 272

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-272/problems/find-first-palindromic-string-in-the-array/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-272/problems/adding-spaces-to-a-string/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-272/problems/number-of-smooth-descent-periods-of-a-stock/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-272/problems/minimum-operations-to-make-the-array-k-increasing/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1999 / 4697 | MiracleSNeko | 12   | 0:45:11  | 0:03:31                                                      | 0:06:26                                                      | 0:40:11 1                                                    |                                                              |

>   懂了，以后做不出四道题就只能狗比

## T1 5956. 找出数组中的第一个回文字符串

-   **User Accepted:**3691
-   **User Tried:**3730
-   **Total Accepted:**3742
-   **Total Submissions:**4426
-   **Difficulty:** **Easy**

给你一个字符串数组 `words` ，找出并返回数组中的 **第一个回文字符串** 。如果不存在满足要求的字符串，返回一个 **空字符串** `""` 。

**回文字符串** 的定义为：如果一个字符串正着读和反着读一样，那么该字符串就是一个 **回文字符串** 。

**示例 1：**

```
输入：words = ["abc","car","ada","racecar","cool"]
输出："ada"
解释：第一个回文字符串是 "ada" 。
注意，"racecar" 也是回文字符串，但它不是第一个。
```

**示例 2：**

```
输入：words = ["notapalindrome","racecar"]
输出："racecar"
解释：第一个也是唯一一个回文字符串是 "racecar" 。
```

**示例 3：**

```
输入：words = ["def","ghi"]
输出：""
解释：不存在回文字符串，所以返回一个空字符串。
```

**提示：**

-   `1 <= words.length <= 100`
-   `1 <= words[i].length <= 100`
-   `words[i]` 仅由小写英文字母组成

**提交：**

```rust
impl Solution {
    pub fn first_palindrome(words: Vec<String>) -> String {
        words.into_iter().find(|word| { let rev = String::from_utf8(word.bytes().rev().collect::<Vec<_>>()).unwrap(); rev == *word})
        .unwrap_or("".to_string())
    }
}
```

>   从下次开始这种题目就不再记录了

## T2 5957. 向字符串添加空格

-   **User Accepted:**3386
-   **User Tried:**3637
-   **Total Accepted:**3432
-   **Total Submissions:**5816
-   **Difficulty:** **Medium**

给你一个下标从 **0** 开始的字符串 `s` ，以及一个下标从 **0** 开始的整数数组 `spaces` 。

数组 `spaces` 描述原字符串中需要添加空格的下标。每个空格都应该插入到给定索引处的字符值 **之前** 。

-   例如，`s = "EnjoyYourCoffee"` 且 `spaces = [5, 9]` ，那么我们需要在 `'Y'` 和 `'C'` 之前添加空格，这两个字符分别位于下标 `5` 和下标 `9` 。因此，最终得到 `"Enjoy ***Y***our ***C***offee"` 。

请你添加空格，并返回修改后的字符串*。* 

**示例 1：**

```
输入：s = "LeetcodeHelpsMeLearn", spaces = [8,13,15]
输出："Leetcode Helps Me Learn"
解释：
下标 8、13 和 15 对应 "LeetcodeHelpsMeLearn" 中加粗斜体字符。
接着在这些字符前添加空格。
```

**示例 2：**

```
输入：s = "icodeinpython", spaces = [1,5,7,9]
输出："i code in py thon"
解释：
下标 1、5、7 和 9 对应 "icodeinpython" 中加粗斜体字符。
接着在这些字符前添加空格。
```

**示例 3：**

```
输入：s = "spacing", spaces = [0,1,2,3,4,5,6]
输出：" s p a c i n g"
解释：
字符串的第一个字符前可以添加空格。
```

**提示：**

-   `1 <= s.length <= 3 * 105`
-   `s` 仅由大小写英文字母组成
-   `1 <= spaces.length <= 3 * 105`
-   `0 <= spaces[i] <= s.length - 1`
-   `spaces` 中的所有值 **严格递增**

**提交：**

```rust
impl Solution {
    pub fn add_spaces(mut s: String, mut spaces: Vec<i32>) -> String {
        let mut cnt = 0;
        spaces.iter_mut().for_each(|i| { *i += cnt; cnt += 1;});
        spaces.iter().for_each(|&i| {
            s.insert(i as usize, ' ');
        });
        s
    }
}
```



## T3 5958. 股票平滑下跌阶段的数目

-   **User Accepted:**3000
-   **User Tried:**3365
-   **Total Accepted:**3050
-   **Total Submissions:**6874
-   **Difficulty:** **Medium**

给你一个整数数组 `prices` ，表示一支股票的历史每日股价，其中 `prices[i]` 是这支股票第 `i` 天的价格。

一个 **平滑下降的阶段** 定义为：对于 **连续一天或者多天** ，每日股价都比 **前一日股价恰好少** `1` ，这个阶段第一天的股价没有限制。

请你返回 **平滑下降阶段** 的数目。

**示例 1：**

```
输入：prices = [3,2,1,4]
输出：7
解释：总共有 7 个平滑下降阶段：
[3], [2], [1], [4], [3,2], [2,1] 和 [3,2,1]
注意，仅一天按照定义也是平滑下降阶段。
```

**示例 2：**

```
输入：prices = [8,6,7,7]
输出：4
解释：总共有 4 个连续平滑下降阶段：[8], [6], [7] 和 [7]
由于 8 - 6 ≠ 1 ，所以 [8,6] 不是平滑下降阶段。
```

**示例 3：**

```
输入：prices = [1]
输出：1
解释：总共有 1 个平滑下降阶段：[1]
```

**提示：**

-   `1 <= prices.length <= 105`
-   `1 <= prices[i] <= 105`

**提交：**

>   更简单的解法是用动态规划，维护一个 acc 记录由以前一天为结尾的平滑下跌阶段的数目

```c++
class Solution {
public:
    long long getDescentPeriods(vector<int>& prices) {
        long long ans = 0;
        int acc = 1, last = 0;
        for (int price : prices) {
            if (price == last - 1)
                ans += ++acc;
            else
                ans += acc = 1;
            last = price;
        }
        
        return ans;
    }
};

作者：吴自华
链接：https://leetcode-cn.com/circle/discuss/s1k590/view/l1QubD/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

>   滑动窗口，求每个下降区间并累加

```rust
pub fn get_descent_periods(mut prices: Vec<i32>) -> i64 {
        if prices.len() == 1 {
            return 1;
        };
        let mut length = vec![];
        let len = prices.len();
        let (mut l, mut r) = (0, 0);
        while r < len {
            while r < len && (r == len - 1 || prices[r] - prices[r + 1] == 1) {
                r += 1;
            }
            if r != l {
                length.push(r - l + if r == len { 0 } else { 1 })
            }
            l = r + 1;
            r += 1;
        }
        let mut ans = prices.len() as i64;
        for l in length.into_iter().filter(|&i| i != 1) {
            ans += ((l as i128) * (l as i128 + 1) / 2 - l as i128) as i64;
        }
        ans as i64
    }
}
```



## T4 5959. 使数组 K 递增的最少操作次数

-   **User Accepted:**889
-   **User Tried:**2284
-   **Total Accepted:**972
-   **Total Submissions:**4965
-   **Difficulty:** **Hard**

给你一个下标从 **0** 开始包含 `n` 个正整数的数组 `arr` ，和一个正整数 `k` 。

如果对于每个满足 `k <= i <= n-1` 的下标 `i` ，都有 `arr[i-k] <= arr[i]` ，那么我们称 `arr` 是 **K** **递增** 的。

-   比方说，

    ```
    arr = [4, 1, 5, 2, 6, 2]
    ```

     对于 

    ```
    k = 2
    ```

     是 K 递增的，因为：

    -   `arr[0] <= arr[2] (4 <= 5)`
    -   `arr[1] <= arr[3] (1 <= 2)`
    -   `arr[2] <= arr[4] (5 <= 6)`
    -   `arr[3] <= arr[5] (2 <= 2)`

-   但是，相同的数组 `arr` 对于 `k = 1` 不是 K 递增的（因为 `arr[0] > arr[1]`），对于 `k = 3` 也不是 K 递增的（因为 `arr[0] > arr[3]` ）。

每一次 **操作** 中，你可以选择一个下标 `i` 并将 `arr[i]` **改成任意** 正整数。

请你返回对于给定的 `k` ，使数组变成 K 递增的 **最少操作次数** 。

**示例 1：**

```
输入：arr = [5,4,3,2,1], k = 1
输出：4
解释：
对于 k = 1 ，数组最终必须变成非递减的。
可行的 K 递增结果数组为 [5,6,7,8,9]，[1,1,1,1,1]，[2,2,3,4,4] 。它们都需要 4 次操作。
次优解是将数组变成比方说 [6,7,8,9,10] ，因为需要 5 次操作。
显然我们无法使用少于 4 次操作将数组变成 K 递增的。
```

**示例 2：**

```
输入：arr = [4,1,5,2,6,2], k = 2
输出：0
解释：
这是题目描述中的例子。
对于每个满足 2 <= i <= 5 的下标 i ，有 arr[i-2] <= arr[i] 。
由于给定数组已经是 K 递增的，我们不需要进行任何操作。
```

**示例 3：**

```
输入：arr = [4,1,5,2,6,2], k = 3
输出：2
解释：
下标 3 和 5 是仅有的 3 <= i <= 5 且不满足 arr[i-3] <= arr[i] 的下标。
将数组变成 K 递增的方法之一是将 arr[3] 变为 4 ，且将 arr[5] 变成 5 。
数组变为 [4,1,5,4,6,5] 。
可能有其他方法将数组变为 K 递增的，但没有任何一种方法需要的操作次数小于 2 次。
```

**提示：**

-   `1 <= arr.length <= 105`
-   `1 <= arr[i], k <= arr.length`

**答案：**

>   LIS 板子题。比赛的时候用 binary_search 写崩了

```rust
impl Solution {
    pub fn k_increasing(arr: Vec<i32>, k: i32) -> i32 {
        let mut arrs = vec![vec![]; k as usize];
        arr.iter()
            .enumerate()
            .for_each(|(i, &v)| arrs[i % k as usize].push(v));
        println!("{:?}", arrs);
        let mut ans = 0;
        for arr in arrs {
            ans += Solution::lis(&arr);
        }
        ans
    }

    fn lis(arr: &Vec<i32>) -> i32 {
        let mut lis = vec![];
        for &num in arr {
            let pos = lis.partition_point(|&x| x > num);
            if pos == lis.len() {
                lis.push(num);
            } else {
                lis[pos] = num;
            }
        }
        (arr.len() - lis.len()) as i32
    }
}
```

**拓展：**

如果这道题回归应有的难度，即要求**严格递增**，那么应该怎么做？

显然直接分组是不可行的，因为因为题目另一个限制条件是数组里必须存正整数，那么类似2,1,2,3,4这样的序列是不能改成0,1,2,3,4的。

求解方法参考 [0x3f 佬的评论](https://leetcode-cn.com/circle/discuss/wMSHqV/view/qk9OXm/):

>   把每个数减去其下标，然后对所有正整数求最长非降子序列。
>
>   举个例子，现在每 kk 个数选一个数，假设选出来的数组是 [3,2,4,5,5,6,6][3,2,4,5,5,6,6]。
>
>   每个数减去其下标后就是 [3,1,2,2,1,1,0][3,1,2,2,1,1,0]。
>
>   对这个数组中的正整数求最长非降子序列，那就是 [1,1,1][1,1,1] 了，对应原始数组的 [*,2,*,*,5,6,*][∗,2,∗,∗,5,6,∗]，这三个数字保留，其余数字修改完成后就是 [1,2,3,4,5,6,7][1,2,3,4,5,6,7]，符合严格递增且均为正整数的要求。
>
>   注：上述减去下标的技巧，主要是为了能让保留的数字之间可以容纳严格递增的数字。否则，若直接按照最长严格递增子序列的求法，会得到例如 $[*,2,4,5,*,6,*]$ 这样的错误结果。
>
>   作者：灵茶山艾府
>   链接：https://leetcode-cn.com/circle/discuss/wMSHqV/view/qk9OXm/
>   来源：力扣（LeetCode）
>   著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
