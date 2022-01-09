---
title: LeetCode 周赛 273
date: 2021-12-26 17:26:21
tags: LeetCode 周赛
---

# LeetCode 周赛 273

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-273/problems/a-number-after-a-double-reversal/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-273/problems/execution-of-all-suffix-instructions-staying-in-a-grid/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-273/problems/intervals-between-identical-elements/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-273/problems/recover-the-original-array/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1777 / 4367 | MiracleSNeko | 7    | 0:21:23  | 0:01:26                                                      | 0:21:23                                                      |                                                              |                                                              |

>   写 bugfree 代码的能力是目前最大的瓶颈

## T3 5965. 相同元素的间隔之和

-   **User Accepted:**1331
-   **User Tried:**2930
-   **Total Accepted:**1377
-   **Total Submissions:**6371
-   **Difficulty:** **Medium**

给你一个下标从 **0** 开始、由 `n` 个整数组成的数组 `arr` 。

`arr` 中两个元素的 **间隔** 定义为它们下标之间的 **绝对差** 。更正式地，`arr[i]` 和 `arr[j]` 之间的间隔是 `|i - j|` 。

返回一个长度为 `n` 的数组 `intervals` ，其中 `intervals[i]` 是 `arr[i]` 和 `arr` 中每个相同元素（与 `arr[i]` 的值相同）的 **间隔之和** *。*

**注意：**`|x|` 是 `x` 的绝对值。

**示例 1：**

```
输入：arr = [2,1,3,1,2,3,3]
输出：[4,2,7,2,4,4,5]
解释：
- 下标 0 ：另一个 2 在下标 4 ，|0 - 4| = 4
- 下标 1 ：另一个 1 在下标 3 ，|1 - 3| = 2
- 下标 2 ：另两个 3 在下标 5 和 6 ，|2 - 5| + |2 - 6| = 7
- 下标 3 ：另一个 1 在下标 1 ，|3 - 1| = 2
- 下标 4 ：另一个 2 在下标 0 ，|4 - 0| = 4
- 下标 5 ：另两个 3 在下标 2 和 6 ，|5 - 2| + |5 - 6| = 4
- 下标 6 ：另两个 3 在下标 2 和 5 ，|6 - 2| + |6 - 5| = 5
```

**示例 2：**

```
输入：arr = [10,5,10,10]
输出：[5,0,3,4]
解释：
- 下标 0 ：另两个 10 在下标 2 和 3 ，|0 - 2| + |0 - 3| = 5
- 下标 1 ：只有这一个 5 在数组中，所以到相同元素的间隔之和是 0
- 下标 2 ：另两个 10 在下标 0 和 3 ，|2 - 0| + |2 - 3| = 3
- 下标 3 ：另两个 10 在下标 0 和 2 ，|3 - 0| + |3 - 2| = 4
```

**提示：**

-   `n == arr.length`
-   `1 <= n <= 105`
-   `1 <= arr[i] <= 105`

**提交：**

>   其实不需要记录数组，这样写只是看着方便

```rust
use std::collections::HashMap;

impl Solution {
    pub fn get_distances(arr: Vec<i32>) -> Vec<i64> {
        let mut cnts = arr.iter().enumerate().fold(HashMap::new(), |mut mp, (idx, &val)| {
            (*mp.entry(val).or_insert(vec![])).push(idx);
            mp
        });
        let mut ans = vec![0; arr.len()];
        cnts.iter().for_each(|(_, arr)| Solution::distance(arr, &mut ans));
        ans
    }

    pub fn distance(arr: &Vec<usize>, ans: &mut Vec<i64>) {
        let mut diff = vec![0; arr.len()];
        let mut sum = vec![0; arr.len()];
        for i in 1..arr.len() {
            diff[i] = (arr[i] - arr[0]) as i64;
        }
        for i in 1..arr.len() {
            sum[i] = sum[i - 1] + diff[i];
        }
        ans[arr[0]] = sum[arr.len() - 1];
        for i in 1..arr.len() {
            // diff 记录 [0, x1 - x0, ... xn-1 - x0]
            // sum 记录 \sum (xi - x0)
            // 以 x_i 为分界，左侧和取反加上 i 个 xi - x0
            // 右侧和减去 n - i + 1 个 xi - x0
            let left = sum[i - 1];
            let right = sum[arr.len() - 1] - sum[i - 1];
            let curr = -left + i as i64 * diff[i] + right - (arr.len() - i) as i64 * diff[i];
            ans[arr[i]] = curr;
        }
    }
}
```

## T4 5966. 还原原数组

-   **User Accepted:**531
-   **User Tried:**998
-   **Total Accepted:**603
-   **Total Submissions:**2237
-   **Difficulty:** **Hard**

Alice 有一个下标从 **0** 开始的数组 `arr` ，由 `n` 个正整数组成。她会选择一个任意的 **正整数** `k` 并按下述方式创建两个下标从 **0** 开始的新整数数组 `lower` 和 `higher` ：

1.  对每个满足 `0 <= i < n` 的下标 `i` ，`lower[i] = arr[i] - k`
2.  对每个满足 `0 <= i < n` 的下标 `i` ，`higher[i] = arr[i] + k`

不幸地是，Alice 丢失了全部三个数组。但是，她记住了在数组 `lower` 和 `higher` 中出现的整数，但不知道每个整数属于哪个数组。请你帮助 Alice 还原原数组。

给你一个由 2n 个整数组成的整数数组 `nums` ，其中 **恰好** `n` 个整数出现在 `lower` ，剩下的出现在 `higher` ，还原并返回 **原数组** `arr` 。如果出现答案不唯一的情况，返回 **任一** 有效数组。

**注意：**生成的测试用例保证存在 **至少一个** 有效数组 `arr` 。

 **示例 1：**

```
输入：nums = [2,10,6,4,8,12]
输出：[3,7,11]
解释：
如果 arr = [3,7,11] 且 k = 1 ，那么 lower = [2,6,10] 且 higher = [4,8,12] 。
组合 lower 和 higher 得到 [2,6,10,4,8,12] ，这是 nums 的一个排列。
另一个有效的数组是 arr = [5,7,9] 且 k = 3 。在这种情况下，lower = [2,4,6] 且 higher = [8,10,12] 。
```

**示例 2：**

```
输入：nums = [1,1,3,3]
输出：[2,2]
解释：
如果 arr = [2,2] 且 k = 1 ，那么 lower = [1,1] 且 higher = [3,3] 。
组合 lower 和 higher 得到 [1,1,3,3] ，这是 nums 的一个排列。
注意，数组不能是 [1,3] ，因为在这种情况下，获得 [1,1,3,3] 唯一可行的方案是 k = 0 。
这种方案是无效的，k 必须是一个正整数。
```

**示例 3：**

```
输入：nums = [5,435]
输出：[220]
解释：
唯一可行的组合是 arr = [220] 且 k = 215 。在这种情况下，lower = [5] 且 higher = [435] 。
```

 **提示：**

-   `2 * n == nums.length`
-   `1 <= n <= 1000`
-   `1 <= nums[i] <= 109`
-   生成的测试用例保证存在 **至少一个** 有效数组 `arr`

**答案：**

>   枚举可行的所有 k 值即可。显然 k 可以通过枚举数组里最大值或最小值与其他值的差值得到。

```rust
impl Solution {
    pub fn recover_array(mut nums: Vec<i32>) -> Vec<i32> {
        nums.sort();
        let len = nums.len();
        for i in 1..len {
            let k = nums[i] - nums[0];
            if k == 0 || k & 1 == 1 {
                continue;
            }
            let mut vis = vec![false; len];
            let mut ans = vec![nums[0] + k / 2];
            vis[i] = true;
            let (mut l, mut r) = (1, i + 1);
            while r < len {
                while l < len && vis[l] {
                    l += 1;
                }
                while r < len && nums[r] - nums[l] < k {
                    r += 1;
                }
                if r < len && nums[r] - nums[l] > k {
                    break;
                }
                vis[l] = true;
                vis[r] = true;
                ans.push(nums[l] + k / 2);
                l += 1;
                r += 1;
            }
            if ans.len() == len / 2 {
                return ans;
            }
        }
        unreachable!()
    }
}
```



