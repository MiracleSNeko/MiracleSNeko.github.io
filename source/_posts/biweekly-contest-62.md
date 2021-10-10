---
title: LeetCode 双周赛 62
date: 2021-10-06 11:51:36
tags: LeetCode 周赛总结
---
---

# LeetCode 双周赛 62

| 排名       | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/biweekly-contest-62/problems/convert-1d-array-into-2d-array/) | [题目2 (4)](https://leetcode-cn.com/contest/biweekly-contest-62/problems/number-of-pairs-of-strings-with-concatenation-equal-to-target/) | [题目3 (5)](https://leetcode-cn.com/contest/biweekly-contest-62/problems/maximize-the-confusion-of-an-exam/) | [题目4 (6)](https://leetcode-cn.com/contest/biweekly-contest-62/problems/maximum-number-of-ways-to-partition-an-array/) |
| ---------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 715 / 2619 | MiracleSNeko | 12   | 1:07:20  | 0:03:20                                                      | 0:08:27 1                                                    | 0:52:20 2                                                    |                                                              |

## T1 2022. 将一维数组转变成二维数组

-   **通过的用户数**1673
-   **尝试过的用户数**1718
-   **用户总通过次数**1706
-   **用户总提交次数**2629
-   **题目难度** **Easy**

给你一个下标从 **0** 开始的一维整数数组 `original` 和两个整数 `m` 和 `n` 。你需要使用 `original` 中 **所有** 元素创建一个 `m` 行 `n` 列的二维数组。

`original` 中下标从 `0` 到 `n - 1` （都 **包含** ）的元素构成二维数组的第一行，下标从 `n` 到 `2 * n - 1` （都 **包含** ）的元素构成二维数组的第二行，依此类推。

请你根据上述过程返回一个 `m x n` 的二维数组。如果无法构成这样的二维数组，请你返回一个空的二维数组。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/08/26/image-20210826114243-1.png)

```
输入：original = [1,2,3,4], m = 2, n = 2
输出：[[1,2],[3,4]]
解释：
构造出的二维数组应该包含 2 行 2 列。
original 中第一个 n=2 的部分为 [1,2] ，构成二维数组的第一行。
original 中第二个 n=2 的部分为 [3,4] ，构成二维数组的第二行。
```

**示例 2：**

```
输入：original = [1,2,3], m = 1, n = 3
输出：[[1,2,3]]
解释：
构造出的二维数组应该包含 1 行 3 列。
将 original 中所有三个元素放入第一行中，构成要求的二维数组。
```

**示例 3：**

```
输入：original = [1,2], m = 1, n = 1
输出：[]
解释：
original 中有 2 个元素。
无法将 2 个元素放入到一个 1x1 的二维数组中，所以返回一个空的二维数组。
```

**示例 4：**

```
输入：original = [3], m = 1, n = 2
输出：[]
解释：
original 中只有 1 个元素。
无法将 1 个元素放满一个 1x2 的二维数组，所以返回一个空的二维数组。
```

**提示：**

-   `1 <= original.length <= 5 * 104`
-   `1 <= original[i] <= 105`
-   `1 <= m, n <= 4 * 104`

**我的提交：**

>   应该有 api 可以直接用，但是比赛懒得查了

```rust
impl Solution {
    pub fn construct2_d_array(original: Vec<i32>, m: i32, n: i32) -> Vec<Vec<i32>> {
        if m * n != original.len() as i32 {
            vec![]
        } else {
            let mut ret = Vec::with_capacity(m as usize);
            let mut tmp = Vec::with_capacity(n as usize);
            for i in 0..original.len() {
                if tmp.len() == n as usize {
                    ret.push(tmp.clone());
                    tmp.clear();
                }
                tmp.push(original[i]);
            }
            ret.push(tmp);
            ret
        }
    }
}
```



## T2 2023. 连接后等于目标字符串的字符串对

-   **通过的用户数**1623
-   **尝试过的用户数**1654
-   **用户总通过次数**1647
-   **用户总提交次数**2202
-   **题目难度** **Medium**

给你一个 **数字** 字符串数组 `nums` 和一个 **数字** 字符串 `target` ，请你返回 `nums[i] + nums[j]` （两个字符串连接）结果等于 `target` 的下标 `(i, j)` （需满足 `i != j`）的数目。



**示例 1：**

```
输入：nums = ["777","7","77","77"], target = "7777"
输出：4
解释：符合要求的下标对包括：
- (0, 1)："777" + "7"
- (1, 0)："7" + "777"
- (2, 3)："77" + "77"
- (3, 2)："77" + "77"
```

**示例 2：**

```
输入：nums = ["123","4","12","34"], target = "1234"
输出：2
解释：符合要求的下标对包括
- (0, 1)："123" + "4"
- (2, 3)："12" + "34"
```

**示例 3：**

```
输入：nums = ["1","1","1"], target = "11"
输出：6
解释：符合要求的下标对包括
- (0, 1)："1" + "1"
- (1, 0)："1" + "1"
- (0, 2)："1" + "1"
- (2, 0)："1" + "1"
- (1, 2)："1" + "1"
- (2, 1)："1" + "1"
```



**提示：**

-   `2 <= nums.length <= 100`
-   `1 <= nums[i].length <= 100`
-   `2 <= target.length <= 100`
-   `nums[i]` 和 `target` 只包含数字。
-   `nums[i]` 和 `target` 不含有任何前导 0 。

**我的提交：**

>   一开始想写 dfs，看了眼数据量果断模拟。同一场周赛出现两道模拟送分题还真是少见。

```rust
impl Solution {
    pub fn num_of_pairs(nums: Vec<String>, target: String) -> i32 {
        let mut cnt = 0;
        for i in 0..nums.len() {
            for j in 0..nums.len() {
                if i == j  {
                    continue;
                }
                let cat = nums[i].clone() + &nums[j];
                if cat == target {
                    cnt += 1;
                }
            }
        }
        cnt
    }
}
```



## T3 2024. 考试的最大困扰度

-   **通过的用户数**861
-   **尝试过的用户数**1139
-   **用户总通过次数**889
-   **用户总提交次数**2307
-   **题目难度** **Medium**

一位老师正在出一场由 `n` 道判断题构成的考试，每道题的答案为 true （用 `'T'` 表示）或者 false （用 `'F'` 表示）。老师想增加学生对自己做出答案的不确定性，方法是 **最大化** 有 **连续相同** 结果的题数。（也就是连续出现 true 或者连续出现 false）。

给你一个字符串 `answerKey` ，其中 `answerKey[i]` 是第 `i` 个问题的正确结果。除此以外，还给你一个整数 `k` ，表示你能进行以下操作的最多次数：

-   每次操作中，将问题的正确答案改为 `'T'` 或者 `'F'` （也就是将 `answerKey[i]` 改为 `'T'` 或者 `'F'` ）。

请你返回在不超过 `k` 次操作的情况下，**最大** 连续 `'T'` 或者 `'F'` 的数目。

 

**示例 1：**

```
输入：answerKey = "TTFF", k = 2
输出：4
解释：我们可以将两个 'F' 都变为 'T' ，得到 answerKey = "TTTT" 。
总共有四个连续的 'T' 。
```

**示例 2：**

```
输入：answerKey = "TFFT", k = 1
输出：3
解释：我们可以将最前面的 'T' 换成 'F' ，得到 answerKey = "FFFT" 。
或者，我们可以将第二个 'T' 换成 'F' ，得到 answerKey = "TFFF" 。
两种情况下，都有三个连续的 'F' 。
```

**示例 3：**

```
输入：answerKey = "TTFTTFTT", k = 1
输出：5
解释：我们可以将第一个 'F' 换成 'T' ，得到 answerKey = "TTTTTFTT" 。
或者我们可以将第二个 'F' 换成 'T' ，得到 answerKey = "TTFTTTTT" 。
两种情况下，都有五个连续的 'T' 。
```

 

**提示：**

-   `n == answerKey.length`
-   `1 <= n <= 5 * 104`
-   `answerKey[i]` 要么是 `'T'` ，要么是 `'F'`
-   `1 <= k <= n`

**我的提交：**

>   参考 [LC487](https://leetcode-cn.com/problems/max-consecutive-ones-ii/) 和 [LC1004](https://leetcode-cn.com/problems/max-consecutive-ones-iii) 。该题的最佳解法为滑动窗口，周赛的时候写了一个次优的二分。

```rust
use std::cell::RefCell;
use std::cmp::{max, min, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::rc::Rc;
use std::ops::Bound;

impl Solution {
    pub fn max_consecutive_answers(answer_key: String, k: i32) -> i32 {
        let mut ans = 0;
        let len = answer_key.len();
        let answer_key = answer_key.as_bytes();
        let ts = answer_key
            .iter()
            .map(|&c| if c == b'T' { 1 } else { 0 })
            .fold(vec![0], |mut vec, i| {
                vec.push(vec.last().unwrap() + i);
                vec
            });
        let fs = answer_key
            .iter()
            .map(|&c| if c == b'F' { 1 } else { 0 })
            .fold(vec![0], |mut vec, i| {
                vec.push(vec.last().unwrap() + i);
                vec
            });
        // T 和 F 各算一遍算逑
        fn lower_bound(val: &Vec<i32>, tar: i32) -> usize {
            let (mut l, mut r) = (0, val.len());
            while l < r {
                let m = (l + r) >> 1;
                if val[m] < tar {
                    l = m + 1;
                } else {
                    r = m;
                }
            }
            l
        }
        for r in 1..ts.len() {
            let l = lower_bound(&ts, ts[r] - k);
            ans = ans.max(r - l);
        }
        for r in 1..fs.len() {
            let l = lower_bound(&fs, fs[r] - k);
            ans = ans.max(r - l);
        }
        ans as i32
    }
}
```



## T4 2025. 分割数组的最多方案数

-   **通过的用户数**267
-   **尝试过的用户数**641
-   **用户总通过次数**296
-   **用户总提交次数**1955
-   **题目难度** **Hard**

给你一个下标从 **0** 开始且长度为 `n` 的整数数组 `nums` 。**分割** 数组 `nums` 的方案数定义为符合以下两个条件的 `pivot` 数目：

-   `1 <= pivot < n`
-   `nums[0] + nums[1] + ... + nums[pivot - 1] == nums[pivot] + nums[pivot + 1] + ... + nums[n - 1]`

同时给你一个整数 `k` 。你可以将 `nums` 中 **一个** 元素变为 `k` 或 **不改变** 数组。

请你返回在 **至多** 改变一个元素的前提下，**最多** 有多少种方法 **分割** `nums` 使得上述两个条件都满足。

 

**示例 1：**

```
输入：nums = [2,-1,2], k = 3
输出：1
解释：一个最优的方案是将 nums[0] 改为 k 。数组变为 [3,-1,2] 。
有一种方法分割数组：
- pivot = 2 ，我们有分割 [3,-1 | 2]：3 + -1 == 2 。
```

**示例 2：**

```
输入：nums = [0,0,0], k = 1
输出：2
解释：一个最优的方案是不改动数组。
有两种方法分割数组：
- pivot = 1 ，我们有分割 [0 | 0,0]：0 == 0 + 0 。
- pivot = 2 ，我们有分割 [0,0 | 0]: 0 + 0 == 0 。
```

**示例 3：**

```
输入：nums = [22,4,-25,-20,-15,15,-16,7,19,-10,0,-13,-14], k = -33
输出：4
解释：一个最优的方案是将 nums[2] 改为 k 。数组变为 [22,4,-33,-20,-15,15,-16,7,19,-10,0,-13,-14] 。
有四种方法分割数组。
```

 

**提示：**

-   `n == nums.length`
-   `2 <= n <= 105`
-   `-105 <= k, nums[i] <= 105`
