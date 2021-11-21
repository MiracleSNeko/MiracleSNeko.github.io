---
title: LeetCode 周赛 268
date: 2021-11-21 12:33:15
tags: LeetCode 周赛总结
---

----------

# LeetCode 周赛 268

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-268/problems/two-furthest-houses-with-different-colors/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-268/problems/watering-plants/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-268/problems/range-frequency-queries/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-268/problems/sum-of-k-mirror-numbers/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1087 / 4397 | MiracleSNeko | 12   | 1:23:11  | 0:12:25                                                      | 0:25:11                                                      | 1:08:11 3                                                    |                                                              |

>   T1 的数据量直接暴力，直接暴力，直接暴力，不然就会出现我这种 T1 写十分钟的 sb 事情

## T1 5930. 两栋颜色不同且距离最远的房子

-   **User Accepted:**3449
-   **User Tried:**3518
-   **Total Accepted:**3504
-   **Total Submissions:**4578
-   **Difficulty:** **Easy**

街上有 `n` 栋房子整齐地排成一列，每栋房子都粉刷上了漂亮的颜色。给你一个下标从 **0** 开始且长度为 `n` 的整数数组 `colors` ，其中 `colors[i]` 表示第 `i` 栋房子的颜色。

返回 **两栋** 颜色 **不同** 房子之间的 **最大** 距离。

第 `i` 栋房子和第 `j` 栋房子之间的距离是 `abs(i - j)` ，其中 `abs(x)` 是 `x` 的绝对值。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/10/31/eg1.png)

```
输入：colors = [1,1,1,6,1,1,1]
输出：3
解释：上图中，颜色 1 标识成蓝色，颜色 6 标识成红色。
两栋颜色不同且距离最远的房子是房子 0 和房子 3 。
房子 0 的颜色是颜色 1 ，房子 3 的颜色是颜色 6 。两栋房子之间的距离是 abs(0 - 3) = 3 。
注意，房子 3 和房子 6 也可以产生最佳答案。
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/10/31/eg2.png)

```
输入：colors = [1,8,3,8,3]
输出：4
解释：上图中，颜色 1 标识成蓝色，颜色 8 标识成黄色，颜色 3 标识成绿色。
两栋颜色不同且距离最远的房子是房子 0 和房子 4 。
房子 0 的颜色是颜色 1 ，房子 4 的颜色是颜色 3 。两栋房子之间的距离是 abs(0 - 4) = 4 。
```

**示例 3：**

```
输入：colors = [0,1]
输出：1
解释：两栋颜色不同且距离最远的房子是房子 0 和房子 1 。
房子 0 的颜色是颜色 0 ，房子 1 的颜色是颜色 1 。两栋房子之间的距离是 abs(0 - 1) = 1 。
```

**提示：**

-   `n == colors.length`
-   `2 <= n <= 100`
-   `0 <= colors[i] <= 100`
-   生成的测试数据满足 **至少** 存在 2 栋颜色不同的房子

**提交：**

```rust
impl Solution {
    pub fn max_distance(colors: Vec<i32>) -> i32 {
        let len = colors.len();
        let mut ans = 0;
        for i in 0..len {
            for j in i..len {
                if colors[i] != colors[j] {
                    ans = ans.max(j - i);
                }
            }
        }
        ans as i32
    }
}
```

## T2 5201. 给植物浇水

-   **User Accepted:**3240
-   **User Tried:**3308
-   **Total Accepted:**3274
-   **Total Submissions:**4243
-   **Difficulty:** **Medium**

你打算用一个水罐给花园里的 `n` 株植物浇水。植物排成一行，从左到右进行标记，编号从 `0` 到 `n - 1` 。其中，第 `i` 株植物的位置是 `x = i` 。`x = -1` 处有一条河，你可以在那里重新灌满你的水罐。

每一株植物都需要浇特定量的水。你将会按下面描述的方式完成浇水：

-   按从左到右的顺序给植物浇水。
-   在给当前植物浇完水之后，如果你没有足够的水 **完全** 浇灌下一株植物，那么你就需要返回河边重新装满水罐。
-   你 **不能** 提前重新灌满水罐。

最初，你在河边（也就是，`x = -1`），在 x 轴上每移动 **一个单位** 都需要 **一步** 。

给你一个下标从 **0** 开始的整数数组 `plants` ，数组由 `n` 个整数组成。其中，`plants[i]` 为第 `i` 株植物需要的水量。另有一个整数 `capacity` 表示水罐的容量，返回浇灌所有植物需要的 **步数** 。

**示例 1：**

```
输入：plants = [2,2,3,3], capacity = 5
输出：14
解释：从河边开始，此时水罐是装满的：
- 走到植物 0 (1 步) ，浇水。水罐中还有 3 单位的水。
- 走到植物 1 (1 步) ，浇水。水罐中还有 1 单位的水。
- 由于不能完全浇灌植物 2 ，回到河边取水 (2 步)。
- 走到植物 2 (3 步) ，浇水。水罐中还有 2 单位的水。
- 由于不能完全浇灌植物 3 ，回到河边取水 (3 步)。
- 走到植物 3 (4 步) ，浇水。
需要的步数是 = 1 + 1 + 2 + 3 + 3 + 4 = 14 。
```

**示例 2：**

```
输入：plants = [1,1,1,4,2,3], capacity = 4
输出：30
解释：从河边开始，此时水罐是装满的：
- 走到植物 0，1，2 (3 步) ，浇水。回到河边取水 (3 步)。
- 走到植物 3 (4 步) ，浇水。回到河边取水 (4 步)。
- 走到植物 4 (5 步) ，浇水。回到河边取水 (5 步)。
- 走到植物 5 (6 步) ，浇水。
需要的步数是 = 3 + 3 + 4 + 4 + 5 + 5 + 6 = 30 。
```

**示例 3：**

```
输入：plants = [7,7,7,7,7,7,7], capacity = 8
输出：49
解释：每次浇水都需要重新灌满水罐。
需要的步数是 = 1 + 1 + 2 + 2 + 3 + 3 + 4 + 4 + 5 + 5 + 6 + 6 + 7 = 49 。
```

**提示：**

-   `n == plants.length`
-   `1 <= n <= 1000`
-   `1 <= plants[i] <= 106`
-   `max(plants[i]) <= capacity <= 109`

**提交：**

>   模拟，注意是在前一个植物的位置就发现水不足需要折返

```rust
impl Solution {
    pub fn watering_plants(plants: Vec<i32>, capacity: i32) -> i32 {
        let mut steps = 1;
        let mut i = 0;
        let mut rest = capacity;
        while i != plants.len() - 1 {
            rest -= plants[i];
            // next
            if rest >= plants[i + 1] {
                steps += 1;
            }
            // back
            else {
                steps += 1 + 2 * (i + 1);
                rest = capacity;
            }
            i += 1;
        }
        steps as i32
    }
}
```

## T3 5186. 区间内查询数字的频率

-   **User Accepted:**1280
-   **User Tried:**2923
-   **Total Accepted:**1332
-   **Total Submissions:**9561
-   **Difficulty:** **Medium**

请你设计一个数据结构，它能求出给定子数组内一个给定值的 **频率** 。

子数组中一个值的 **频率** 指的是这个子数组中这个值的出现次数。

请你实现 `RangeFreqQuery` 类：

-   `RangeFreqQuery(int[] arr)` 用下标从 **0** 开始的整数数组 `arr` 构造一个类的实例。
-   `int query(int left, int right, int value)` 返回子数组 `arr[left...right]` 中 `value` 的 **频率** 。

一个 **子数组** 指的是数组中一段连续的元素。`arr[left...right]` 指的是 `nums` 中包含下标 `left` 和 `right` **在内** 的中间一段连续元素。

**示例 1：**

```
输入：
["RangeFreqQuery", "query", "query"]
[[[12, 33, 4, 56, 22, 2, 34, 33, 22, 12, 34, 56]], [1, 2, 4], [0, 11, 33]]
输出：
[null, 1, 2]

解释：
RangeFreqQuery rangeFreqQuery = new RangeFreqQuery([12, 33, 4, 56, 22, 2, 34, 33, 22, 12, 34, 56]);
rangeFreqQuery.query(1, 2, 4); // 返回 1 。4 在子数组 [33, 4] 中出现 1 次。
rangeFreqQuery.query(0, 11, 33); // 返回 2 。33 在整个子数组中出现 2 次。
```

**提示：**

-   `1 <= arr.length <= 105`
-   `1 <= arr[i], value <= 104`
-   `0 <= left <= right < arr.length`
-   调用 `query` 不超过 `105` 次。

**提交：**

>   一开始的 upper_bound 写的有错，需要几个特判边界条件，修改后没了

```rust
use std::collections::HashMap;

struct RangeFreqQuery {
    pos: HashMap<i32, Vec<usize>>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl RangeFreqQuery {
    fn new(arr: Vec<i32>) -> Self {
        let pos = arr
            .iter()
            .enumerate()
            .fold(HashMap::new(), |mut mp, (u, &i)| {
                (*mp.entry(i).or_insert(vec![])).push(u);
                mp
            });
        Self { pos }
    }

    fn query(&self, left: i32, right: i32, value: i32) -> i32 {
        if let Some(arr) = self.pos.get(&value) {
            let l = Self::lower_bound(arr, left as usize);
            let r = Self::upper_bound(arr, right as usize);
            r as i32 - l as i32
        } else {
            0
        }
    }

    fn lower_bound(arr: &Vec<usize>, target: usize) -> usize {
        let mut l = 0;
        let mut r = arr.len();
        while l < r {
            let m = (l + r) >> 1;
            if arr[m] >= target {
                r = m;
            } else {
                l = m + 1;
            }
        }
        l
    }

    fn upper_bound(arr: &Vec<usize>, target: usize) -> usize {
        let mut l = 0;
        let mut r = arr.len();
        while l < r {
            let m = (l + r) >> 1;
            if arr[m] > target {
                r = m;
            } else {
                l = m + 1;
            }
        }
        l
    }
}
```

## T4 5933. k 镜像数字的和

-   **User Accepted:**238
-   **User Tried:**498
-   **Total Accepted:**284
-   **Total Submissions:**967
-   **Difficulty:** **Hard**

一个 **k 镜像数字** 指的是一个在十进制和 k 进制下从前往后读和从后往前读都一样的 **没有前导 0** 的 **正** 整数。

-   比方说，`9` 是一个 2 镜像数字。`9` 在十进制下为 `9` ，二进制下为 `1001` ，两者从前往后读和从后往前读都一样。
-   相反地，`4` 不是一个 2 镜像数字。`4` 在二进制下为 `100` ，从前往后和从后往前读不相同。

给你进制 `k` 和一个数字 `n` ，请你返回 k 镜像数字中 **最小** 的 `n` 个数 **之和** 。

**示例 1：**

```
输入：k = 2, n = 5
输出：25
解释：
最小的 5 个 2 镜像数字和它们的二进制表示如下：
  十进制       二进制
    1          1
    3          11
    5          101
    7          111
    9          1001
它们的和为 1 + 3 + 5 + 7 + 9 = 25 。
```

**示例 2：**

```
输入：k = 3, n = 7
输出：499
解释：
7 个最小的 3 镜像数字和它们的三进制表示如下：
  十进制       三进制
    1          1
    2          2
    4          11
    8          22
    121        11111
    151        12121
    212        21212
它们的和为 1 + 2 + 4 + 8 + 121 + 151 + 212 = 499 。
```

**示例 3：**

```
输入：k = 7, n = 17
输出：20379000
解释：17 个最小的 7 镜像数字分别为：
1, 2, 3, 4, 5, 6, 8, 121, 171, 242, 292, 16561, 65656, 2137312, 4602064, 6597956, 6958596
```

**提示：**

-   `2 <= k <= 9`
-   `1 <= n <= 30`

**题解：**

>   OEIS 可以直接查表。打表的思路和正经做差不多。遍历 k 进制对称数，并判断十进制是否对称。生成对称数时按位数 &1 分为两种情况 dfs 即可

