---
title: LeetCode 周赛 271
date: 2021-12-12 19:43:20
tags: LeetCode 周赛
---

----------

# LeetCode 周赛 271

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-271/problems/rings-and-rods/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-271/problems/sum-of-subarray-ranges/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-271/problems/watering-plants-ii/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-271/problems/maximum-fruits-harvested-after-at-most-k-steps/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2014 / 4561 | MiracleSNeko | 12   | 1:22:25  | 0:38:43                                                      | 0:56:50 1                                                    | 1:17:25                                                      |                                                              |

>   两个周没打，手慢的一笔。要是早上没有晚起半小时其实能 AK 的
>
>   还有，这周赛也越来越水了，怎么全都是些一眼板子题

## T1 5952. 环和杆

-   **User Accepted:** 3486
-   **User Tried:** 3540
-   **Total Accepted:** 3524
-   **Total Submissions:** 4249
-   **Difficulty:** **Easy**

总计有 `n` 个环，环的颜色可以是红、绿、蓝中的一种。这些环分布穿在 10 根编号为 `0` 到 `9` 的杆上。

给你一个长度为 `2n` 的字符串 `rings` ，表示这 `n` 个环在杆上的分布。`rings` 中每两个字符形成一个 **颜色位置对** ，用于描述每个环：

-   第 `i` 对中的 **第一个** 字符表示第 `i` 个环的 **颜色**（`'R'`、`'G'`、`'B'`）。
-   第 `i` 对中的 **第二个** 字符表示第 `i` 个环的 **位置**，也就是位于哪根杆上（`'0'` 到 `'9'`）。

例如，`"R3G2B1"` 表示：共有 `n == 3` 个环，红色的环在编号为 3 的杆上，绿色的环在编号为 2 的杆上，蓝色的环在编号为 1 的杆上。

找出所有集齐 **全部三种颜色** 环的杆，并返回这种杆的数量。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/11/23/ex1final.png)

```
输入：rings = "B0B6G0R6R0R6G9"
输出：1
解释：
- 编号 0 的杆上有 3 个环，集齐全部颜色：红、绿、蓝。
- 编号 6 的杆上有 3 个环，但只有红、蓝两种颜色。
- 编号 9 的杆上只有 1 个绿色环。
因此，集齐全部三种颜色环的杆的数目为 1 。
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/11/23/ex2final.png)

```
输入：rings = "B0R0G0R9R0B0G0"
输出：1
解释：
- 编号 0 的杆上有 6 个环，集齐全部颜色：红、绿、蓝。
- 编号 9 的杆上只有 1 个红色环。
因此，集齐全部三种颜色环的杆的数目为 1 。
```

**示例 3：**

```
输入：rings = "G4"
输出：0
解释：
只给了一个环，因此，不存在集齐全部三种颜色环的杆。
```

**提示：**

-   `rings.length == 2 * n`
-   `1 <= n <= 100`
-   如 `i` 是 **偶数** ，则 `rings[i]` 的值可以取 `'R'`、`'G'` 或 `'B'`（下标从 **0** 开始计数）
-   如 `i` 是 **奇数** ，则 `rings[i]` 的值可以取 `'0'` 到 `'9'` 中的一个数字（下标从 **0** 开始计数）

**提交：**

>   纯模拟

```rust
use std::collections::HashMap;

impl Solution {
    pub fn count_points(rings: String) -> i32 {
        let mut ring_cnt = vec![HashMap::new(); 10];
        rings.as_bytes().windows(2).step_by(2)
            .for_each(|ch| {
                *ring_cnt[(ch[1] - b'0') as usize].entry(ch[0]).or_insert(0) += 1;
            });
        ring_cnt.into_iter().filter(|mp| mp.len() == 3).count() as i32
    }
}
```



## T2 5953. 子数组范围和

-   **User Accepted:** 2520
-   **User Tried:** 3068
-   **Total Accepted:** 2556
-   **Total Submissions:** 5122
-   **Difficulty:** **Medium**

给你一个整数数组 `nums` 。`nums` 中，子数组的 **范围** 是子数组中最大元素和最小元素的差值。

返回 `nums` 中 **所有** 子数组范围的 **和** *。*

子数组是数组中一个连续 **非空** 的元素序列。

 

**示例 1：**

```
输入：nums = [1,2,3]
输出：4
解释：nums 的 6 个子数组如下所示：
[1]，范围 = 最大 - 最小 = 1 - 1 = 0 
[2]，范围 = 2 - 2 = 0
[3]，范围 = 3 - 3 = 0
[1,2]，范围 = 2 - 1 = 1
[2,3]，范围 = 3 - 2 = 1
[1,2,3]，范围 = 3 - 1 = 2
所有范围的和是 0 + 0 + 0 + 1 + 1 + 2 = 4
```

**示例 2：**

```
输入：nums = [1,3,3]
输出：4
解释：nums 的 6 个子数组如下所示：
[1]，范围 = 最大 - 最小 = 1 - 1 = 0
[3]，范围 = 3 - 3 = 0
[3]，范围 = 3 - 3 = 0
[1,3]，范围 = 3 - 1 = 2
[3,3]，范围 = 3 - 3 = 0
[1,3,3]，范围 = 3 - 1 = 2
所有范围的和是 0 + 0 + 0 + 2 + 0 + 2 = 4
```

**示例 3：**

```
输入：nums = [4,-2,-3,4,1]
输出：59
解释：nums 中所有子数组范围的和是 59
```

**提示：**

-   `1 <= nums.length <= 1000`
-   `-109 <= nums[i] <= 109`

**提交：**

>   想都没想直接复制了 ST 表的板子……其实这个数据量直接暴力就可以了。
>
>   也可以用 dp 计算当前区间最值，甚至还可以用单调栈。

```rust
struct SparseTable {
    datamin: Vec<Vec<i32>>,
    datamax: Vec<Vec<i32>>,
    prelog: Vec<usize>,
}

impl SparseTable {
    pub fn init_with(vals: &Vec<i32>) -> Self {
        let mut prelog = vec![0; 100010];
        prelog[1] = 0;
        for i in 2..100010 {
            prelog[i] = prelog[i - 1];
            if (1 << prelog[i] + 1) == i {
                prelog[i] += 1;
            }
        }
        let mut datamin = vec![vec![0; 20]; 100010];
        let mut datamax = vec![vec![0; 20]; 100010];
        let len = vals.len();
        for i in (0..len).rev() {
            datamax[i][0] = vals[i];
            datamin[i][0] = vals[i];
            for j in 1..20 {
                if i + (1 << j - 1) < len {
                    datamax[i][j] = datamax[i][j - 1].max(datamax[i + (1 << (j - 1))][j - 1]);
                    datamin[i][j] = datamin[i][j - 1].min(datamin[i + (1 << (j - 1))][j - 1]);
                }
                else {
                    break;
                }
            }
        }
        Self { datamax, datamin, prelog }
    }
    pub fn query(&self, l: usize, r: usize) -> i32 {
        let k = self.prelog[r - l + 1];
        self.datamax[l][k].max(self.datamax[r + 1 - (1 << k)][k]) - self.datamin[l][k].min(self.datamin[r + 1 - (1 << k)][k])
    }
}

impl Solution {
    pub fn sub_array_ranges(nums: Vec<i32>) -> i64 {
        let st = SparseTable::init_with(&nums);
        let len = nums.len();
        let mut sum = 0i64;
        for i in 0..len {
            for j in i..len {
                sum += st.query(i, j) as i64;
            }
        }
        sum
    }
}
```



## T3 5954. 给植物浇水 II

-   **User Accepted:** 2591
-   **User Tried:** 2806
-   **Total Accepted:** 2625
-   **Total Submissions:** 4789
-   **Difficulty:** **Medium**

Alice 和 Bob 打算给花园里的 `n` 株植物浇水。植物排成一行，从左到右进行标记，编号从 `0` 到 `n - 1` 。其中，第 `i` 株植物的位置是 `x = i` 。

每一株植物都需要浇特定量的水。Alice 和 Bob 每人有一个水罐，**最初是满的** 。他们按下面描述的方式完成浇水：

-    Alice 按 **从左到右** 的顺序给植物浇水，从植物 `0` 开始。Bob 按 **从右到左** 的顺序给植物浇水，从植物 `n - 1` 开始。他们 **同时** 给植物浇水。
-   如果没有足够的水 **完全** 浇灌下一株植物，他 / 她会立即重新灌满浇水罐。
-   不管植物需要多少水，浇水所耗费的时间都是一样的。
-   **不能** 提前重新灌满水罐。
-   每株植物都可以由 Alice 或者 Bob 来浇水。
-   如果 Alice 和 Bob 到达同一株植物，那么当前水罐中水更多的人会给这株植物浇水。如果他俩水量相同，那么 Alice 会给这株植物浇水。

给你一个下标从 **0** 开始的整数数组 `plants` ，数组由 `n` 个整数组成。其中，`plants[i]` 为第 `i` 株植物需要的水量。另有两个整数 `capacityA` 和 `capacityB` 分别表示 Alice 和 Bob 水罐的容量。返回两人浇灌所有植物过程中重新灌满水罐的 **次数** 。

 

**示例 1：**

```
输入：plants = [2,2,3,3], capacityA = 5, capacityB = 5
输出：1
解释：
- 最初，Alice 和 Bob 的水罐中各有 5 单元水。
- Alice 给植物 0 浇水，Bob 给植物 3 浇水。
- Alice 和 Bob 现在分别剩下 3 单元和 2 单元水。
- Alice 有足够的水给植物 1 ，所以她直接浇水。Bob 的水不够给植物 2 ，所以他先重新装满水，再浇水。
所以，两人浇灌所有植物过程中重新灌满水罐的次数 = 0 + 0 + 1 + 0 = 1 。
```

**示例 2：**

```
输入：plants = [2,2,3,3], capacityA = 3, capacityB = 4
输出：2
解释：
- 最初，Alice 的水罐中有 3 单元水，Bob 的水罐中有 4 单元水。
- Alice 给植物 0 浇水，Bob 给植物 3 浇水。
- Alice 和 Bob 现在都只有 1 单元水，并分别需要给植物 1 和植物 2 浇水。
- 由于他们的水量均不足以浇水，所以他们重新灌满水罐再进行浇水。
所以，两人浇灌所有植物过程中重新灌满水罐的次数 = 0 + 1 + 1 + 0 = 2 。
```

**示例 3：**

```
输入：plants = [5], capacityA = 10, capacityB = 8
输出：0
解释：
- 只有一株植物
- Alice 的水罐有 10 单元水，Bob 的水罐有 8 单元水。因此 Alice 的水罐中水更多，她会给这株植物浇水。
所以，两人浇灌所有植物过程中重新灌满水罐的次数 = 0 。
```

**示例 4：**

```
输入：plants = [1,2,4,4,5], capacityA = 6, capacityB = 5
输出：2
解释：
- 最初，Alice 的水罐中有 6 单元水，Bob 的水罐中有 5 单元水。
- Alice 给植物 0 浇水，Bob 给植物 4 浇水。
- Alice 和 Bob 现在分别剩下 5 单元和 0 单元水。
- Alice 有足够的水给植物 1 ，所以她直接浇水。Bob 的水不够给植物 3 ，所以他先重新装满水，再浇水。
- Alice 和 Bob 现在分别剩下 3 单元和 1 单元水。
- 由于 Alice 的水更多，所以由她给植物 2 浇水。然而，她水罐里的水不够给植物 2 ，所以她先重新装满水，再浇水。 
所以，两人浇灌所有植物过程中重新灌满水罐的次数 = 0 + 0 + 1 + 1 + 0 = 2 。
```

**示例 5：**

```
输入：plants = [2,2,5,2,2], capacityA = 5, capacityB = 5
输出：1
解释：
Alice 和 Bob 都会到达中间的植物，并且此时他俩剩下的水量相同，所以 Alice 会给这株植物浇水。
由于她到达时只剩下 1 单元水，所以需要重新灌满水罐。
这是唯一一次需要重新灌满水罐的情况。所以，两人浇灌所有植物过程中重新灌满水罐的次数 = 1 。
```

**提示：**

-   `n == plants.length`
-   `1 <= n <= 105`
-   `1 <= plants[i] <= 106`
-   `max(plants[i]) <= capacityA, capacityB <= 109`

**提交：**

>   纯傻逼模拟题

```rust
impl Solution {
    pub fn minimum_refill(plants: Vec<i32>, ca: i32, cb: i32) -> i32 {
        let mut refiil = 0;
        let (mut resta, mut restb) = (ca, cb);
        let (mut a, mut b) = (0, plants.len() - 1);
        while a <= b {
            if a != b {
                if resta < plants[a] {
                    refiil += 1;
                    resta = ca;
                }
                if restb < plants[b] {
                    refiil += 1;
                    restb = cb;
                }
                resta -= plants[a];
                restb -= plants[b];
            }
            else {
                if resta < restb {
                    if restb < plants[b] {
                        refiil += 1;
                    }
                } else {
                    if resta < plants[a] {
                        refiil += 1;
                    }
                }
                break;	// 不加可能会导致下面出现回绕。考虑只有一个元素的时候。
            }
            a += 1;
            b -= 1;
        }
        refiil
    }
}
```



## T4 5955. 摘水果

-   **User Accepted:** 640
-   **User Tried:** 1181
-   **Total Accepted: **702
-   **Total Submissions: **3125
-   **Difficulty:** **Hard**

在一个无限的 x 坐标轴上，有许多水果分布在其中某些位置。给你一个二维整数数组 `fruits` ，其中 `fruits[i] = [positioni, amounti]` 表示共有 `amounti` 个水果放置在 `positioni` 上。`fruits` 已经按 `positioni` **升序排列** ，每个 `positioni` **互不相同** 。

另给你两个整数 `startPos` 和 `k` 。最初，你位于 `startPos` 。从任何位置，你可以选择 **向左或者向右** 走。在 x 轴上每移动 **一个单位** ，就记作 **一步** 。你总共可以走 **最多** `k` 步。你每达到一个位置，都会摘掉全部的水果，水果也将从该位置消失（不会再生）。

返回你可以摘到水果的 **最大总数** 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/11/21/1.png)

```
输入：fruits = [[2,8],[6,3],[8,6]], startPos = 5, k = 4
输出：9
解释：
最佳路线为：
- 向右移动到位置 6 ，摘到 3 个水果
- 向右移动到位置 8 ，摘到 6 个水果
移动 3 步，共摘到 3 + 6 = 9 个水果
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/11/21/2.png)

```
输入：fruits = [[0,9],[4,1],[5,7],[6,2],[7,4],[10,9]], startPos = 5, k = 4
输出：14
解释：
可以移动最多 k = 4 步，所以无法到达位置 0 和位置 10 。
最佳路线为：
- 在初始位置 5 ，摘到 7 个水果
- 向左移动到位置 4 ，摘到 1 个水果
- 向右移动到位置 6 ，摘到 2 个水果
- 向右移动到位置 7 ，摘到 4 个水果
移动 1 + 3 = 4 步，共摘到 7 + 1 + 2 + 4 = 14 个水果
```

**示例 3：**

![img](https://assets.leetcode.com/uploads/2021/11/21/3.png)

```
输入：fruits = [[0,3],[6,4],[8,5]], startPos = 3, k = 2
输出：0
解释：
最多可以移动 k = 2 步，无法到达任一有水果的地方
```

 

**提示：**

-   `1 <= fruits.length <= 105`
-   `fruits[i].length == 2`
-   `0 <= startPos, positioni <= 2 * 105`
-   对于任意 `i > 0` ，`positioni-1 < positioni` 均成立（下标从 **0** 开始计数）
-   `1 <= amounti <= 104`
-   `0 <= k <= 2 * 105`

**题解：**

方法一：双指针
显然在最优的方案中，最多掉头一次。那么就有两种情况：

先往左，如果还有步数，再往右
先往右，如果还有步数，再往左
第二种情况就相当于将所有坐标取为镜像之后再按照第一种情况处理，所以我们这里只需要考虑第一种情况。

我们首先二分找到startPos左侧（含本身）最靠近的一个位置，设为p。因为我们规定了是先往左再往右，所以我们的行程所覆盖的区间的左端点不会超过p。

接下来，我们从0开始枚举左端点的位置。

在确定了最左边的位置之后，我们需要确定最后的落脚点在哪里。显然，在最左边的位置右移时，最后的落脚点必然右移，所以可以使用双指针的方法，一个指针代表左端点，另一个指针代表最后的落脚点。

比较最后的落脚点和p的大小，就可以确定覆盖区间的右端点。这时我们需要求左端点到右端点之间的水果总和，显然，这可以通过前缀和解决。但更进一步，可以发现，如果在双指针移动过程中动态维护当前指针区间内的元素和，就不需要建立前缀和数组，从而将额外空间复杂度优化到O(1)。

处理完向左再向右的情况后，按照前面说的，将所有坐标（包括fruitsfruits中的坐标和startPos都取为相反数，也即关于原点的镜像）再计算一次，就可以得到最后的结果。

时间复杂度O(N)
空间复杂度O(1)

```c++
const int INF = 1e9;

class Solution {
    int solve(vector<vector<int>>& fruits, int startPos, int k) {
        int n = fruits.size();
        int p = upper_bound(fruits.begin(), fruits.end(), vector<int>{startPos, INF}) - fruits.begin() - 1;
        int ans = 0, r = 0, sum = 0;
        for (int i = 0; i <= p; ++i)
            sum += fruits[i][1];
		for (int l = 0; l <= p; ++l) {
        if (l >= 1)
            sum -= fruits[l - 1][1];
        if (startPos - fruits[l][0] > k)
            continue;
        r = max(r, l);
        while (r + 1 < n && startPos - fruits[l][0] + fruits[r + 1][0] - fruits[l][0] <= k) {
            r++;
            if (r > p)
                sum += fruits[r][1];
        }
        ans = max(ans, sum);
    }
    
    return ans;
}
public:
    int maxTotalFruits(vector<vector<int>>& fruits, int startPos, int k) {
        int ans = solve(fruits, startPos, k);
        for (auto &fruit : fruits) fruit[0] = -fruit[0];
        reverse(fruits.begin(), fruits.end());
        ans = max(ans, solve(fruits, -startPos, k));
        return ans;
    }
};

/*
作者：吴自华
链接：https://leetcode-cn.com/circle/discuss/MRfrww/view/EVQW6k/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
*/
```
