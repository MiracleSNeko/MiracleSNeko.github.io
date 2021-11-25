---
title: LeetCode 周赛 262
date: 2021-10-10 13:28:07
tags: LeetCode 周赛总结
---

---------

# LeetCode 周赛 262

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-262/problems/two-out-of-three/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-262/problems/minimum-operations-to-make-a-uni-value-grid/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-262/problems/stock-price-fluctuation/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-262/problems/partition-array-into-two-arrays-to-minimize-sum-difference/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 2378 / 4260 | MiracleSNeko | 3    | 0:21:15  | 0:16:15 1                                                    |                                                              |                                                              |                                                              |

>大写的寄

## T1 5894. 至少在两个数组中出现的值

- **通过的用户数**3063
- **尝试过的用户数**3147
- **用户总通过次数**3114
- **用户总提交次数**4627
- **题目难度** **Easy**

给你三个整数数组 `nums1`、`nums2` 和 `nums3` ，请你构造并返回一个 **不同** 数组，且由 **至少** 在 **两个** 数组中出现的所有值组成*。*数组中的元素可以按 **任意** 顺序排列。

**示例 1：**

```
输入：nums1 = [1,1,3,2], nums2 = [2,3], nums3 = [3]
输出：[3,2]
解释：至少在两个数组中出现的所有值为：
- 3 ，在全部三个数组中都出现过。
- 2 ，在数组 nums1 和 nums2 中出现过。
```

**示例 2：**

```
输入：nums1 = [3,1], nums2 = [2,3], nums3 = [1,2]
输出：[2,3,1]
解释：至少在两个数组中出现的所有值为：
- 2 ，在数组 nums2 和 nums3 中出现过。
- 3 ，在数组 nums1 和 nums2 中出现过。
- 1 ，在数组 nums1 和 nums3 中出现过。
```

**示例 3：**

```
输入：nums1 = [1,2,2], nums2 = [4,3,3], nums3 = [5]
输出：[]
解释：不存在至少在两个数组中出现的值。
```

**提示：**

- `1 <= nums1.length, nums2.length, nums3.length <= 100`
- `1 <= nums1[i], nums2[j], nums3[k] <= 100`

**我的提交：**

```rust
/// Dummy Luogu/LeetCode Playground
use std::cell::RefCell;
use std::cmp::{max, min, Reverse};
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::rc::Rc;

macro_rules! init_cin {
    () => {{
        (io::stdin(), String::new())
    }};
}
macro_rules! scanf {
    ($buf: expr, $div: expr, $($x:ty), +) => {{
        let mut iter = $buf.split($div);
        ($(iter.next().and_then(|token| token.parse::<$x>().ok()), ) *)
    }};
}
macro_rules! getline {
    ($cin: expr, $buf: expr) => {{
        $buf.clear();
        $cin.read_line(&mut $buf)?;
    }};
}
macro_rules! map_or_insert {
    ($map: expr, $key: expr, $fn: expr, $val: expr) => {{
        match $map.get_mut(&$key) {
            Some(mut v) => {
                $fn(v);
            }
            None => {
                $map.insert($key, $val);
            }
        }
    }};
}

impl Solution {
    pub fn two_out_of_three(nums1: Vec<i32>, nums2: Vec<i32>, nums3: Vec<i32>) -> Vec<i32> {
        let mut mp = HashMap::new();
        for i in nums1 {
            map_or_insert!(mp, i, |x: &mut i32| *x |= 1, 1);
        }
        for i in nums2 {
            map_or_insert!(mp, i, |x: &mut i32| *x |= 2, 2);
        }
        for i in nums3 {
            map_or_insert!(mp, i, |x: &mut i32| *x |= 4, 4);
        }
        mp.into_iter()
            .filter(|&(_, v)| v == 3 || v == 5 || v == 6 || v == 7)
            .map(|(k, _)| k)
            .collect()
    }
}
```



## T2 5895. 获取单值网格的最小操作数

- **通过的用户数**1442
- **尝试过的用户数**2095
- **用户总通过次数**1486
- **用户总提交次数**5233
- **题目难度** **Medium**

给你一个大小为 `m x n` 的二维整数网格 `grid` 和一个整数 `x` 。每一次操作，你可以对 `grid` 中的任一元素 **加** `x` 或 **减** `x` 。

**单值网格** 是全部元素都相等的网格。

返回使网格化为单值网格所需的 **最小** 操作数。如果不能，返回 `-1` 。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/09/21/gridtxt.png)

```
输入：grid = [[2,4],[6,8]], x = 2
输出：4
解释：可以执行下述操作使所有元素都等于 4 ： 
- 2 加 x 一次。
- 6 减 x 一次。
- 8 减 x 两次。
共计 4 次操作。
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/09/21/gridtxt-1.png)

```
输入：grid = [[1,5],[2,3]], x = 1
输出：5
解释：可以使所有元素都等于 3 。
```

**示例 3：**

![img](https://assets.leetcode.com/uploads/2021/09/21/gridtxt-2.png)

```
输入：grid = [[1,2],[3,4]], x = 2
输出：-1
解释：无法使所有元素相等。
```

**提示：**

- `m == grid.length`
- `n == grid[i].length`
- `1 <= m, n <= 105`
- `1 <= m * n <= 105`
- `1 <= x, grid[i][j] <= 104`

**代码：**

> [某道题](https://leetcode-cn.com/problems/minimum-moves-to-equal-array-elements-ii/)的原题改编。找到中位数即可。至于为什么找中位数：
>
> 因为可以证明 $f(x) = |x - x_1| + ... + |x - x_n|$ 在中位数处取得最小值。

```rust
impl Solution {
    pub fn min_operations(grid: Vec<Vec<i32>>, x: i32) -> i32 {
        let mut flat = vec![];
        grid.iter()
            .for_each(|v| v.iter().for_each(|&i| flat.push(i)));
        flat.sort();
        let mid = flat[flat.len() / 2];
        let ans = 0;
        for i in flat {
            let diff = (i - mid).abs();
            if diff % x != 0 {
                return -1;
            }
            ans += diff / x;
        }
        ans
    }
}
```



## T3 5896. 股票价格波动

- **通过的用户数**853
- **尝试过的用户数**1843
- **用户总通过次数**878
- **用户总提交次数**4663
- **题目难度** **Medium**

给你一支股票价格的数据流。数据流中每一条记录包含一个 **时间戳** 和该时间点股票对应的 **价格** 。

不巧的是，由于股票市场内在的波动性，股票价格记录可能不是按时间顺序到来的。某些情况下，有的记录可能是错的。如果两个有相同时间戳的记录出现在数据流中，前一条记录视为错误记录，后出现的记录 **更正** 前一条错误的记录。

请你设计一个算法，实现：

- **更新** 股票在某一时间戳的股票价格，如果有之前同一时间戳的价格，这一操作将 **更正** 之前的错误价格。
- 找到当前记录里 **最新股票价格** 。**最新股票价格** 定义为时间戳最晚的股票价格。
- 找到当前记录里股票的 **最高价格** 。
- 找到当前记录里股票的 **最低价格** 。

请你实现 `StockPrice` 类：

- `StockPrice()` 初始化对象，当前无股票价格记录。
- `void update(int timestamp, int price)` 在时间点 `timestamp` 更新股票价格为 `price` 。
- `int current()` 返回股票 **最新价格** 。
- `int maximum()` 返回股票 **最高价格** 。
- `int minimum()` 返回股票 **最低价格** 。

**示例 1：**

```
输入：
["StockPrice", "update", "update", "current", "maximum", "update", "maximum", "update", "minimum"]
[[], [1, 10], [2, 5], [], [], [1, 3], [], [4, 2], []]
输出：
[null, null, null, 5, 10, null, 5, null, 2]

解释：
StockPrice stockPrice = new StockPrice();
stockPrice.update(1, 10); // 时间戳为 [1] ，对应的股票价格为 [10] 。
stockPrice.update(2, 5);  // 时间戳为 [1,2] ，对应的股票价格为 [10,5] 。
stockPrice.current();     // 返回 5 ，最新时间戳为 2 ，对应价格为 5 。
stockPrice.maximum();     // 返回 10 ，最高价格的时间戳为 1 ，价格为 10 。
stockPrice.update(1, 3);  // 之前时间戳为 1 的价格错误，价格更新为 3 。
                          // 时间戳为 [1,2] ，对应股票价格为 [3,5] 。
stockPrice.maximum();     // 返回 5 ，更正后最高价格为 5 。
stockPrice.update(4, 2);  // 时间戳为 [1,2,4] ，对应价格为 [3,5,2] 。
stockPrice.minimum();     // 返回 2 ，最低价格时间戳为 4 ，价格为 2 。
```

**提示：**

- `1 <= timestamp, price <= 109`
- `update`，`current`，`maximum` 和 `minimum` **总** 调用次数不超过 `105` 。
- `current`，`maximum` 和 `minimum` 被调用时，`update` 操作 **至少** 已经被调用过 **一次** 。

**题解：**

> 对 C++ 来说是个简单题，对 Rust 来说，坐牢（

```c++
class StockPrice {
public:
    StockPrice() {
        lastTime = 0;
    }
    
    void update(int timestamp, int price) {
        //更新某一时间戳价格时
        if(f.count(timestamp)){
            int x = p.count(f[timestamp]);
            //erase会删除所有值为f[timestamp]的数据
            p.erase(f[timestamp]);
            //只删去一个,多删的补回来
            for(int i=0; i<x-1; i++) p.insert(f[timestamp]);
        }
        //更新价格
        f[timestamp] = price;
        //记录最新时间戳
        if(timestamp > lastTime) lastTime = timestamp;
        //插入价格
        p.insert(price);
    }
    
    int current() {
        return f[lastTime];
    }
    
    int maximum() {
        return *p.rbegin();
    }
    
    int minimum() {
        return *p.begin();
        
    }
private:.
    //从大到小排序的股票价格
    multiset<int> p;
    //时间戳对应的股票价格
    unordered_map<int,int> f;
    //最新时间戳
    int lastTime;
};

/**
 * Your StockPrice object will be instantiated and called as such:
 * StockPrice* obj = new StockPrice();
 * obj->update(timestamp,price);
 * int param_2 = obj->current();
 * int param_3 = obj->maximum();
 * int param_4 = obj->minimum();
 */

作者：foreversun
链接：https://leetcode-cn.com/problems/stock-price-fluctuation/solution/5896-gu-piao-jie-ge-bo-dong-by-foreversu-28c2/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



## T4 5897. 将数组分成两个数组并最小化数组和的差

- **通过的用户数**140
- **尝试过的用户数**653
- **用户总通过次数**160
- **用户总提交次数**1707
- **题目难度** **Hard**

给你一个长度为 `2 * n` 的整数数组。你需要将 `nums` 分成 **两个** 长度为 `n` 的数组，分别求出两个数组的和，并 **最小化** 两个数组和之 **差的绝对值** 。`nums` 中每个元素都需要放入两个数组之一。

请你返回 **最小** 的数组和之差。

**示例 1：**

![example-1](https://assets.leetcode.com/uploads/2021/10/02/ex1.png)

```
输入：nums = [3,9,7,3]
输出：2
解释：最优分组方案是分成 [3,9] 和 [7,3] 。
数组和之差的绝对值为 abs((3 + 9) - (7 + 3)) = 2 。
```

**示例 2：**

```
输入：nums = [-36,36]
输出：72
解释：最优分组方案是分成 [-36] 和 [36] 。
数组和之差的绝对值为 abs((-36) - (36)) = 72 。
```

**示例 3：**

![example-3](https://assets.leetcode.com/uploads/2021/10/02/ex3.png)

```
输入：nums = [2,-1,0,4,-2,-9]
输出：0
解释：最优分组方案是分成 [2,4,-9] 和 [-1,0,-2] 。
数组和之差的绝对值为 abs((2 + 4 + -9) - (-1 + 0 + -2)) = 0 。
```

**提示：**

- `1 <= n <= 15`
- `nums.length == 2 * n`
- `-107 <= nums[i] <= 107`

**代码：**

> 参考 1755. 最接近目标值的子序列和。$2^{30}$ 的大小是 $10^9$ 级别，直接枚举必定超时。
>
> 所以要将这 $2n$ 长度的数组拆分为左右两部分，重点考虑到左右都取了元素的情况

>   详情参考折半搜索相关内容

```java
class Solution {

    public int minimumDifference(int[] nums) {
        // 前 n 个元素元素组合情况存储在left 中, 后 n 个元素组合请情况存储在 right 中
        // Map<元素个数, Set<key个元素的总和>>
        Map<Integer, TreeSet<Integer>> left = new HashMap<>();
        Map<Integer, TreeSet<Integer>> right = new HashMap<>();

        int min = Integer.MAX_VALUE;
        int total = 0;

        int n = nums.length / 2;
        for(int i=0;i < 2 * n;i++){
            total += nums[i];

            if(i < n){
                left.put(i+1, new TreeSet<>());
            }else{
                right.put(i - n + 1, new TreeSet<>());
            }
        }

        dfs(nums, 0, 0, 0, n, left);
        dfs(nums, 0, 0, n, 2*n, right);

        // 情况一, 一部分元素在左侧，一部分元素在右侧
        for(int i=1;i<n;i++){
            TreeSet<Integer> set = left.get(i);
            for(int leftSum : set){
                // 前 i 个元素在  left 中, 后  n - i 个元素在 right 中
                // 最佳情况是分成两侧相等即  total / 2, 寻找最佳组合最近的组合
                Integer rightSum = right.get(n-i).ceiling(total / 2 - leftSum);
                if(null != rightSum){
                    int sum = leftSum + rightSum;
                    min = Math.min(min, Math.abs(sum - (total - sum)));
                }

                rightSum = right.get(n-i).floor(total / 2 - leftSum);
                if(null != rightSum){
                    int sum = leftSum + rightSum;
                    min = Math.min(min, Math.abs(sum - (total - sum)));
                }

                if(min == 0){
                    return 0;
                }
            }
        }

        // 情况二,  所有元素都来源与一侧
        TreeSet<Integer> set = left.get(n);
        for(int sum : set){
            min = Math.min(min, Math.abs(sum - (total - sum)));
        }

        return min;
    }

    /**
     * 递归枚举所有的元素组合,将元素组合情况存 Map<元素个数, Set<key个元素的总和>> 中
     *
     * @param nums
     * @param sum   已选数组和
     * @param count 已选数个数
     * @param idx   当前索引
     * @param limit   索引边界
     * @param visited
     */
    public void dfs(int[] nums, int sum, int count, int idx, int limit, Map<Integer, TreeSet<Integer>> visited){
        if(visited.containsKey(count)){
            visited.get(count).add(sum);
        }

        if(idx >= limit) return ;

        // 选择当前元素
        dfs(nums, sum + nums[idx], count+1, idx+1, limit, visited);

        // 不选当前元素
        dfs(nums, sum, count, idx+1, limit, visited);
    }
}

作者：smqk
链接：https://leetcode-cn.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference/solution/fen-zu-mei-ju-shuang-100-by-smqk-wnpe/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

>   单纯的折半查找，会 TLE 在第 175 个测试点

```rust
use std::collections::HashMap;

macro_rules! lowbit {
    ($x: expr) => {{
        $x & (!$x + 1)
    }};
}

impl Solution {
    pub fn minimum_difference(mut left: Vec<i32>) -> i32 {
        let n = left.len() / 2;
        let sum = left.iter().sum::<i32>();
        let right = left.split_off(n);
        let (left, right) = (Solution::to_count(&left), Solution::to_count(&right));
        left.iter().flat_map(|lhs| right.iter().map(move |rhs| (lhs, rhs)))
            .filter(|(lhs, rhs)| lhs.0 + rhs.0 == n as i32)    
            .fold(i32::MAX, |ans, (lhs, rhs)| {
                lhs.1.iter().flat_map(|l| rhs.1.iter().map(move |r| (l, r)))
                    .fold(ans, |ans, (l, r) | {
                        i32::min(ans, (2 * l + 2 * r - sum).abs())
                    })
            })
    }

    fn to_count(vals: &Vec<i32>) -> HashMap<i32, Vec<i32>> {
        let n = vals.len();
        let mut map = HashMap::new();
        for mask in 0..(1 << n) {
            let key = Solution::count_mask(mask);
            let val = Solution::sum_mask(vals, mask);
            (*map.entry(key).or_insert(vec![])).push(val);
        }
        map
    }

    #[inline]
    fn count_mask(mut mask: usize) -> i32 {
        let mut cnt = 0;
        while mask > 0 {
            cnt += 1;
            mask -= lowbit!(mask);
        }
        cnt
    }

    #[inline]
    fn sum_mask(vals: &Vec<i32>, mask: usize) -> i32 {
        let mut sum = 0;
        for i in 0..vals.len() {
            if mask & (1 << i) != 0 {
                sum += vals[i];
            }
        }
        sum
    }
}
```

>   实际上可以二分降低复杂度。对于匹配的左侧右侧状态，如果要使得 $|lsum + rsum|$ 最小，显然是取两者符号相反，且 $|lsum|$ 和 $|rsum|$ 最接近的一组数。对于给定的 $rsum$ ，在所有 $lsum$ 里找到第一个不小于 $-rsum$ 的 $lsum$，则此时的 $lsum + rsum$ 是当前的最小值。遍历所有 $rsum$ 即可得到答案。

```rust
use std::collections::HashMap;

macro_rules! lowbit {
    ($x: expr) => {{
        $x & (!$x + 1)
    }}
}

impl Solution {
    pub fn minimum_difference(mut left: Vec<i32>) -> i32 {
        let n = left.len() / 2;
        let right = left.split_off(n);
        let (left, right) = (Solution::to_count(&left), Solution::to_count(&right));
        left.iter().flat_map(|lhs| right.iter().map(move |rhs| (lhs, rhs)))
            .filter(|(lhs, rhs)| lhs.0 + rhs.0 == n as i32)    
            .fold(i32::MAX, |ans, (lhs, rhs)| {
                rhs.1.iter().fold(ans, |ans, r| {
                    let pos = lhs.1.binary_search(&-r).unwrap_or_else(|err| err);
                    if pos != lhs.1.len() { i32::min(ans, r + lhs.1[pos]) } else {ans}
                })
            })
    }

    fn to_count(vals: &Vec<i32>) -> HashMap<i32, Vec<i32>> {
        let n = vals.len();
        let mut map = HashMap::new();
        for mask in 0..(1 << n) {
            let key = Solution::count_mask(mask);
            let val = Solution::sum_mask(vals, mask);
            (*map.entry(key).or_insert(vec![])).push(val);
        }
        map.iter_mut().for_each(|(_, v)| v.sort());
        map
    }

    #[inline]
    fn count_mask(mut mask: usize) -> i32 {
        let mut cnt = 0;
        while mask > 0 {
            cnt += 1;
            mask -= lowbit!(mask);
        }
        cnt
    }

    #[inline]
    fn sum_mask(vals: &Vec<i32>, mask: usize) -> i32 {
        let mut sum = 0;
        for i in 0..vals.len() {
            if mask & (1 << i) != 0 {
                sum += vals[i];
            } else {
                sum -= vals[i];
            }
        }
        sum
    }
}
```



不一定对的复杂度分析：

-   不使用二分，直接折半遍历
    -   预处理阶段，左右各自遍历 $2^n$ 个 $mask$ 构造哈希表
    -   遍历左侧的 $n$ 种情况，右侧的情况与之对应。设左侧取 $i$ 个数，右侧取 $n - i$ 个，则需要计算 $C_n^i C_n^{n - i}$ 次
    -   总复杂度 $O(2^{n}) + O(\sum_{i = 0}^n C_n^i C_n^{n - i})$ = $O(2^{n+1}) + O(\frac{4^n\Gamma(n + \frac{1}{2})}{\sqrt{\pi} n!})$ ~~，$\lim_{n \to \infty} \frac{\Gamma(n + \frac{1}{2})}{\sqrt{\pi} n! \sqrt n} = 0.56419$，因此该解法的复杂度比不折半低一个$O(\sqrt n)$？~~
    -   用 Wolfram Alpha 计算 $n = 30$ 时后半部分的求和是 $1e9$ 的 $1e8$ 倍
-   使用二分
    -   预处理阶段，左右各遍历 $2^n$ 个 $mask$ 构造哈希表
    -   遍历哈希表，每一侧取 $i$ 个数时的 $C_n^i$ 种情况需要排序，合计 $2 \sum_{i = 0}^n C_n^i\log{C_n^i}$
    -   遍历左侧的 $n$ 种情况，设左侧取 $i$ 个数，二分查找右侧位置，需要计算 $C_n^i\log{C_n^{n-i}}$ 次
    -   总复杂度 $O(2^{n}) + O(\sum_{i = 0}^n C_n^i\log{C_n^i})$ 
    -   用 Wolfram Alpha 计算 $n = 30$ 时后两者的比值，结果为 $4.15e6$，后半部分是 $1e9$ 的约 28 倍

