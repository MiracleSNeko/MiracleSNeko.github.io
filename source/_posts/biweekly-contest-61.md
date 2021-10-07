---
title: LeetCode 双周赛 61
date: 2021-09-19 14:56:09
tags: LeetCode 周赛总结
---

---

# LeetCode 双周赛 61

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/biweekly-contest-61/problems/count-number-of-pairs-with-absolute-difference-k/) | [题目2 (4)](https://leetcode-cn.com/contest/biweekly-contest-61/problems/find-original-array-from-doubled-array/) | [题目3 (5)](https://leetcode-cn.com/contest/biweekly-contest-61/problems/maximum-earnings-from-taxi/) | [题目4 (6)](https://leetcode-cn.com/contest/biweekly-contest-61/problems/minimum-number-of-operations-to-make-array-continuous/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1103 / 2534 | MiracleSNeko | 7    | 1:33:40  | 0:02:08                                                      | 0:48:40 9                                                    |                                                              |                                                              |

## T1 2006. 差的绝对值为 K 的数对数目

-   **通过的用户数**1773
-   **尝试过的用户数**1793
-   **用户总通过次数**1806
-   **用户总提交次数**1985
-   **题目难度** **Easy**

给你一个整数数组 `nums` 和一个整数 `k` ，请你返回数对 `(i, j)` 的数目，满足 `i < j` 且 `|nums[i] - nums[j]| == k` 。

`|x|` 的值定义为：

-   如果 `x >= 0` ，那么值为 `x` 。
-   如果 `x < 0` ，那么值为 `-x` 。

**示例 1：**

```
输入：nums = [1,2,2,1], k = 1
输出：4
解释：差的绝对值为 1 的数对为：
- [1,2,2,1]
- [1,2,2,1]
- [1,2,2,1]
- [1,2,2,1]
```

**示例 2：**

```
输入：nums = [1,3], k = 3
输出：0
解释：没有任何数对差的绝对值为 3 。
```

**示例 3：**

```
输入：nums = [3,2,1,5,4], k = 2
输出：3
解释：差的绝对值为 2 的数对为：
- [3,2,1,5,4]
- [3,2,1,5,4]
- [3,2,1,5,4]
```

**提示：**

-   `1 <= nums.length <= 200`
-   `1 <= nums[i] <= 100`
-   `1 <= k <= 99`

>   T1 的数据量暴力就完事了

**我的提交：**

```c++
class Solution
{
public:
    int countKDifference(std::vector<int> &nums, int k)
    {
        auto len = nums.size();
        auto ans = 0;
        for (auto i = 0; i < len; ++i)
        {
            auto vi = nums[i];
            for (auto j = i; j < len; ++j)
            {
                auto vj = nums[j];
                if (abs(vi - vj) == k) ans++;
            }
        }
        return ans;
    }
};
```



## T2 2007. 从双倍数组中还原原数组

-   **通过的用户数**1149
-   **尝试过的用户数**1598
-   **用户总通过次数**1187
-   **用户总提交次数**5278
-   **题目难度** **Medium**

一个整数数组 `original` 可以转变成一个 **双倍** 数组 `changed` ，转变方式为将 `original` 中每个元素 **值乘以 2** 加入数组中，然后将所有元素 **随机打乱** 。

给你一个数组 `changed` ，如果 `change` 是 **双倍** 数组，那么请你返回 `original`数组，否则请返回空数组。`original` 的元素可以以 **任意** 顺序返回。

**示例 1：**

```
输入：changed = [1,3,4,2,6,8]
输出：[1,3,4]
解释：一个可能的 original 数组为 [1,3,4] :
- 将 1 乘以 2 ，得到 1 * 2 = 2 。
- 将 3 乘以 2 ，得到 3 * 2 = 6 。
- 将 4 乘以 2 ，得到 4 * 2 = 8 。
其他可能的原数组方案为 [4,3,1] 或者 [3,1,4] 。
```

**示例 2：**

```
输入：changed = [6,3,0,1]
输出：[]
解释：changed 不是一个双倍数组。
```

**示例 3：**

```
输入：changed = [1]
输出：[]
解释：changed 不是一个双倍数组。
```

**提示：**

-   `1 <= changed.length <= 105`
-   `0 <= changed[i] <= 105`

**我的提交：**

>   写的时候用数组折腾来折腾去 WA 的死去活来 （9次啊9次），后来发现不如 `std::map` 减减乐

```c++
// 这个破代码还可以改很多地方
// 但是如果是用 Rust 写怕不是就炸了，一个 map 在做迭代器的时候还要被改
class Solution
{
public:
    Veci findOriginalArray(Veci &changed)
    {
        Veci ans;
        std::sort(ALL(changed));
        auto st = UpperBS(changed, 0);
        if ((st & 1) != 0)
            return Veci();
        for (auto i = 0; i < st; i += 2)
        {
            ans.push_back(0);
        }
        changed = Veci(changed.begin() + st, changed.end());
        auto len = changed.size();
        if ((len & 1) != 0)
            return Veci();
        Veci odd;
        std::map<i32, i32> even;
        for (auto &val : changed)
        {
            if ((val & 1) == 0)
            {
                if (even.count(val) == 0)
                {
                    even[val] = 1;
                }
                else
                {
                    even[val] += 1;
                }
            }
            else
            {
                odd.push_back(val);
            }
        }
        for (auto ii : odd)
        {
            if (even.count(2 * ii) == 0 || even[2 * ii] == 0)
                return Veci();
            ans.push_back(ii);
            even[2 * ii] -= 1;
        }
        for(auto [k, v]: even)
        {
            if (v == 0) continue;
            if (even.count(2 * k) == 0) return Veci();
            while(v--)
            {
                ans.push_back(k);
                even[2 * k] -= 1;
                if (even[2 * k] < 0) return Veci();
            }
        }
        return ans;
    }
};
```

**题解：**

小的值优先匹配小的，所以先用sort进行排序
然后可以将暂时没匹配到的数字存放至队列中，使用队列也是为了优先匹配小的，小的先进，小的先出。
每次等到匹配到的时候，就将队列中的数取出。最后判断队列是否为空。

```c++
class Solution {
public:
    vector<int> findOriginalArray(vector<int>& changed) {
        sort(changed.begin(),changed.end());
        queue<int> q;
        vector<int> res,empty;
        int n = changed.size();
        if(n%2)return empty;
        for(int i=0;i<n;i++){
            if(q.empty())
                q.push(changed[i]);
            else{
                if(q.front()*2 == changed[i]){
                    res.push_back(q.front());
                    q.pop();
                }
                else
                    q.push(changed[i]);
            }
        }
        if(!q.empty())
            return empty;
        return res;
    }
};

作者：zhu-146
链接：https://leetcode-cn.com/problems/find-original-array-from-doubled-array/solution/pai-xu-dui-lie-by-zhu-146-a5bo/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



## T3 2008. 出租车的最大盈利

-   **通过的用户数**519
-   **尝试过的用户数**832
-   **用户总通过次数**543
-   **用户总提交次数**1850
-   **题目难度** **Medium**

你驾驶出租车行驶在一条有 `n` 个地点的路上。这 `n` 个地点从近到远编号为 `1` 到 `n` ，你想要从 `1` 开到 `n` ，通过接乘客订单盈利。你只能沿着编号递增的方向前进，不能改变方向。

乘客信息用一个下标从 **0** 开始的二维数组 `rides` 表示，其中 `rides[i] = [starti, endi, tipi]` 表示第 `i` 位乘客需要从地点 `starti` 前往 `endi` ，愿意支付 `tipi` 元的小费。

**每一位** 你选择接单的乘客 `i` ，你可以 **盈利** `endi - starti + tipi` 元。你同时 **最多** 只能接一个订单。

给你 `n` 和 `rides` ，请你返回在最优接单方案下，你能盈利 **最多** 多少元。

**注意：**你可以在一个地点放下一位乘客，并在同一个地点接上另一位乘客。

**示例 1：**

```
输入：n = 5, rides = [[2,5,4],[1,5,1]]
输出：7
解释：我们可以接乘客 0 的订单，获得 5 - 2 + 4 = 7 元。
```

**示例 2：**

```
输入：n = 20, rides = [[1,6,1],[3,10,2],[10,12,3],[11,12,2],[12,15,2],[13,18,1]]
输出：20
解释：我们可以接以下乘客的订单：
- 将乘客 1 从地点 3 送往地点 10 ，获得 10 - 3 + 2 = 9 元。
- 将乘客 2 从地点 10 送往地点 12 ，获得 12 - 10 + 3 = 5 元。
- 将乘客 5 从地点 13 送往地点 18 ，获得 18 - 13 + 1 = 6 元。
我们总共获得 9 + 5 + 6 = 20 元。
```

**提示：**

-   `1 <= n <= 105`
-   `1 <= rides.length <= 3 * 104`
-   `rides[i].length == 3`
-   `1 <= start_i < end_i <= n`
-   `1 <= tip_i <= 105`

>   我也不知道我是怎么把这么明显的 DP 看成图的，可能是 T2 WA 麻了

**题解：**

定义 f[i] 表示行驶到 i 时的最大盈利。考虑状态转移，一方面，我们可以不接终点为 i 的乘客，这样有 f[i]=f[i-1]；另一方面，我们可以接所有终点为 i 的乘客中收益最大的，这样有 f[i] = \max (f[start]+end-start+tip)  ，二者取最大值。

最终答案为 f[n]。

```c++
class Solution
{
public:
    i64 maxTaxiEarnings(i32 n, VecVec<i32> &rides)
    {
        auto dp = Vecl(n+1, 0);
        auto prof = VecVec<std::tuple<i64, i64>>(n+1);
        // 记录每个 end 对应的 start 和 tip
        for(auto&& r: rides)
        {
            auto start = r[0], end = r[1], tip = r[2];
            prof[end].push_back(TUPLE(start, tip));
        }
        FORINC(ed, 1, n+1)
        {
            dp[ed] = dp[ed - 1];
            for(auto&& [st, tip]: prof[ed])
            {
                dp[ed] = std::max(dp[ed], dp[st] + ed - st + tip);
            }
        }
        return dp[n];
    }
};
```



## T4 2009. 使数组连续的最少操作数

-   **通过的用户数**353
-   **尝试过的用户数**545
-   **用户总通过次数**386
-   **用户总提交次数**1243
-   **题目难度** **Hard**

给你一个整数数组 `nums` 。每一次操作中，你可以将 `nums` 中 **任意** 一个元素替换成 **任意** 整数。

如果 `nums` 满足以下条件，那么它是 **连续的** ：

-   `nums` 中所有元素都是 **互不相同** 的。
-   `nums` 中 **最大** 元素与 **最小** 元素的差等于 `nums.length - 1` 。

比方说，`nums = [4, 2, 5, 3]` 是 **连续的** ，但是 `nums = [1, 2, 3, 5, 6]` **不是连续的** 。

请你返回使 `nums` **连续** 的 **最少** 操作次数。

**示例 1：**

```
输入：nums = [4,2,5,3]
输出：0
解释：nums 已经是连续的了。
```

**示例 2：**

```
输入：nums = [1,2,3,5,6]
输出：1
解释：一个可能的解是将最后一个元素变为 4 。
结果数组为 [1,2,3,5,4] ，是连续数组。
```

**示例 3：**

```
输入：nums = [1,10,100,1000]
输出：3
解释：一个可能的解是：
- 将第二个元素变为 2 。
- 将第三个元素变为 3 。
- 将第四个元素变为 4 。
结果数组为 [1,2,3,4] ，是连续数组。
```

**提示：**

-   `1 <= nums.length <= 105`
-   `1 <= nums[i] <= 109`

