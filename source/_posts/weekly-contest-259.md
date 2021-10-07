---
title: LeetCode 周赛 259
date: 2021-09-19 14:55:49
tags: LeetCode 周赛总结
---

---

# LeetCode 周赛 259

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-259/problems/final-value-of-variable-after-performing-operations/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-259/problems/sum-of-beauty-in-the-array/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-259/problems/detect-squares/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-259/problems/longest-subsequence-repeated-k-times/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1091 / 3774 | MiracleSNeko | 7    | 0:15:15  | 0:01:25                                                      | 0:15:15                                                      |                                                              |                                                              |



## T1 5875. 执行操作后的变量值

-   **通过的用户数**2937
-   **尝试过的用户数**2956
-   **用户总通过次数**2960
-   **用户总提交次数**3107
-   **题目难度** **Easy**

存在一种仅支持 4 种操作和 1 个变量 `X` 的编程语言：

-   `++X` 和 `X++` 使变量 `X` 的值 **加** `1`
-   `--X` 和 `X--` 使变量 `X` 的值 **减** `1`

最初，`X` 的值是 `0`

给你一个字符串数组 `operations` ，这是由操作组成的一个列表，返回执行所有操作后， `X` 的 **最终值** 。

**示例 1：**

```
输入：operations = ["--X","X++","X++"]
输出：1
解释：操作按下述步骤执行：
最初，X = 0
--X：X 减 1 ，X =  0 - 1 = -1
X++：X 加 1 ，X = -1 + 1 =  0
X++：X 加 1 ，X =  0 + 1 =  1
```

**示例 2：**

```
输入：operations = ["++X","++X","X++"]
输出：3
解释：操作按下述步骤执行： 
最初，X = 0
++X：X 加 1 ，X = 0 + 1 = 1
++X：X 加 1 ，X = 1 + 1 = 2
X++：X 加 1 ，X = 2 + 1 = 3
```

**示例 3：**

```
输入：operations = ["X++","++X","--X","X--"]
输出：0
解释：操作按下述步骤执行：
最初，X = 0
X++：X 加 1 ，X = 0 + 1 = 1
++X：X 加 1 ，X = 1 + 1 = 2
--X：X 减 1 ，X = 2 - 1 = 1
X--：X 减 1 ，X = 1 - 1 = 0
```

**提示：**

-   `1 <= operations.length <= 100`
-   `operations[i]` 将会是 `"++X"`、`"X++"`、`"--X"` 或 `"X--"`

**我的提交：**

```c++
class Solution
{
public:
    int finalValueAfterOperations(Vec<str> &operations)
    {
        auto ans = 0;
        for(auto op: operations)
        {
            if (op == "++X" || op == "X++") ans++;
            else ans--;
        }
        return ans;
    }
};
```



## T2 5876. 数组美丽值求和

-   **通过的用户数**2050
-   **尝试过的用户数**2749
-   **用户总通过次数**2088
-   **用户总提交次数**6951
-   **题目难度** **Medium**

给你一个下标从 **0** 开始的整数数组 `nums` 。对于每个下标 `i`（`1 <= i <= nums.length - 2`），`nums[i]` 的 **美丽值** 等于：

-   `2`，对于所有 `0 <= j < i` 且 `i < k <= nums.length - 1` ，满足 `nums[j] < nums[i] < nums[k]`
-   `1`，如果满足 `nums[i - 1] < nums[i] < nums[i + 1]` ，且不满足前面的条件
-   `0`，如果上述条件全部不满足

返回符合 `1 <= i <= nums.length - 2` 的所有 `nums[i]` 的 **美丽值的总和** 。

**示例 1：**

```
输入：nums = [1,2,3]
输出：2
解释：对于每个符合范围 1 <= i <= 1 的下标 i :
- nums[1] 的美丽值等于 2
```

**示例 2：**

```
输入：nums = [2,4,6,4]
输出：1
解释：对于每个符合范围 1 <= i <= 2 的下标 i :
- nums[1] 的美丽值等于 1
- nums[2] 的美丽值等于 0
```

**示例 3：**

```
输入：nums = [3,2,1]
输出：0
解释：对于每个符合范围 1 <= i <= 1 的下标 i :
- nums[1] 的美丽值等于 0
```

**提示：**

-   `3 <= nums.length <= 105`
-   `1 <= nums[i] <= 105`

**我的提交：**

>   预处理前后最值

```c++
class Solution
{
public:
    int sumOfBeauties(Veci &nums)
    {
        auto len = nums.size();
        Veci leftmax(len, INT_MIN), rightmin(len, INT_MAX);
        leftmax[1] = nums[0], rightmin[len - 2] = nums[len - 1];
        FORINC(i, 2, len)
        {
            leftmax[i] = std::max(leftmax[i - 1], nums[i - 1]);
            rightmin[len - i - 1] = std::min(rightmin[len - i], nums[len - i]);
        }
        auto ans = 0;
        FORINC(i, 1, len - 1)
        {
            if (leftmax[i] < nums[i] && nums[i] < rightmin[i])
                ans += 2;
            else if (nums[i - 1] < nums[i] && nums[i] < nums[i + 1])
                ans += 1;
            else
                continue;
        }
        return ans;
    }
};
```



## T3 5877. 检测正方形

-   **通过的用户数**952
-   **尝试过的用户数**1535
-   **用户总通过次数**988
-   **用户总提交次数**4100
-   **题目难度** **Medium**

给你一个在 X-Y 平面上的点构成的数据流。设计一个满足下述要求的算法：

-   **添加** 一个在数据流中的新点到某个数据结构中**。**可以添加 **重复** 的点，并会视作不同的点进行处理。
-   给你一个查询点，请你从数据结构中选出三个点，使这三个点和查询点一同构成一个 **面积为正** 的 **轴对齐正方形** ，**统计** 满足该要求的方案数目**。**

**轴对齐正方形** 是一个正方形，除四条边长度相同外，还满足每条边都与 x-轴 或 y-轴 平行或垂直。

实现 `DetectSquares` 类：

-   `DetectSquares()` 使用空数据结构初始化对象
-   `void add(int[] point)` 向数据结构添加一个新的点 `point = [x, y]`
-   `int count(int[] point)` 统计按上述方式与点 `point = [x, y]` 共同构造 **轴对齐正方形** 的方案数。

**示例：**

![img](https://assets.leetcode.com/uploads/2021/09/01/image.png)

```
输入：
["DetectSquares", "add", "add", "add", "count", "count", "add", "count"]
[[], [[3, 10]], [[11, 2]], [[3, 2]], [[11, 10]], [[14, 8]], [[11, 2]], [[11, 10]]]
输出：
[null, null, null, null, 1, 0, null, 2]

解释：
DetectSquares detectSquares = new DetectSquares();
detectSquares.add([3, 10]);
detectSquares.add([11, 2]);
detectSquares.add([3, 2]);
detectSquares.count([11, 10]); // 返回 1 。你可以选择：
                               //   - 第一个，第二个，和第三个点
detectSquares.count([14, 8]);  // 返回 0 。查询点无法与数据结构中的这些点构成正方形。
detectSquares.add([11, 2]);    // 允许添加重复的点。
detectSquares.count([11, 10]); // 返回 2 。你可以选择：
                               //   - 第一个，第二个，和第三个点
                               //   - 第一个，第三个，和第四个点
```

**提示：**

-   `point.length == 2`
-   `0 <= x, y <= 1000`
-   调用 `add` 和 `count` 的 **总次数** 最多为 `5000`

**题解：**

定位正方形时可通过枚举对角线进行，确定能否构成正方形只需要一个点。因为只有 `1000` 个点，所以可以直接开一个静态数组。

>   周赛的时候写了一堆 `std::map` ，没想到静态数组和一点定正方形，跑去算了半天对角线长度，欢声笑语中打出 GG

```c++
class DetectSquares
{
public:
    DetectSquares()
    {
        std::ios::sync_with_stdio(false);
        memset(points, 0, sizeof(int) * 1001 * 1001);
    }
    void add(std::vector<int> point)
    {
        points[point[0]][point[1]] += 1;
    }
    int count(std::vector<int> point)
    {
        int x = point[0], y = point[1], ans = 0;
        for (int ny = 0; ny < 1001; ++ny)
        {
            if (points[x][ny] == 0 || ny == y)
                continue;
            auto d = ny - y;
            if ( x + d >= 0 && x + d < 1001)
            {
                ans += points[x + d][y] * points[x + d][ny] * points[x][ny];
            }
            if ( x - d >= 0 && x - d < 1001)
            {
                ans += points[x - d][y] * points[x - d][ny] * points[x][ny];
            }
        }
        return ans;
    }

private:
    int points[1001][1001];
};
```



## T4 5878. 重复 K 次的最长子序列

-   **通过的用户数**111
-   **尝试过的用户数**186
-   **用户总通过次数**149
-   **用户总提交次数**567
-   **题目难度** **Hard**

给你一个长度为 `n` 的字符串 `s` ，和一个整数 `k` 。请你找出字符串 `s` 中 **重复** `k` 次的 **最长子序列** 。

**子序列** 是由其他字符串删除某些（或不删除）字符派生而来的一个字符串。

如果 `seq * k` 是 `s` 的一个子序列，其中 `seq * k` 表示一个由 `seq` 串联 `k` 次构造的字符串，那么就称 `seq` 是字符串 `s` 中一个 **重复 `k` 次** 的子序列。

-   举个例子，`"bba"` 是字符串 `"bababcba"` 中的一个重复 `2` 次的子序列，因为字符串 `"bbabba"` 是由 `"bba"` 串联 `2` 次构造的，而 `"bbabba"` 是字符串 `"bababcba"` 的一个子序列。

返回字符串 `s` 中 **重复 k 次的最长子序列** 。如果存在多个满足的子序列，则返回 **字典序最大** 的那个。如果不存在这样的子序列，返回一个 **空** 字符串。

**示例 1：**

![example 1](https://assets.leetcode.com/uploads/2021/08/30/longest-subsequence-repeat-k-times.png)

```
输入：s = "letsleetcode", k = 2
输出："let"
解释：存在两个最长子序列重复 2 次：let" 和 "ete" 。
"let" 是其中字典序最大的一个。
```

**示例 2：**

```
输入：s = "bb", k = 2
输出："b"
解释：重复 2 次的最长子序列是 "b" 。
```

**示例 3：**

```
输入：s = "ab", k = 2
输出：""
解释：不存在重复 2 次的最长子序列。返回空字符串。
```

**示例 4：**

```
输入：s = "bbabbabbbbabaababab", k = 3
输出："bbbb"
解释：在 "bbabbabbbbabaababab" 中重复 3 次的最长子序列是 "bbbb" 。
```

**提示：**

-   `n == s.length`
-   `2 <= k <= 2000`
-   `2 <= n < k * 8`
-   `s` 由小写英文字母组成

**题解：**

题目所给数据实际上隐含了一条信息：子序列长度不会超过 8，所以对长度为 1 到 7 的全部子序列依次验证即可。

>   纯暴力，C++ 似乎也可以通过全排列函数求解

```c++
bool check(string &a, string &b, int k){//检测b能否在a中出现k次
    int cnt = 0;
    for(int i = 0, j = 0; i < a.size(); i++){
        if(a[i]==b[j]){
            j++;
            if(j==b.size()){
                j = 0, cnt++;
                if(cnt==k) return true; 
            }
        }
    }
    return false;
}
string longestSubsequenceRepeatedK(string s, int k) {
    //易知：子序列长度最多为7 故而可以暴力找出长度为7以内的序列作为要找的子序列 依次验证即可
    
    //若对于长度为len的符合条件的子序列 则它一定是由长度为len-1的序列通过添加一个字母生成的
    //故而递推生成所有可能的子序列即可 然后验证是否符合条件 加入答案数组 最后返回最大的即可
    vector<string> ans[8];//ans[i]表示长度为i的符合条件的字符串
    ans[0].push_back("");
    int i = 1;
    for( ; i < 8; i++){
        for(auto v : ans[i-1]){
            for(char j = 'a'; j <= 'z'; j++){
                string t = v + j;
                if(check(s,t,k)) ans[i].push_back(t);
            }
        }
        if(ans[i].empty()) break;
    }
    i--;
    sort(ans[i].begin(),ans[i].end());
    return ans[i].back();
}

作者：zeroac
链接：https://leetcode-cn.com/problems/longest-subsequence-repeated-k-times/solution/c-bao-li-mei-ju-chang-du-7yi-nei-xu-lie-febao/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

