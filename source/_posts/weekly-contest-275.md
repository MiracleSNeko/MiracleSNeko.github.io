---
title: LeetCode 周赛 275
date: 2022-01-09 14:12:09
tags: LeetCode 周赛
---

# LeetCode 周赛 275

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-275/problems/check-if-every-row-and-column-contains-all-numbers/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-275/problems/minimum-swaps-to-group-all-1s-together-ii/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-275/problems/count-words-obtained-after-adding-a-letter/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-275/problems/earliest-possible-day-of-full-bloom/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 3022 / 4786 | MiracleSNeko | 3    | 0:29:35  | 0:14:35 3                                                    |                                                              |                                                              |                                                              |

>   报名了，一题都不会，技不如人有什么好说的。要知耻！要知耻！

## T2 5977. 最少交换次数来组合所有的 1 II

-   **User Accepted:**1572
-   **User Tried:**2239
-   **Total Accepted:**1611
-   **Total Submissions:**4569
-   **Difficulty:** **Medium**

**交换** 定义为选中一个数组中的两个 **互不相同** 的位置并交换二者的值。

**环形** 数组是一个数组，可以认为 **第一个** 元素和 **最后一个** 元素 **相邻** 。

给你一个 **二进制环形** 数组 `nums` ，返回在 **任意位置** 将数组中的所有 `1` 聚集在一起需要的最少交换次数。

 **示例 1：**

```
输入：nums = [0,1,0,1,1,0,0]
输出：1
解释：这里列出一些能够将所有 1 聚集在一起的方案：
[0,0,1,1,1,0,0] 交换 1 次。
[0,1,1,1,0,0,0] 交换 1 次。
[1,1,0,0,0,0,1] 交换 2 次（利用数组的环形特性）。
无法在交换 0 次的情况下将数组中的所有 1 聚集在一起。
因此，需要的最少交换次数为 1 。
```

**示例 2：**

```
输入：nums = [0,1,1,1,0,0,1,1,0]
输出：2
解释：这里列出一些能够将所有 1 聚集在一起的方案：
[1,1,1,0,0,0,0,1,1] 交换 2 次（利用数组的环形特性）。
[1,1,1,1,1,0,0,0,0] 交换 2 次。
无法在交换 0 次或 1 次的情况下将数组中的所有 1 聚集在一起。
因此，需要的最少交换次数为 2 。
```

**示例 3：**

```
输入：nums = [1,1,0,0,1]
输出：0
解释：得益于数组的环形特性，所有的 1 已经聚集在一起。
因此，需要的最少交换次数为 0 。
```

**提示：**

-   `1 <= nums.length <= 105`
-   `nums[i]` 为 `0` 或者 `1`

**题解：**

>   滑动窗口。窗口大小显然应该是 1 的个数，求所有窗口中最大的 1 个数即可

```rust
impl Solution {
    pub fn min_swaps(nums: Vec<i32>) -> i32 {
        let total_ones = nums.iter().sum::<i32>();
        let mut curr_ones = nums.iter().take(total_ones as usize).sum::<i32>();
        let mut max_ones = curr_ones;
        (1..nums.len()).for_each(|i| {
            curr_ones += nums[(i - 1 + total_ones as usize) % nums.len()] - nums[i - 1];
            max_ones = max_ones.max(curr_ones);
        });
        total_ones - max_ones
    }
}
```

## T3 5978. 统计追加字母可以获得的单词数

-   **User Accepted:**1157
-   **User Tried:**2261
-   **Total Accepted:**1213
-   **Total Submissions:**6189
-   **Difficulty:** **Medium**

给你两个下标从 **0** 开始的字符串数组 `startWords` 和 `targetWords` 。每个字符串都仅由 **小写英文字母** 组成。

对于 `targetWords` 中的每个字符串，检查是否能够从 `startWords` 中选出一个字符串，执行一次 **转换操作** ，得到的结果与当前 `targetWords` 字符串相等。

**转换操作** 如下面两步所述：

1.  追加任何**不存在**于当前字符串的任一小写字母到当前字符串的末尾。

    -   例如，如果字符串为 `"abc"` ，那么字母 `'d'`、`'e'` 或 `'y'` 都可以加到该字符串末尾，但 `'a'` 就不行。如果追加的是 `'d'` ，那么结果字符串为 `"abcd"` 。

2.  **重排**新字符串中的字母，可以按**任意**顺序重新排布字母。

    -   例如，`"abcd"` 可以重排为 `"acbd"`、`"bacd"`、`"cbda"`，以此类推。注意，它也可以重排为 `"abcd"` 自身。

找出 `targetWords` 中有多少字符串能够由 `startWords` 中的 **任一** 字符串执行上述转换操作获得。返回 `targetWords` 中这类 **字符串的数目** 。

**注意：**你仅能验证 `targetWords` 中的字符串是否可以由 `startWords` 中的某个字符串经执行操作获得。`startWords` 中的字符串在这一过程中 **不** 发生实际变更。

**示例 1：**

```
输入：startWords = ["ant","act","tack"], targetWords = ["tack","act","acti"]
输出：2
解释：
- 为了形成 targetWords[0] = "tack" ，可以选用 startWords[1] = "act" ，追加字母 'k' ，并重排 "actk" 为 "tack" 。
- startWords 中不存在可以用于获得 targetWords[1] = "act" 的字符串。
  注意 "act" 确实存在于 startWords ，但是 必须 在重排前给这个字符串追加一个字母。
- 为了形成 targetWords[2] = "acti" ，可以选用 startWords[1] = "act" ，追加字母 'i' ，并重排 "acti" 为 "acti" 自身。
```

**示例 2：**

```
输入：startWords = ["ab","a"], targetWords = ["abc","abcd"]
输出：1
解释：
- 为了形成 targetWords[0] = "abc" ，可以选用 startWords[0] = "ab" ，追加字母 'c' ，并重排为 "abc" 。
- startWords 中不存在可以用于获得 targetWords[1] = "abcd" 的字符串。
```

**提示：**

-   `1 <= startWords.length, targetWords.length <= 5 * 104`
-   `1 <= startWords[i].length, targetWords[j].length <= 26`
-   `startWords` 和 `targetWords` 中的每个字符串都仅由小写英文字母组成
-   在 `startWords` 或 `targetWords` 的任一字符串中，每个字母至多出现一次

**题解：**

>   审题发现每个字母最多出现一次，并且 targetWord 比 startWord 多且仅多一个字母，所以将 startWord 映射成整数存个 HashSet ，并且对每个 targetWord 尝试删掉一个字母，查找是否在 Set 里

```rust
use std::collections::HashSet;

macro_rules! index {
    ($ch: expr) => {{
        ($ch - b'a') as usize
    }};
}

impl Solution {
    pub fn word_count(start_words: Vec<String>, target_words: Vec<String>) -> i32 {
        let set = start_words.iter().map(|s| Solution::word_to_mask(s)).collect::<HashSet<_>>();
        target_words.iter().map(|s| Solution::word_to_mask(s))
            .filter(|mask| {
                (0..26).any(|i| mask & (1 << i) != 0 && set.contains(&(mask ^ (1 << i))))
            })
            .count() as i32
    }

    pub fn word_to_mask(word: &String) -> i32 {
        word.bytes().fold(0, |mask, ch| mask | (1 << index!(ch)))
    }
}
```



## T4 5979. 全部开花的最早一天

-   **User Accepted:**709
-   **User Tried:**937
-   **Total Accepted:**774
-   **Total Submissions:**1531
-   **Difficulty:** **Hard**

你有 `n` 枚花的种子。每枚种子必须先种下，才能开始生长、开花。播种需要时间，种子的生长也是如此。给你两个下标从 **0** 开始的整数数组 `plantTime` 和 `growTime` ，每个数组的长度都是 `n` ：

-   `plantTime[i]` 是 **播种** 第 `i` 枚种子所需的 **完整天数** 。每天，你只能为播种某一枚种子而劳作。**无须** 连续几天都在种同一枚种子，但是种子播种必须在你工作的天数达到 `plantTime[i]` 之后才算完成。
-   `growTime[i]` 是第 `i` 枚种子完全种下后生长所需的 **完整天数** 。在它生长的最后一天 **之后** ，将会开花并且永远 **绽放** 。

从第 `0` 开始，你可以按 **任意** 顺序播种种子。

返回所有种子都开花的 **最早** 一天是第几天。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/12/21/1.png)

```
输入：plantTime = [1,4,3], growTime = [2,3,1]
输出：9
解释：灰色的花盆表示播种的日子，彩色的花盆表示生长的日子，花朵表示开花的日子。
一种最优方案是：
第 0 天，播种第 0 枚种子，种子生长 2 整天。并在第 3 天开花。
第 1、2、3、4 天，播种第 1 枚种子。种子生长 3 整天，并在第 8 天开花。
第 5、6、7 天，播种第 2 枚种子。种子生长 1 整天，并在第 9 天开花。
因此，在第 9 天，所有种子都开花。 
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/12/21/2.png)

```
输入：plantTime = [1,2,3,2], growTime = [2,1,2,1]
输出：9
解释：灰色的花盆表示播种的日子，彩色的花盆表示生长的日子，花朵表示开花的日子。 
一种最优方案是：
第 1 天，播种第 0 枚种子，种子生长 2 整天。并在第 4 天开花。
第 0、3 天，播种第 1 枚种子。种子生长 1 整天，并在第 5 天开花。
第 2、4、5 天，播种第 2 枚种子。种子生长 2 整天，并在第 8 天开花。
第 6、7 天，播种第 3 枚种子。种子生长 1 整天，并在第 9 天开花。
因此，在第 9 天，所有种子都开花。 
```

**示例 3：**

```
输入：plantTime = [1], growTime = [1]
输出：2
解释：第 0 天，播种第 0 枚种子。种子需要生长 1 整天，然后在第 2 天开花。
因此，在第 2 天，所有种子都开花。
```

**提示：**

-   `n == plantTime.length == growTime.length`
-   `1 <= n <= 105`
-   `1 <= plantTime[i], growTime[i] <= 104`

**题解：**

### 方法一：贪心

先种生长时间长的，再种生长时间短的。

>   为什么这一贪心是正确的？

对于任意两盆花，假设第一盆的种植和生长时间为 (a, b)(*a*,*b*)，第二盆的种植和生长时间为 (c,d)(*c*,*d*)，且有 b>d*b*>*d*。

暂时不考虑交错种植的情况。

-   先种一，再种二：总时间为 \max(a+b, a+c+d)
-   先种二，再种一：总时间为 \max(c+d,c+a+b)。因为 c+a+b>c+a+d>c+d，所以结果就为 c+a+b。

二者对比，显然有 c+a+b>c+a+d 且 c+a+b>a+b。所以应该先种一。

下面考虑交错种植。我们可以发现，无论是几盆花发生交错，交错种植并不能让种植结束时间提前。所以在存在交错种植的情况下，我们总是可以按照种植结束时间的顺序，以无交错的方式进行种植，此时所有花的种植结束时间均不会比原来延后。在这样的情况下，结合上面的讨论，就说明我们的贪心策略确实是最优的。

-   时间复杂度\mathcal{O}(N\log N)
-   空间复杂度\mathcal{O}(N)

#### 参考代码（C++）

```c++
class Solution {
public:
    int earliestFullBloom(vector<int>& plantTime, vector<int>& growTime) {
        int n = plantTime.size();
        vector<int> order(n);
        for (int i = 0; i < n; ++i)
            order[i] = i;
        sort(order.begin(), order.end(), [&](int i, int j){
            return growTime[i] > growTime[j]; 
        });
        
        int ans = 0, day = 0;
        for (int i : order) {
            day += plantTime[i];
            ans = max(ans, day + growTime[i]);
        }
        return ans;
    }
};
```

作者：吴自华
链接：https://leetcode-cn.com/circle/discuss/9TMMfX/view/9xfeIr/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
