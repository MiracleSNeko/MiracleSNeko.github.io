---
title: LeetCode 双周赛 65
date: 2021-11-14 23:22:25
tags: LeetCode 周赛总结
---

-----------

# LeetCode 双周赛 65

| 排名       | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/biweekly-contest-65/problems/check-whether-two-strings-are-almost-equivalent/) | [题目2 (4)](https://leetcode-cn.com/contest/biweekly-contest-65/problems/walking-robot-simulation-ii/) | [题目3 (5)](https://leetcode-cn.com/contest/biweekly-contest-65/problems/most-beautiful-item-for-each-query/) | [题目4 (6)](https://leetcode-cn.com/contest/biweekly-contest-65/problems/maximum-number-of-tasks-you-can-assign/) |
| ---------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 634 / 2676 | MiracleSNeko | 8    | 0:50:07  | 0:04:52                                                      |                                                              | 0:45:07 1                                                    |                                                              |

## T1 5910. 检查两个字符串是否几乎相等

-   **通过的用户数**1956
-   **尝试过的用户数**1988
-   **用户总通过次数**1977
-   **用户总提交次数**2707
-   **题目难度** **Easy**

如果两个字符串 `word1` 和 `word2` 中从 `'a'` 到 `'z'` 每一个字母出现频率之差都 **不超过** `3` ，那么我们称这两个字符串 `word1` 和 `word2` **几乎相等** 。

给你两个长度都为 `n` 的字符串 `word1` 和 `word2` ，如果 `word1` 和 `word2` **几乎相等** ，请你返回 `true` ，否则返回 `false` 。

一个字母 `x` 的出现 **频率** 指的是它在字符串中出现的次数。

 

**示例 1：**

```
输入：word1 = "aaaa", word2 = "bccb"
输出：false
解释：字符串 "aaaa" 中有 4 个 'a' ，但是 "bccb" 中有 0 个 'a' 。
两者之差为 4 ，大于上限 3 。
```

**示例 2：**

```
输入：word1 = "abcdeef", word2 = "abaaacc"
输出：true
解释：word1 和 word2 中每个字母出现频率之差至多为 3 ：
- 'a' 在 word1 中出现了 1 次，在 word2 中出现了 4 次，差为 3 。
- 'b' 在 word1 中出现了 1 次，在 word2 中出现了 1 次，差为 0 。
- 'c' 在 word1 中出现了 1 次，在 word2 中出现了 2 次，差为 1 。
- 'd' 在 word1 中出现了 1 次，在 word2 中出现了 0 次，差为 1 。
- 'e' 在 word1 中出现了 2 次，在 word2 中出现了 0 次，差为 2 。
- 'f' 在 word1 中出现了 1 次，在 word2 中出现了 0 次，差为 1 。
```

**示例 3：**

```
输入：word1 = "cccddabba", word2 = "babababab"
输出：true
解释：word1 和 word2 中每个字母出现频率之差至多为 3 ：
- 'a' 在 word1 中出现了 2 次，在 word2 中出现了 4 次，差为 2 。
- 'b' 在 word1 中出现了 2 次，在 word2 中出现了 5 次，差为 3 。
- 'c' 在 word1 中出现了 3 次，在 word2 中出现了 0 次，差为 3 。
- 'd' 在 word1 中出现了 2 次，在 word2 中出现了 0 次，差为 2 。
```

 

**提示：**

-   `n == word1.length == word2.length`
-   `1 <= n <= 100`
-   `word1` 和 `word2` 都只包含小写英文字母。

**提交：**

>   简单模拟

```rust
impl Solution {
    pub fn check_almost_equivalent(word1: String, word2: String) -> bool {
        fn str_to_freq(word: &String) -> Vec<i32> {
            let mut ret = vec![0; 26];
            word.bytes().for_each(|ch| ret[(ch - b'a') as usize] += 1);
            ret
        }
        let vec1 = str_to_freq(&word1);
        let vec2 = str_to_freq(&word2);
        vec1.iter().zip(vec2.iter()).all(|(v1, v2)| (*v1 - *v2).abs() <= 3)
    }
}
```



## T2 5911. 模拟行走机器人 II

-   **通过的用户数**694
-   **尝试过的用户数**1532
-   **用户总通过次数**711
-   **用户总提交次数**6760
-   **题目难度** **Medium**

给你一个在 XY 平面上的 `width x height` 的网格图，**左下角** 的格子为 `(0, 0)` ，**右上角** 的格子为 `(width - 1, height - 1)` 。网格图中相邻格子为四个基本方向之一（`"North"`，`"East"`，`"South"` 和 `"West"`）。一个机器人 **初始** 在格子 `(0, 0)` ，方向为 `"East"` 。

机器人可以根据指令移动指定的 **步数** 。每一步，它可以执行以下操作。

1.  沿着当前方向尝试 **往前一步** 。
2.  如果机器人下一步将到达的格子 **超出了边界** ，机器人会 **逆时针** 转 90 度，然后再尝试往前一步。

如果机器人完成了指令要求的移动步数，它将停止移动并等待下一个指令。

请你实现 `Robot` 类：

-   `Robot(int width, int height)` 初始化一个 `width x height` 的网格图，机器人初始在 `(0, 0)` ，方向朝 `"East"` 。
-   `void move(int num)` 给机器人下达前进 `num` 步的指令。
-   `int[] getPos()` 返回机器人当前所处的格子位置，用一个长度为 2 的数组 `[x, y]` 表示。
-   `String getDir()` 返回当前机器人的朝向，为 `"North"` ，`"East"` ，`"South"` 或者 `"West"` 。

 

**示例 1：**

![example-1](https://assets.leetcode.com/uploads/2021/10/09/example-1.png)

```
输入：
["Robot", "move", "move", "getPos", "getDir", "move", "move", "move", "getPos", "getDir"]
[[6, 3], [2], [2], [], [], [2], [1], [4], [], []]
输出：
[null, null, null, [4, 0], "East", null, null, null, [1, 2], "West"]

解释：
Robot robot = new Robot(6, 3); // 初始化网格图，机器人在 (0, 0) ，朝东。
robot.move(2);  // 机器人朝东移动 2 步，到达 (2, 0) ，并朝东。
robot.move(2);  // 机器人朝东移动 2 步，到达 (4, 0) ，并朝东。
robot.getPos(); // 返回 [4, 0]
robot.getDir(); // 返回 "East"
robot.move(2);  // 朝东移动 1 步到达 (5, 0) ，并朝东。
                // 下一步继续往东移动将出界，所以逆时针转变方向朝北。
                // 然后，往北移动 1 步到达 (5, 1) ，并朝北。
robot.move(1);  // 朝北移动 1 步到达 (5, 2) ，并朝 北 （不是朝西）。
robot.move(4);  // 下一步继续往北移动将出界，所以逆时针转变方向朝西。
                // 然后，移动 4 步到 (1, 2) ，并朝西。
robot.getPos(); // 返回 [1, 2]
robot.getDir(); // 返回 "West"
```

 

**提示：**

-   `2 <= width, height <= 100`
-   `1 <= num <= 105`
-   `move` ，`getPos` 和 `getDir` **总共** 调用次数不超过 `104` 次。

**答案：**

>   脑筋急转弯，机器人只会在外环转悠。这题 Rust 没法写，move 是个关键字

```c++
const int DIR_R[] = {0, 1, 0, -1};
const int DIR_C[] = {1, 0, -1, 0};
vector<string> NAMES = {"East", "North", "West", "South"};

class Robot {
public:
    Robot(int width, int height) {
        this->width = width;
        this->height = height;
        x = y = d = 0;
    }
    
    void move(int num) {
        int sx = -1, sy = -1, step = 0;
        while (num--) {
            if (step && x == sx && y == sy) {
                num %= step;
            }
            if (sx == -1) {
                if (x == 0 || y == 0 || x == height - 1 || y == width - 1) {
                    sx = x;
                    sy = y;
                    step = 0;
                }
            }
            while (true) {
                int tx = x + DIR_R[d];
                int ty = y + DIR_C[d];
                if (!(0 <= tx && tx < height && 0 <= ty && ty < width)) {
                    d = (d + 1) % 4;
                    continue;
                }
                x = tx;
                y = ty;
                ++step;
                break;
            }
        }
    }
    
    vector<int> getPos() {
        return {y, x};
    }
    
    string getDir() {
        return NAMES[d];
    }
    
private:
    int width, height;
    int x, y, d;
};
```



## T3 5912. 每一个查询的最大美丽值

-   **通过的用户数**870
-   **尝试过的用户数**1264
-   **用户总通过次数**876
-   **用户总提交次数**2596
-   **题目难度** **Medium**

给你一个二维整数数组 `items` ，其中 `items[i] = [pricei, beautyi]` 分别表示每一个物品的 **价格** 和 **美丽值** 。

同时给你一个下标从 **0** 开始的整数数组 `queries` 。对于每个查询 `queries[j]` ，你想求出价格小于等于 `queries[j]` 的物品中，**最大的美丽值** 是多少。如果不存在符合条件的物品，那么查询的结果为 `0` 。

请你返回一个长度与 `queries` 相同的数组 `answer`，其中 `answer[j]`是第 `j` 个查询的答案。

**示例 1：**

```
输入：items = [[1,2],[3,2],[2,4],[5,6],[3,5]], queries = [1,2,3,4,5,6]
输出：[2,4,5,5,6,6]
解释：
- queries[0]=1 ，[1,2] 是唯一价格 <= 1 的物品。所以这个查询的答案为 2 。
- queries[1]=2 ，符合条件的物品有 [1,2] 和 [2,4] 。
  它们中的最大美丽值为 4 。
- queries[2]=3 和 queries[3]=4 ，符合条件的物品都为 [1,2] ，[3,2] ，[2,4] 和 [3,5] 。
  它们中的最大美丽值为 5 。
- queries[4]=5 和 queries[5]=6 ，所有物品都符合条件。
  所以，答案为所有物品中的最大美丽值，为 6 。
```

**示例 2：**

```
输入：items = [[1,2],[1,2],[1,3],[1,4]], queries = [1]
输出：[4]
解释：
每个物品的价格均为 1 ，所以我们选择最大美丽值 4 。
注意，多个物品可能有相同的价格和美丽值。
```

**示例 3：**

```
输入：items = [[10,1000]], queries = [5]
输出：[0]
解释：
没有物品的价格小于等于 5 ，所以没有物品可以选择。
因此，查询的结果为 0 。
```

**提示：**

-   `1 <= items.length, queries.length <= 105`
-   `items[i].length == 2`
-   `1 <= pricei, beautyi, queries[j] <= 109`

**提交：**

>   离线查询，其实只需要前缀最大值即可。哪怕写个树状数组也比 ST 表简单啊……

```rust
use std::cell::RefCell;
use std::cmp::{max, min, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::convert::{From, Into, TryFrom, TryInto};
// use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::marker::PhantomData;
use std::rc::Rc;

macro_rules! new_bufio {
    () => {{
        (io::stdin(), io::stdout(), String::new())
    }};
}
macro_rules! init_stdio {
    ($cin: expr, $cout: expr) => {{
        (BufReader::new($cin.lock()), BufWriter::new($cout.lock()))
    }};
}
macro_rules! scanf {
    ($buf: expr, $div: expr, $($x: ty), +) => {{
        let mut iter = $buf.split($div);
        ($(iter.next().and_then(|token| token.parse::<$x>().ok()),)*)
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
            Some(v) => {
                $fn(v);
            }
            None => {
                $map.insert($key, $val);
            }
        }
    }};
}
macro_rules! lowbit {
    ($x: expr) => {{
        $x & (!$x + 1)
    }};
}

impl Solution {
    pub fn maximum_beauty(mut items: Vec<Vec<i32>>, queries: Vec<i32>) -> Vec<i32> {
        // discrete
        items.sort_by(|v1, v2| v1[0].cmp(&v2[0]));
        let mut pris = vec![];
        let mut btys = vec![];
        items.iter().for_each(|v| {
            pris.push(v[0]);
            btys.push(v[1]);
        });
        // println!("{:?}, {:?}", pris, btys);
        // sparse table
        let mut st = vec![vec![0; 20]; 100010];
        let mut prelog = vec![0; 100010];
        let n = items.len();
        for i in 2..100010 {
            prelog[i] = prelog[i - 1] + if (1 << prelog[i - 1] + 1) == i { 1 } else { 0 };
        }
        for i in (0..n).rev() {
            st[i][0] = btys[i];
            for j in 1..20 {
                if i + (1 << j - 1) >= n {
                    break;
                }
                st[i][j] = st[i][j - 1].max(st[i + (1 << j - 1)][j - 1]);
            }
        }
        // println!("{:?},{:?},{:?},{:?}", st[0], st[1], st[2], st[3]);
        // query
        let mut ans = vec![];
        for q in queries {
            if q < pris[0] {
                ans.push(0);
            } else {
                let mut pos = 0;
                if q >= pris[n - 1] {
                    pos = n - 1;
                } else {
                    pos = Solution::upper_bound(&pris, q) - 1;
                }
                // println!("{:?}", pos);
                let k = prelog[pos + 1];
                // 这个地方 l 恒为 0，但是比赛的时候把 ST 表的查询写成了 l = k ……
                ans.push(st[0][k].max(st[pos + 1 - (1 << k)][k]));
            }
        }
        ans
    }

    fn upper_bound(vals: &Vec<i32>, tar: i32) -> usize {
        let mut l = 0;
        let mut r = vals.len() - 1;
        while l < r {
            let m = (l + r) >> 1;
            if vals[m] > tar {
                r = m;
            } else {
                l = m + 1;
            }
        }
        l
    }
}
```



## T4 5913. 你可以安排的最多任务数目

-   **通过的用户数**84
-   **尝试过的用户数**254
-   **用户总通过次数**93
-   **用户总提交次数**526
-   **题目难度** **Hard**

给你 `n` 个任务和 `m` 个工人。每个任务需要一定的力量值才能完成，需要的力量值保存在下标从 **0** 开始的整数数组 `tasks` 中，第 `i` 个任务需要 `tasks[i]` 的力量才能完成。每个工人的力量值保存在下标从 **0** 开始的整数数组 `workers` 中，第 `j` 个工人的力量值为 `workers[j]` 。每个工人只能完成 **一个** 任务，且力量值需要 **大于等于** 该任务的力量要求值（即 `workers[j] >= tasks[i]` ）。

除此以外，你还有 `pills` 个神奇药丸，可以给 **一个工人的力量值** 增加 `strength` 。你可以决定给哪些工人使用药丸，但每个工人 **最多** 只能使用 **一片** 药丸。

给你下标从 **0** 开始的整数数组`tasks` 和 `workers` 以及两个整数 `pills` 和 `strength` ，请你返回 **最多** 有多少个任务可以被完成。

**示例 1：**

```
输入：tasks = [3,2,1], workers = [0,3,3], pills = 1, strength = 1
输出：3
解释：
我们可以按照如下方案安排药丸：
- 给 0 号工人药丸。
- 0 号工人完成任务 2（0 + 1 >= 1）
- 1 号工人完成任务 1（3 >= 2）
- 2 号工人完成任务 0（3 >= 3）
```

**示例 2：**

```
输入：tasks = [5,4], workers = [0,0,0], pills = 1, strength = 5
输出：1
解释：
我们可以按照如下方案安排药丸：
- 给 0 号工人药丸。
- 0 号工人完成任务 0（0 + 5 >= 5）
```

**示例 3：**

```
输入：tasks = [10,15,30], workers = [0,10,10,10,10], pills = 3, strength = 10
输出：2
解释：
我们可以按照如下方案安排药丸：
- 给 0 号和 1 号工人药丸。
- 0 号工人完成任务 0（0 + 10 >= 10）
- 1 号工人完成任务 1（10 + 10 >= 15）
```

**示例 4：**

```
输入：tasks = [5,9,8,5,9], workers = [1,6,4,2,6], pills = 1, strength = 5
输出：3
解释：
我们可以按照如下方案安排药丸：
- 给 2 号工人药丸。
- 1 号工人完成任务 0（6 >= 5）
- 2 号工人完成任务 2（4 + 5 >= 8）
- 4 号工人完成任务 3（6 >= 5）
```

**提示：**

-   `n == tasks.length`
-   `m == workers.length`
-   `1 <= n, m <= 5 * 104`
-   `0 <= pills <= m`
-   `0 <= tasks[i], workers[j], strength <= 109`

**思路：**

>   二分答案+贪心
>   本题显然具有决策单调性：如果能安排K个任务，一定能安排K-1个任务；如果不能安排K个任务，一定不能安排K+1个任务，因此可以二分答案。
>
>   现在考虑安排K个任务。显然，我们应该选择最容易的K个任务，同时选择最强的K个人。
>
>   我们从难到易来考虑这K个任务。
>
>   一种贪心策略是：
>
>   如果有人能完成当前任务，我们就安排其中能力值最小的那个人去做这一任务。
>   如果没有人能完成当前任务，但当前有药，并且有人能在服药后完成这一任务，我们就安排其中能力值最小的那个人去做这一任务。
>   否则说明无法完成K个任务。
>   另一种贪心策略是：
>
>   如果当前有药，我们就安排服药后能够完成任务的人中能力值最小的那个人去做这一任务。但要注意这个人可能不吃药也能完成任务，此时就不必吃药了。
>   如果当前没有药，我们就安排能完成任务的人中能力值最小的那个人去做这一任务。
>   否则说明无法完成K个任务。
>   这两种贪心策略都是正确的。我们可以这样考虑：在有药丸的情况下，可能会存在A服药能完成任务，B不服药也能完成任务这样的情形。此时我们应该选择谁呢？实际上，因为后面的任务只会更简单，所以A+药或B都一定能完成后面的任务，因此此时使用A+药或使用B其实对后面的任务没有影响。
>
>   时间复杂度\mathcal{O}(N\log^2N)。
>   空间复杂度\mathcal{O}(N)。
>
>   作者：吴自华
>   链接：https://leetcode-cn.com/circle/discuss/cj4dO9/view/Mwspom/
>   来源：力扣（LeetCode）
>   著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int maxTaskAssign(vector<int>& tasks, vector<int>& workers, int pills, int strength) {
        int n = tasks.size(), m = workers.size();
        
        sort(tasks.begin(), tasks.end());
        sort(workers.begin(), workers.end());
        
        auto check = [&](int k) {
            if (m < k)
                return false;
            
            multiset<int> ms(workers.rbegin(), workers.rbegin() + k);
            int rem = pills;
            for (int i = k - 1; i >= 0; --i) {
                // 贪心策略1
                auto it = ms.lower_bound(tasks[i]);
                if (it == ms.end()) {
                    if (rem == 0)
                        return false;
                    it = ms.lower_bound(tasks[i] - strength);
                    if (it == ms.end())
                        return false;
                    rem--;
                    ms.erase(it);
                } else {
                    ms.erase(it);
                }
                
                // 贪心策略2
                // if (rem) {
                //     auto it = ms.lower_bound(tasks[i] - strength);
                //     if (it == ms.end())
                //         return false;
                //     if (*it < tasks[i])
                //         rem--;
                //     ms.erase(it);
                // } else {
                //     auto it = ms.lower_bound(tasks[i]);
                //     if (it == ms.end())
                //         return false;
                //     ms.erase(it);
                // }
            }
            
            return true;
        };
        
        int lo = 1, hi = n;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
                        
            if (check(mid))
                lo = mid + 1;
            else
                hi = mid - 1;
        }
        
        return hi;
    }
};

作者：吴自华
链接：https://leetcode-cn.com/circle/discuss/cj4dO9/view/Mwspom/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

>   单调队列
>   在二分的大框架下，我们也可以从弱到强来考虑选出的K个工人。
>
>   显然，每个工人都必须做一个任务，否则总共做不到K个。对于第ii个工人，我们将所有难度值不超过workers[i] + strength的任务维护在一个双端队列中。由于我们已经对任务进行排序，这个队列天然就是一个单调队列。
>
>   首先考虑这个工人不吃药的情况。此时我们看队列最前面，也即当前最容易的任务是否能够被完成。如果可以，则让该工人做这个最容易的任务。因为任务是必须要做的，而后面的人能力都比当前这个人要强，所以安排当前这个人来做任务是不亏的。
>   如果他不吃药就做不了任务，那就必须吃药。吃药之后，我们应该让他做当前最难的任务，也即队尾的任务。
>   如果吃了药也做不了任何任务，则说明无法完成K个任务。
>   这样，时间复杂度就优化掉了一个log。
>
>   时间复杂度\mathcal{O}(N\log N)。
>   空间复杂度\mathcal{O}(N)。
>
>   作者：吴自华
>   链接：https://leetcode-cn.com/circle/discuss/cj4dO9/view/Mwspom/
>   来源：力扣（LeetCode）
>   著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```c++
class Solution {
public:
    int maxTaskAssign(vector<int>& tasks, vector<int>& workers, int pills, int strength) {
        int n = tasks.size(), m = workers.size();
        
        sort(tasks.begin(), tasks.end());
        sort(workers.begin(), workers.end());
        
        auto check = [&](int k) {
            if (m < k)
                return false;
 
            int ptr = -1, rem = pills;
            deque<int> dq;
            for (int i = m - k; i < m; ++i) {
                while (ptr + 1 < k && tasks[ptr + 1] <= workers[i] + strength)
                    dq.push_back(tasks[++ptr]);
                if (dq.empty())
                    return false;
                if (dq.front() <= workers[i])
                    dq.pop_front();
                else if (rem > 0) {
                    rem--;
                    dq.pop_back();
                } else 
                    return false;
            }

            return true;
        };
        
        int lo = 1, hi = n;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
                        
            if (check(mid))
                lo = mid + 1;
            else
                hi = mid - 1;
        }
        
        return hi;
    }
};

作者：吴自华
链接：https://leetcode-cn.com/circle/discuss/cj4dO9/view/Mwspom/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

