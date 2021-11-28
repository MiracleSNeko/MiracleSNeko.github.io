---
title: LeetCode 周赛 269
date: 2021-11-28 21:05:37
tags: LeetCode 周赛
---

----------

# LeetCode 周赛 268

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-269/problems/find-target-indices-after-sorting-array/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-269/problems/k-radius-subarray-averages/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-269/problems/removing-minimum-and-maximum-from-array/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-269/problems/find-all-people-with-secret/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1340 / 4292 | MiracleSNeko | 12   | 0:40:39  | 0:01:58                                                      | 0:16:32 1                                                    | 0:35:39                                                      |                                                              |

>   手速慢了是真没办法

## T1 5938. 找出数组排序后的目标下标

-   **User Accepted:**3482
-   **User Tried:**3512
-   **Total Accepted:**3520
-   **Total Submissions:**4023
-   **Difficulty:** **Easy**

给你一个下标从 **0** 开始的整数数组 `nums` 以及一个目标元素 `target` 。

**目标下标** 是一个满足 `nums[i] == target` 的下标 `i` 。

将 `nums` 按 **非递减** 顺序排序后，返回由 `nums` 中目标下标组成的列表。如果不存在目标下标，返回一个 **空** 列表。返回的列表必须按 **递增** 顺序排列。

**示例 1：**

```
输入：nums = [1,2,5,2,3], target = 2
输出：[1,2]
解释：排序后，nums 变为 [1,2,2,3,5] 。
满足 nums[i] == 2 的下标是 1 和 2 。
```

**示例 2：**

```
输入：nums = [1,2,5,2,3], target = 3
输出：[3]
解释：排序后，nums 变为 [1,2,2,3,5] 。
满足 nums[i] == 3 的下标是 3 。
```

**示例 3：**

```
输入：nums = [1,2,5,2,3], target = 5
输出：[4]
解释：排序后，nums 变为 [1,2,2,3,5] 。
满足 nums[i] == 5 的下标是 4 。
```

**示例 4：**

```
输入：nums = [1,2,5,2,3], target = 4
输出：[]
解释：nums 中不含值为 4 的元素。
```

**提示：**

-   `1 <= nums.length <= 100`
-   `1 <= nums[i], target <= 100`

**提交：**

```rust
impl Solution {
    pub fn target_indices(mut nums: Vec<i32>, target: i32) -> Vec<i32> {
        nums.sort();
        nums.iter().enumerate()
            .filter(|(_, &v)| v == target)
            .map(|(idx, _)| idx as i32)
            .collect()
    }
}
```

## T2 5939. 半径为 k 的子数组平均值

-   **User Accepted:**2893
-   **User Tried:**3329
-   **Total Accepted:**2923
-   **Total Submissions:**9155
-   **Difficulty:** **Medium**

给你一个下标从 **0** 开始的数组 `nums` ，数组中有 `n` 个整数，另给你一个整数 `k` 。

**半径为 k 的子数组平均值** 是指：`nums` 中一个以下标 `i` 为 **中心** 且 **半径** 为 `k` 的子数组中所有元素的平均值，即下标在 `i - k` 和 `i + k` 范围（**含** `i - k` 和 `i + k`）内所有元素的平均值。如果在下标 `i` 前或后不足 `k` 个元素，那么 **半径为 k 的子数组平均值** 是 `-1` 。

构建并返回一个长度为 `n` 的数组 `avgs` ，其中 `avgs[i]` 是以下标 `i` 为中心的子数组的 **半径为 k 的子数组平均值** 。

`x` 个元素的 **平均值** 是 `x` 个元素相加之和除以 `x` ，此时使用截断式 **整数除法** ，即需要去掉结果的小数部分。

-   例如，四个元素 `2`、`3`、`1` 和 `5` 的平均值是 `(2 + 3 + 1 + 5) / 4 = 11 / 4 = 3.75`，截断后得到 `3` 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/11/07/eg1.png)

```
输入：nums = [7,4,3,9,1,8,5,2,6], k = 3
输出：[-1,-1,-1,5,4,4,-1,-1,-1]
解释：
- avg[0]、avg[1] 和 avg[2] 是 -1 ，因为在这几个下标前的元素数量都不足 k 个。
- 中心为下标 3 且半径为 3 的子数组的元素总和是：7 + 4 + 3 + 9 + 1 + 8 + 5 = 37 。
  使用截断式 整数除法，avg[3] = 37 / 7 = 5 。
- 中心为下标 4 的子数组，avg[4] = (4 + 3 + 9 + 1 + 8 + 5 + 2) / 7 = 4 。
- 中心为下标 5 的子数组，avg[5] = (3 + 9 + 1 + 8 + 5 + 2 + 6) / 7 = 4 。
- avg[6]、avg[7] 和 avg[8] 是 -1 ，因为在这几个下标后的元素数量都不足 k 个。
```

**示例 2：**

```
输入：nums = [100000], k = 0
输出：[100000]
解释：
- 中心为下标 0 且半径 0 的子数组的元素总和是：100000 。
  avg[0] = 100000 / 1 = 100000 。
```

**示例 3：**

```
输入：nums = [8], k = 100000
输出：[-1]
解释：
- avg[0] 是 -1 ，因为在下标 0 前后的元素数量均不足 k 。
```

**提示：**

-   `n == nums.length`
-   `1 <= n <= 105`
-   `0 <= nums[i], k <= 105`

>   注意前缀和 $10^5 \times 10^5$ 是会爆 `i32` 的！！！

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

## T3 5940. 从数组中移除最大值和最小值

-   **User Accepted:**2832
-   **User Tried:**3041
-   **Total Accepted:**2870
-   **Total Submissions:**5009
-   **Difficulty:** **Medium**

给你一个下标从 **0** 开始的数组 `nums` ，数组由若干 **互不相同** 的整数组成。

`nums` 中有一个值最小的元素和一个值最大的元素。分别称为 **最小值** 和 **最大值** 。你的目标是从数组中移除这两个元素。

一次 **删除** 操作定义为从数组的 **前面** 移除一个元素或从数组的 **后面** 移除一个元素。

返回将数组中最小值和最大值 **都** 移除需要的最小删除次数。 

**示例 1：**

```
输入：nums = [2,10,7,5,4,1,8,6]
输出：5
解释：
数组中的最小元素是 nums[5] ，值为 1 。
数组中的最大元素是 nums[1] ，值为 10 。
将最大值和最小值都移除需要从数组前面移除 2 个元素，从数组后面移除 3 个元素。
结果是 2 + 3 = 5 ，这是所有可能情况中的最小删除次数。
```

**示例 2：**

```
输入：nums = [0,-4,19,1,8,-2,-3,5]
输出：3
解释：
数组中的最小元素是 nums[1] ，值为 -4 。
数组中的最大元素是 nums[2] ，值为 19 。
将最大值和最小值都移除需要从数组前面移除 3 个元素。
结果是 3 ，这是所有可能情况中的最小删除次数。 
```

**示例 3：**

```
输入：nums = [101]
输出：1
解释：
数组中只有这一个元素，那么它既是数组中的最小值又是数组中的最大值。
移除它只需要 1 次删除操作。
```

**提示：**

-   `1 <= nums.length <= 105`
-   `-105 <= nums[i] <= 105`
-   `nums` 中的整数 **互不相同**

>   只有都从左边、都从右边、一左一右三种情况

```rust
impl Solution {
    pub fn minimum_deletions(nums: Vec<i32>) -> i32 {
        let mut maxpos = nums.iter().enumerate().max_by_key(|item| item.1).unwrap().0;
        let mut minpos = nums.iter().enumerate().min_by_key(|item| item.1).unwrap().0;
        let len = nums.len();
        if maxpos < minpos {
            std::mem::swap(&mut maxpos, &mut minpos);
        }
        let (l, r, m) = (minpos + 1, len - maxpos, maxpos - minpos);
        (l + r).min((l + m).min(r + m)) as i32
    }
}
```

## T4 5941. 找出知晓秘密的所有专家

-   **User Accepted:**648
-   **User Tried:**1842
-   **Total Accepted:**756
-   **Total Submissions:**5755
-   **Difficulty:** **Hard**

给你一个整数 `n` ，表示有 `n` 个专家从 `0` 到 `n - 1` 编号。另外给你一个下标从 0 开始的二维整数数组 `meetings` ，其中 `meetings[i] = [xi, yi, timei]` 表示专家 `xi` 和专家 `yi` 在时间 `timei` 要开一场会。一个专家可以同时参加 **多场会议** 。最后，给你一个整数 `firstPerson` 。

专家 `0` 有一个 **秘密** ，最初，他在时间 `0` 将这个秘密分享给了专家 `firstPerson` 。接着，这个秘密会在每次有知晓这个秘密的专家参加会议时进行传播。更正式的表达是，每次会议，如果专家 `xi` 在时间 `timei` 时知晓这个秘密，那么他将会与专家 `yi` 分享这个秘密，反之亦然。

秘密共享是 **瞬时发生** 的。也就是说，在同一时间，一个专家不光可以接收到秘密，还能在其他会议上与其他专家分享。

在所有会议都结束之后，返回所有知晓这个秘密的专家列表。你可以按 **任何顺序** 返回答案。

**示例 1：**

```
输入：n = 6, meetings = [[1,2,5],[2,3,8],[1,5,10]], firstPerson = 1
输出：[0,1,2,3,5]
解释：
时间 0 ，专家 0 将秘密与专家 1 共享。
时间 5 ，专家 1 将秘密与专家 2 共享。
时间 8 ，专家 2 将秘密与专家 3 共享。
时间 10 ，专家 1 将秘密与专家 5 共享。
因此，在所有会议结束后，专家 0、1、2、3 和 5 都将知晓这个秘密。
```

**示例 2：**

```
输入：n = 4, meetings = [[3,1,3],[1,2,2],[0,3,3]], firstPerson = 3
输出：[0,1,3]
解释：
时间 0 ，专家 0 将秘密与专家 3 共享。
时间 2 ，专家 1 与专家 2 都不知晓这个秘密。
时间 3 ，专家 3 将秘密与专家 0 和专家 1 共享。
因此，在所有会议结束后，专家 0、1 和 3 都将知晓这个秘密。
```

**示例 3：**

```
输入：n = 5, meetings = [[3,4,2],[1,2,1],[2,3,1]], firstPerson = 1
输出：[0,1,2,3,4]
解释：
时间 0 ，专家 0 将秘密与专家 1 共享。
时间 1 ，专家 1 将秘密与专家 2 共享，专家 2 将秘密与专家 3 共享。
注意，专家 2 可以在收到秘密的同一时间分享此秘密。
时间 2 ，专家 3 将秘密与专家 4 共享。
因此，在所有会议结束后，专家 0、1、2、3 和 4 都将知晓这个秘密。
```

**示例 4：**

```
输入：n = 6, meetings = [[0,2,1],[1,3,1],[4,5,1]], firstPerson = 1
输出：[0,1,2,3]
解释：
时间 0 ，专家 0 将秘密与专家 1 共享。
时间 1 ，专家 0 将秘密与专家 2 共享，专家 1 将秘密与专家 3 共享。
因此，在所有会议结束后，专家 0、1、2 和 3 都将知晓这个秘密。
```

**提示：**

-   `2 <= n <= 105`
-   `1 <= meetings.length <= 105`
-   `meetings[i].length == 3`
-   `0 <= xi, yi <= n - 1`
-   `xi != yi`
-   `1 <= timei <= 105`
-   `1 <= firstPerson <= n - 1`

**题解：**

>   并查集，注意需要处理同一时刻的多组会议：
>
>   排序完成后，遍历所有时刻。同一时刻可能存在多场会议，由于秘密共享是瞬时发生的，且同一时刻的会议是乱序的，不存在先后，所以对每一时刻的处理分为两步：
>
>   第一轮遍历：首先判断两位专家中是否有人知道秘密，若有人知道秘密，则将两位专家的祖先节点都置为0。完成该操作后，无论两位专家是否有人知道秘密，都将两个专家合并，因为同一时刻的其他会议中，可能有其他知道秘密的专家将秘密分享给这两位中的任何一个，若存在此情况，则当前时刻过后，这两位专家也知道了秘密。
>   第二轮遍历：处理两种情况，
>   场景一：第一轮遍历中，先遍历到某场会议，此时两位专家都不知道秘密，但在后面的遍历中，其中一位专家知道了秘密，由于上一步做了合并集合处理，此时将两位专家的祖先节点均置为0即可。
>   场景二：第一轮遍历中，先遍历到某场会议，此时两位专家都不知道秘密，在后面的遍历中，这两位专家均没有被分享秘密，这时需要将两位专家从合并的集合中分离出来，如果不分离出来，在后面某时刻，如果这两位专家其中一个知道了秘密，那么会认为这两位专家都知道了秘密，但事实上，由于该时刻已过去，秘密无法分享给另一位专家。示例2即为此情况。
>
>   作者：fudan
>   链接：https://leetcode-cn.com/problems/find-all-people-with-secret/solution/bing-cha-ji-pai-xu-javashuang-bai-xiang-5gbrx/
>   来源：力扣（LeetCode）
>   著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

>   比赛最后提交的代码有一个巨他妈傻逼的 bug：要按照时间排序的场景，我他妈居然用了 HashMap 而不是 BTreeMap。差点就 AK 了草啊

```rust
use std::collections::BTreeMap;

struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<i32>,
    known: Vec<i32>,
}

impl DisjointSet {
    #[inline]
    fn with_capacity(cap: usize) -> Self {
        Self {
            parent: (0..=cap).collect(),
            rank: vec![0; cap + 1],
            known: vec![i32::MAX; cap + 1]
        }
    }

    pub fn find(&mut self, u: usize) -> usize {
        if u != self.parent[u] {
            self.parent[u] = self.find(self.parent[u]);
        }
        self.parent[u]
    }

    pub fn union(&mut self, u: usize, v: usize, t: i32) -> bool {
        let (mut fu, mut fv) = (self.find(u), self.find(v));
        if fu == fv && (self.known[u] > t && self.known[v] > t) {
            false
        } else {
            if self.rank[fu] > self.rank[fv] {
                std::mem::swap(&mut fu, &mut fv);
            }
            self.parent[fu] = fv;
            if self.rank[fu] == self.rank[fv] {
                self.rank[fv] += 1;
            }
            let time = self.known[u].min(self.known[v]);
            self.known[u] = time;
            self.known[v] = time;
            true
        }
    }
}

impl Solution {
    pub fn find_all_people(n: i32,  mut meetings: Vec<Vec<i32>>, first_person: i32) -> Vec<i32> {
        let mut dsj = DisjointSet::with_capacity(n as usize);
        meetings.sort_by(|lhs, rhs| lhs[2].cmp(&rhs[2]));
        let mut mp = BTreeMap::new();
        for meet in meetings {
            (*mp.entry(meet[2]).or_insert(vec![])).push(vec![meet[0], meet[1]]);
        }
        dsj.known[first_person as usize] = 0;
        dsj.known[0] = 0;
        for (time, meet) in mp {
            for m in meet.iter() {
                dsj.union(m[0] as usize, m[1] as usize, time);
            }
            for m in meet.iter().rev() {
                dsj.union(m[0] as usize, m[1] as usize, time);
            }
        }
        dsj.known.into_iter().enumerate()
            .filter(|&(_, t)| t != i32::MAX)
            .map(|(idx, _)| idx as i32)
            .collect()
    }
}
```

>   参考题解，较为精炼的处理重复时间的方法

```rust
use std::collections::BTreeMap;

impl Solution {
    pub fn find_all_people(n: i32,  meetings: Vec<Vec<i32>>, first_person: i32) -> Vec<i32> {
        fn find(tar: usize, parent: &mut Vec<usize>) -> usize {
            if tar != parent[tar] {
                parent[tar] = find(parent[tar], parent);
            }
            parent[tar]
        }
        let mut parent = (0..=n as usize).collect::<Vec<_>>();
        parent[first_person as usize] = 0;
        let mut mp = BTreeMap::new();
        for meet in meetings {
            (*mp.entry(meet[2]).or_insert(vec![])).push((meet[0] as usize, meet[1] as usize))
        }
        for (_, pairs) in mp {
            for &(u, v) in pairs.iter() {
                let (fu, fv) = (find(u, &mut parent), find(v, &mut parent));
                if parent[fu] == 0 || parent[fv] == 0 {
                    parent[fu] = 0;
                    parent[fv] = 0;
                }
                parent[fu] = parent[fv];
            }
            for &(u, v) in pairs.iter() {
                let (fu, fv) = (find(u, &mut parent), find(v, &mut parent));
                if parent[fu] == 0 || parent[fv] == 0 {
                    parent[fu] = 0;
                    parent[fv] = 0;
                } else {
                    parent[u] = u;
                    parent[v] = v;
                }
            }
        }
        parent.into_iter().enumerate()
            .filter(|&(_, p)| p == 0 )
            .map(|(idx, _)| idx as i32 )
            .collect()
    }
}
```

>   另外这一题也有 Dijkstra 解法

>   把所有会议连接关系当成一张网络图，离源点的距离表示知道秘密的时间，对每个节点求最早知道时间。
>   前驱节点的最早知道时间，要小于等于和它开会的后序节点的时间，才能通知到后序节点的那个人。
>   答案为距离更新过的节点。
>
>   作者：Mountain-Ocean
>   链接：https://leetcode-cn.com/problems/find-all-people-with-secret/solution/dijkstra-by-mountain-ocean-i98l/
>   来源：力扣（LeetCode）
>   著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

```c++
class Solution {
public:
    typedef pair<int, int> PII;
    vector<int> findAllPeople(int n, vector<vector<int>>& meetings, int firstPerson) {
        vector<int> ans;
        vector<int> dis(n, INT_MAX);
        vector<vector<PII>> graph(n);
        for (auto& list : meetings) {
            int a = list[0], b = list[1], t = list[2];
            graph[a].push_back({b, t});
            graph[b].push_back({a, t});
        }
        dijkstra(graph, dis, firstPerson);
        for (int i = 0; i < n; ++i) {
            if (dis[i] != INT_MAX) {
                ans.push_back(i);
            }
        }
        return ans;
    }

    void dijkstra(vector<vector<PII>>& graph, vector<int>& dis, int start) {
        priority_queue<PII, vector<PII>, greater<PII>> pq;
        pq.push({0, 0});
        pq.push({0, start});
        dis[0] = 0;
        dis[start] = 0;
        while (!pq.empty()) {
            auto [t, u] = pq.top();
            pq.pop();
            if (t > dis[u]) continue;
            for (auto [v, time] : graph[u]) {
                // t <= time: 前驱节点的最早知道时间，要小于等于和它开会的后序节点的时间
                if (t <= time && time < dis[v]) {
                    dis[v] = time;
                    pq.push({time, v});
                }
            }
        }
    }
};

作者：Mountain-Ocean
链接：https://leetcode-cn.com/problems/find-all-people-with-secret/solution/dijkstra-by-mountain-ocean-i98l/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

