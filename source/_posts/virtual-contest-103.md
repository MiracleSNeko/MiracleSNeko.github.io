---
title: LeetCode 虚拟周赛 103
date: 2021-11-15 23:39:59
tags: LeetCode 周赛总结
---

-----

# LeetCode 虚拟周赛 103

| 排名     | 用户名                  | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-103/problems/smallest-range-i/) | [题目2 (6)](https://leetcode-cn.com/contest/weekly-contest-103/problems/snakes-and-ladders/) | [题目3 (6)](https://leetcode-cn.com/contest/weekly-contest-103/problems/smallest-range-ii/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-103/problems/online-election/) |
| -------- | ----------------------- | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 16 / 574 | MiracleSNeko *虚拟竞赛* | 3    | 0:02:57  | 0:02:57                                                      |                                                              |                                                              |                                                              |

>   人做麻了，被 T2 自己的傻逼想法卡死

## T1 908. 最小差值 I

-   **通过的用户数**257
-   **尝试过的用户数**282
-   **用户总通过次数**263
-   **用户总提交次数**485
-   **题目难度** **Easy**

给你一个整数数组 `nums`，请你给数组中的每个元素 `nums[i]` 都加上一个任意数字 `x` （`-k <= x <= k`），从而得到一个新数组 `result` 。

返回数组 `result` 的最大值和最小值之间可能存在的最小差值。

**示例 1：**

```
输入：nums = [1], k = 0
输出：0
解释：result = [1]
```

**示例 2：**

```
输入：nums = [0,10], k = 2
输出：6
解释：result = [2,8]
```

**示例 3：**

```
输入：nums = [1,3,6], k = 3
输出：0
解释：result = [3,3,3] or result = [4,4,4]
```

**提示：**

-   `1 <= nums.length <= 10000`
-   `0 <= nums[i] <= 10000`
-   `0 <= k <= 10000`

**提交：**

```Rust
impl Solution {
    pub fn smallest_range_i(nums: Vec<i32>, k: i32) -> i32 {
        let max = nums.iter().max().unwrap();
        let min = nums.iter().min().unwrap();
        (max-min-2*k).max(0)
    }
}
```

## T2 909. 蛇梯棋

-   **通过的用户数**11
-   **尝试过的用户数**57
-   **用户总通过次数**11
-   **用户总提交次数**177
-   **题目难度** **Medium**

给你一个大小为 `n x n` 的整数矩阵 `board` ，方格按从 `1` 到 `n2` 编号，编号遵循 [转行交替方式](https://baike.baidu.com/item/牛耕式转行书写法/17195786) ，**从左下角开始** （即，从 `board[n - 1][0]` 开始）每一行交替方向。

玩家从棋盘上的方格 `1` （总是在最后一行、第一列）开始出发。

每一回合，玩家需要从当前方格 `curr` 开始出发，按下述要求前进：

-   选定目标方格 next ，目标方格的编号符合范围 [curr + 1, min(curr + 6, n^2)]

    -   该选择模拟了掷 **六面体骰子** 的情景，无论棋盘大小如何，玩家最多只能有 6 个目的地。

-   传送玩家：如果目标方格 `next` 处存在蛇或梯子，那么玩家会传送到蛇或梯子的目的地。否则，玩家传送到目标方格 `next` 。 

-   当玩家到达编号 `n^2` 的方格时，游戏结束。

`r` 行 `c` 列的棋盘，按前述方法编号，棋盘格中可能存在 “蛇” 或 “梯子”；如果 `board[r][c] != -1`，那个蛇或梯子的目的地将会是 `board[r][c]`。编号为 `1` 和 `n^2` 的方格上没有蛇或梯子。

注意，玩家在每回合的前进过程中最多只能爬过蛇或梯子一次：就算目的地是另一条蛇或梯子的起点，玩家也 **不能** 继续移动。

-   举个例子，假设棋盘是 `[[-1,4],[-1,3]]` ，第一次移动，玩家的目标方格是 `2` 。那么这个玩家将会顺着梯子到达方格 `3` ，但 **不能** 顺着方格 `3` 上的梯子前往方格 `4` 。

返回达到编号为 `n^2` 的方格所需的最少移动次数，如果不可能，则返回 `-1`。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2018/09/23/snakes.png)

```
输入：board = [[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1,-1],[-1,35,-1,-1,13,-1],[-1,-1,-1,-1,-1,-1],[-1,15,-1,-1,-1,-1]]
输出：4
解释：
首先，从方格 1 [第 5 行，第 0 列] 开始。 
先决定移动到方格 2 ，并必须爬过梯子移动到到方格 15 。
然后决定移动到方格 17 [第 3 行，第 4 列]，必须爬过蛇到方格 13 。
接着决定移动到方格 14 ，且必须通过梯子移动到方格 35 。 
最后决定移动到方格 36 , 游戏结束。 
可以证明需要至少 4 次移动才能到达最后一个方格，所以答案是 4 。 
```

**示例 2：**

```
输入：board = [[-1,-1],[-1,3]]
输出：1
```

 **提示：**

-   `n == board.length == board[i].length`
-   `2 <= n <= 20`
-   `grid[i][j]` 的值是 `-1` 或在范围 `[1, n2]` 内
-   编号为 `1` 和 `n2` 的方格上没有蛇或梯子

**题解：**

>   完全想复杂了，转图写了个 Dijk，还有个本质的 bug

```rust
/// Dummy Luogu/LeetCode Playground
use std::cell::RefCell;
use std::cmp::{max, min, Reverse};
use std::collections::{BTreeMap, BinaryHeap, HashMap, HashSet, VecDeque};
use std::convert::{From, Into, TryFrom, TryInto};
// use std::io;
use std::marker::PhantomData;
use std::rc::Rc;
use std::slice::SliceIndex;

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

struct LinkedForwardStar {
    to: Vec<usize>,
    nxt: Vec<usize>,
    wht: Vec<i32>,
    hd: Vec<usize>,
    tot: usize,
}

impl LinkedForwardStar {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            to: vec![usize::MAX; cap],
            nxt: vec![usize::MAX; cap],
            wht: vec![0; cap],
            hd: vec![usize::MAX; cap],
            tot: 0
        }
    }

    pub fn add(&mut self, u: usize, v: usize, w: i32) {
        self.tot += 1;
        self.to[self.tot] = v;
        self.nxt[self.tot] = self.hd[u];
        self.wht[self.tot] = w;
        self.hd[u] = self.tot;
    }

    pub fn dijkstra_solver(&self, src: usize, n: usize) -> i32 {
        const INF: i32 = 0x3f3f3f3f;
        let mut dist = vec![INF; n + 1];
        let mut path = vec![0; n + 1];
        let mut vis = vec![vec![0; 2]; n + 1];
        dist[src] = 0;
        let mut heap = BinaryHeap::new();
        heap.push((dist[src], src, 0));
        while let Some((_, u, flag)) = heap.pop() {
            if vis[u][flag] != 0 {
                continue;
            }
            vis[u][flag] = 1;
            let mut e = self.hd[u];
            while e != usize::MAX {
                let v = self.to[e];
                let w = self.wht[e];
                if dist[v] > dist[u] + w.abs() && !(flag == 1 && w != 1) {
                    dist[v] = dist[u] + w.abs();
                    path[v] = u;
                    if w != 1 {
                        heap.push((dist[v], v, 1));
                    } else {
                        heap.push((dist[v], v, 0));
                    }
                }
                e = self.nxt[e];
            }
        }
        let mut ans = 1;
        let mut prv = path[n];
        let mut cur = path[n];
        while cur != 0 {
            // println!("{:#?}", cur);
            cur = path[cur];
            if prv - cur > 6 {
                prv = cur;
                ans += 1;
            }
        }
        ans
    }
}

impl Solution {
    pub fn snakes_and_ladders(board: Vec<Vec<i32>>) -> i32 {
        let n = board.len();
        let mut lfs = LinkedForwardStar::with_capacity(200020);
        for i in 0..n {
            for j in 0..n {
                let idx = Solution::to_index(i, j, n);
                let i = n - i - 1;
                if idx == n * n {
                    break;
                }
                lfs.add(idx, idx + 1, 1);
                if board[i][j] != -1 {
                    // println!("{}->{}", idx, board[i][j]);
                    lfs.add(idx, board[i][j] as usize, -1);
                }
            }
        }
        lfs.dijkstra_solver(1, n * n)
    }

    pub fn to_index(x: usize, y: usize, n: usize) -> usize {
        if ((x / n) & 1) == 0 {
            x * n + y + 1
        } else {
            x * n + (n - y)
        }
    }
}
```

>   然后挂在了如下测试点
>
>   ```
>   input: [[-1,-1,-1],[-1,9,8],[-1,8,9]]
>   output: 2
>   answer: 1
>   ```
>
>   最生气的是这题我 tm 还做过，BFS 的 AC 代码如下

```go
func init() {
	debug.SetGCPercent(-1)
}

func updatePos(idx int, n int) (r int, c int) {
	r, c = (idx-1)/n, (idx-1)%n
	if r&1 == 1 {
		c = n - 1 - c
	}
	r = n - 1 - r
	return
}

type Pair struct {
	idx  int
	step int
}

func snakesAndLadders(board [][]int) int {
	n := len(board)
	vis := make([]bool, n*n+1)
	q := []Pair{{1, 0}}
	for len(q) > 0 {
		p := q[0]
		q = q[1:]
		for i := 1; i <= 6; i++ {
			nxt := p.idx + i
			if nxt < 0 || nxt > n*n {
				break
			}
			r, c := updatePos(nxt, n)
			// 梯子或者蛇
			if board[r][c] != -1 {
				nxt = board[r][c]
			}
			if nxt == n*n {
				return p.step + 1
			}
			if !vis[nxt] {
				vis[nxt] = true
				q = append(q, Pair{nxt, p.step + 1})
			}
		}
	}
	return -1
}
```

>   其实写到一半已经意识到这个图 tm 所有的边权都是 1，该转 BFS 了，但是还是头铁 md

## L3 910. 最小差值 II

-   **通过的用户数**7
-   **尝试过的用户数**154
-   **用户总通过次数**7
-   **用户总提交次数**646
-   **题目难度** **Medium**

给你一个整数数组 `A`，对于每个整数 `A[i]`，可以选择 **`x = -K` 或是 `x = K`** （`**K**` 总是非负整数），并将 `x` 加到 `A[i]` 中。

在此过程之后，得到数组 `B`。

返回 `B` 的最大值和 `B` 的最小值之间可能存在的最小差值。

**示例 1：**

```
输入：A = [1], K = 0
输出：0
解释：B = [1]
```

**示例 2：**

```
输入：A = [0,10], K = 2
输出：6
解释：B = [2,8]
```

**示例 3：**

```
输入：A = [1,3,6], K = 3
输出：3
解释：B = [4,6,3]
```

**提示：**

-   `1 <= A.length <= 10000`
-   `0 <= A[i] <= 10000`
-   `0 <= K <= 10000`

**题解：**

>   [太难了，只能画图凭直觉 - 最小差值 II - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/smallest-range-ii/solution/tai-nan-liao-zhi-neng-hua-tu-ping-zhi-jue-by-user8/)
>
>   贪心思路。至于为什么是对的，证明可以参考官解。

```rust
impl Solution {
    pub fn smallest_range_ii(mut nums: Vec<i32>, k: i32) -> i32 {
        nums.sort();
        let n = nums.len();
        let mut ans = nums[n - 1] - nums[0];
        for i in 0..n-1 {
            let max = (nums[i] + k).max(nums[n - 1] - k);
            let min = (nums[0] + k).min(nums[i + 1] - k);
            ans = ans.min(max - min);
        }
        ans
    }
}
```

## T4 911. 在线选举

-   **通过的用户数**14
-   **尝试过的用户数**35
-   **用户总通过次数**14
-   **用户总提交次数**94
-   **题目难度** **Medium**

在选举中，第 `i` 张票是在时间为 `times[i]` 时投给 `persons[i]` 的。

现在，我们想要实现下面的查询函数： `TopVotedCandidate.q(int t)` 将返回在 `t` 时刻主导选举的候选人的编号。

在 `t` 时刻投出的选票也将被计入我们的查询之中。在平局的情况下，最近获得投票的候选人将会获胜。

**示例：**

```
输入：["TopVotedCandidate","q","q","q","q","q","q"], [[[0,1,1,0,0,1,0],[0,5,10,15,20,25,30]],[3],[12],[25],[15],[24],[8]]
输出：[null,0,1,1,0,0,1]
解释：
时间为 3，票数分布情况是 [0]，编号为 0 的候选人领先。
时间为 12，票数分布情况是 [0,1,1]，编号为 1 的候选人领先。
时间为 25，票数分布情况是 [0,1,1,0,0,1]，编号为 1 的候选人领先（因为最近的投票结果是平局）。
在时间 15、24 和 8 处继续执行 3 个查询。
```

**提示：**

1.  `1 <= persons.length = times.length <= 5000`
2.  `0 <= persons[i] <= persons.length`
3.  `times` 是严格递增的数组，所有元素都在 `[0, 10^9]` 范围中。
4.  每个测试用例最多调用 `10000` 次 `TopVotedCandidate.q`。
5.  `TopVotedCandidate.q(int t)` 被调用时总是满足 `t >= times[0]`。

**题解：**

>   离线查询。预处理每个选票到达时刻的结果，记录 (Time, Winner) 对，这可以通过维护一个 CurrMaxCnt 和 CurrWinner ，在每次插入选票的时候更新即可。之后对于每次查询，二分位置即可。

```Java
class TopVotedCandidate {
    List<Vote> A;
    public TopVotedCandidate(int[] persons, int[] times) {
        A = new ArrayList();
        Map<Integer, Integer> count = new HashMap();
        int leader = -1;  // current leader
        int m = 0;  // current number of votes for leader

        for (int i = 0; i < persons.length; ++i) {
            int p = persons[i], t = times[i];
            int c = count.getOrDefault(p, 0) + 1;
            count.put(p, c);

            if (c >= m) {
                if (p != leader) {  // lead change
                    leader = p;
                    A.add(new Vote(leader, t));
                }

                if (c > m) m = c;
            }
        }
    }

    public int q(int t) {
        int lo = 1, hi = A.size();
        while (lo < hi) {
            int mi = lo + (hi - lo) / 2;
            if (A.get(mi).time <= t)
                lo = mi + 1;
            else
                hi = mi;
        }

        return A.get(lo - 1).person;
    }
}

class Vote {
    int person, time;
    Vote(int p, int t) {
        person = p;
        time = t;
    }
}

作者：LeetCode
链接：https://leetcode-cn.com/problems/online-election/solution/zai-xian-xuan-ju-by-leetcode/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

