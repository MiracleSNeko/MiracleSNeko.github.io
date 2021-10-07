---
title: LeetCode 周赛 258
date: 2021-09-12 16:51:09
tags: LeetCode 周赛总结
---

---

# LeetCode 周赛 258

| 排名       | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-258/problems/reverse-prefix-of-word/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-258/problems/number-of-pairs-of-interchangeable-rectangles/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-258/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-258/problems/smallest-missing-genetic-value-in-each-subtree/) |
| ---------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 738 / 4518 | MiracleSNeko | 12   | 1:09:58  | 0:20:57                                                      | 0:33:48 1                                                    | 0:59:58 1                                                    |                                                              |

## T1 5867. 反转单词前缀

-   **通过的用户数**3464
-   **尝试过的用户数**3498
-   **用户总通过次数**3507
-   **用户总提交次数**4264
-   **题目难度** **Easy**

给你一个下标从 **0** 开始的字符串 `word` 和一个字符 `ch` 。找出 `ch` 第一次出现的下标 `i` ，**反转** `word` 中从下标 `0` 开始、直到下标 `i` 结束（含下标 `i` ）的那段字符。如果 `word` 中不存在字符 `ch` ，则无需进行任何操作。

-   例如，如果 `word = "abcdefd"` 且 `ch = "d"` ，那么你应该 **反转** 从下标 0 开始、直到下标 `3` 结束（含下标 `3` ）。结果字符串将会是 `"***dcba***efd"` 。

返回 **结果字符串** 。

**示例 1：**

```
输入：word = "abcdefd", ch = "d"
输出："dcbaefd"
解释："d" 第一次出现在下标 3 。 
反转从下标 0 到下标 3（含下标 3）的这段字符，结果字符串是 "dcbaefd" 。
```

**示例 2：**

```
输入：word = "xyxzxe", ch = "z"
输出："zxyxxe"
解释："z" 第一次也是唯一一次出现是在下标 3 。
反转从下标 0 到下标 3（含下标 3）的这段字符，结果字符串是 "zxyxxe" 。
```

**示例 3：**

```
输入：word = "abcd", ch = "z"
输出："abcd"
解释："z" 不存在于 word 中。
无需执行反转操作，结果字符串是 "abcd" 。
```

**提示：**

-   `1 <= word.length <= 250`
-   `word` 由小写英文字母组成
-   `ch` 是一个小写英文字母

**我提交的代码**

```c++
class Solution
{
public:
    std::string reversePrefix(std::string word, char ch)
    {
        CHEATING_HEAD;
        auto pos = word.find(ch);
        if (pos != str::npos)
        {
            str prv = str(word.begin(), word.begin() + pos + 1);
            str pst = str(word.begin() + pos + 1, word.end());
            prv = str(prv.rbegin(), prv.rend());
            return prv + pst;
        }
        return word;
    }
};
```



## T2 5868. 可互换矩形的组数

-   **通过的用户数**2656
-   **尝试过的用户数**3291
-   **用户总通过次数**2705
-   **用户总提交次数**9238
-   **题目难度** **Medium**

用一个下标从 **0** 开始的二维整数数组 `rectangles` 来表示 `n` 个矩形，其中 `rectangles[i] = [widthi, heighti]` 表示第 `i` 个矩形的宽度和高度。

如果两个矩形 `i` 和 `j`（`i < j`）的宽高比相同，则认为这两个矩形 **可互换** 。更规范的说法是，两个矩形满足 `widthi/heighti == widthj/heightj`（使用实数除法而非整数除法），则认为这两个矩形 **可互换** 。

计算并返回 `rectangles` 中有多少对 **可互换** 矩形。

**示例 1：**

```
输入：rectangles = [[4,8],[3,6],[10,20],[15,30]]
输出：6
解释：下面按下标（从 0 开始）列出可互换矩形的配对情况：
- 矩形 0 和矩形 1 ：4/8 == 3/6
- 矩形 0 和矩形 2 ：4/8 == 10/20
- 矩形 0 和矩形 3 ：4/8 == 15/30
- 矩形 1 和矩形 2 ：3/6 == 10/20
- 矩形 1 和矩形 3 ：3/6 == 15/30
- 矩形 2 和矩形 3 ：10/20 == 15/30
```

**示例 2：**

```
输入：rectangles = [[4,5],[7,8]]
输出：0
解释：不存在成对的可互换矩形。
```

**提示：**

-   `n == rectangles.length`
-   `1 <= n <= 105`
-   `rectangles[i].length == 2`
-   `1 <= widthi, heighti <= 105`

**我提交的代码**

```c++
class Solution
{
public:
/*     static constexpr auto Fracs = []()
    {
        std::array<i64, 21> fracs({1});
        for (i64 i = 1; i <= 20; ++i)
        {
            fracs[i] = fracs[i - 1] * i;
        }
        return fracs;
    }(); */

    i64 interchangeableRectangles(std::vector<std::vector<int>> &rectangles)
    {
        // 找比例相等个数，求组合数 nC2
        Vecd fracs;
        for (auto &rect : rectangles)
        {
            fracs.emplace_back(static_cast<f64>(rect[0]) / static_cast<f64>(rect[1]));
        }
        HashMap<f64, i32> mp;
        for (auto &f : fracs)
        {
            if (mp.count(f) == 0)
            {
                mp[f] = 1;
            }
            else
            {
                mp[f] += 1;
            }
        }
        i64 ans = 0;
        for (auto &kv : mp)
        {
            i64 cnt = kv.second;
            if (cnt > 1)
            {
                ans += cnt * (cnt - 1) / 2;
            }
        }
        return ans;
    }
};
```

>   注：此处用 `double` 类型不涉及加减运算，只涉及判等，所以 `double` 作为 `key` 不会导致精度问题。

-   WA 的原因： 把 $C(n, 2) = n(n - 1) / 2$ 写成了 $n!(n-1)!/2$



## L3 5869. 两个回文子序列长度的最大乘积

-   **通过的用户数**971
-   **尝试过的用户数**1189
-   **用户总通过次数**1017
-   **用户总提交次数**2093
-   **题目难度** **Medium**

给你一个字符串 `s` ，请你找到 `s` 中两个 **不相交回文子序列** ，使得它们长度的 **乘积最大** 。两个子序列在原字符串中如果没有任何相同下标的字符，则它们是 **不相交** 的。

请你返回两个回文子序列长度可以达到的 **最大乘积** 。

**子序列** 指的是从原字符串中删除若干个字符（可以一个也不删除）后，剩余字符不改变顺序而得到的结果。如果一个字符串从前往后读和从后往前读一模一样，那么这个字符串是一个 **回文字符串** 。

**示例 1：**

![example-1](https://assets.leetcode.com/uploads/2021/08/24/two-palindromic-subsequences.png)

```
输入：s = "leetcodecom"
输出：9
解释：最优方案是选择 "ete" 作为第一个子序列，"cdc" 作为第二个子序列。
它们的乘积为 3 * 3 = 9 。
```

**示例 2：**

```
输入：s = "bb"
输出：1
解释：最优方案为选择 "b" （第一个字符）作为第一个子序列，"b" （第二个字符）作为第二个子序列。
它们的乘积为 1 * 1 = 1 。
```

**示例 3：**

```
输入：s = "accbcaxxcxx"
输出：25
解释：最优方案为选择 "accca" 作为第一个子序列，"xxcxx" 作为第二个子序列。
它们的乘积为 5 * 5 = 25 。
```

**提示：**

-   `2 <= s.length <= 12`
-   `s` 只含有小写英文字母。

**我提交的代码**

```c++
class Solution
{
public:
    i32 countOnes(i32 x)
    {
        x = ((x >> 1) & 0x55555555) + (x & 0x55555555);
        x = ((x >> 2) & 0x33333333) + (x & 0x33333333);
        x = ((x >> 4) & 0x0f0f0f0f) + (x & 0x0f0f0f0f);
        x = ((x >> 8) & 0x00ff00ff) + (x & 0x00ff00ff);
        x = ((x >> 16) & 0x0000ffff) + (x & 0x0000ffff);
        return x;
    }

    int maxProduct(str s)
    {
        auto check = [&s](i32 status) -> bool
        {
            Veci pos1({});
            for (i32 i = 0; i < 13; ++i)
            {
                if (status & (1 << i))
                    pos1.push_back(i);
            }
            if (pos1.size() == 1)
                return true;
            i32 i = 0, j = pos1.size() - 1;
            while (i <= j)
            {
                if (s[pos1[i]] != s[pos1[j]])
                    return false;
                ++i, --j;
            }
            return true;
        };
        // 预处理所有回文子串
        i32 sts = 1 << s.size();
        Veci checked;
        for (i32 st = 1; st < sts; ++st)
        {
            if (check(st))
                checked.push_back(st);
        }
        // 遍历所有回文字串求结果
        i32 ans = -Inf;
        i32 clen = checked.size();
        for (i32 i = 0; i < clen; ++i)
        {
            i32 sti = checked[i];
            for (i32 j = i + 1; j < clen; ++j)
            {
                i32 stj = checked[j];
                if ((sti & stj) == 0) // 无重复
                {
                    ans = std::max(ans, countOnes(sti) * countOnes(stj));
                }
            }
        }
        return ans;
    }
};
```

>   注：本质是暴力遍历

-   TLE 的原因：一开始直接暴力二重循环，没预处理回文字符串，遍历了大量的垃圾状态

**其他思路**

dfs 两个子序列。对于位置 i ，两个子序列可以选择用或者不用。

```c++
class Solution {
public:
    int ans = 0;
    int maxProduct(string s) {
        string s1, s2;
        dfs(s, s1, s2, 0);
        return ans;
    }
    
    void dfs(string &s, string s1, string s2, int index) {
        if(check(s1) && check(s2)) ans = max(ans, int(s1.size() * s2.size()));
        if(index == s.size()) return;
        dfs(s, s1 + s[index], s2, index + 1);//子序列s1使用该字符
        dfs(s, s1, s2 + s[index], index + 1);//子序列s2使用该字符
        dfs(s, s1, s2, index + 1);//子序列都不使用该字符
    }
    
    bool check(string &s) {
        int l = 0, r = s.size() - 1;
        while(l < r) {
            if(s[l++] != s[r--]) return false;
        }
        return true;
    }
};

作者：ytmartian
链接：https://leetcode-cn.com/problems/maximum-product-of-the-length-of-two-palindromic-subsequences/solution/dfsliang-ge-zi-xu-lie-by-ytmartian-svyn/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```



## T4 5870. 每棵子树内缺失的最小基因值

-   **通过的用户数**175
-   **尝试过的用户数**539
-   **用户总通过次数**213
-   **用户总提交次数**1154
-   **题目难度** **Hard**

有一棵根节点为 `0` 的 **家族树** ，总共包含 `n` 个节点，节点编号为 `0` 到 `n - 1` 。给你一个下标从 **0** 开始的整数数组 `parents` ，其中 `parents[i]` 是节点 `i` 的父节点。由于节点 `0` 是 **根** ，所以 `parents[0] == -1` 。

总共有 `105` 个基因值，每个基因值都用 **闭区间** `[1, 105]` 中的一个整数表示。给你一个下标从 **0** 开始的整数数组 `nums` ，其中 `nums[i]` 是节点 `i` 的基因值，且基因值 **互不相同** 。

请你返回一个数组 `ans` ，长度为 `n` ，其中 `ans[i]` 是以节点 `i` 为根的子树内 **缺失** 的 **最小** 基因值。

节点 `x` 为根的 **子树** 包含节点 `x` 和它所有的 **后代** 节点。

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/08/23/case-1.png)

```
输入：parents = [-1,0,0,2], nums = [1,2,3,4]
输出：[5,1,1,1]
解释：每个子树答案计算结果如下：
- 0：子树包含节点 [0,1,2,3] ，基因值分别为 [1,2,3,4] 。5 是缺失的最小基因值。
- 1：子树只包含节点 1 ，基因值为 2 。1 是缺失的最小基因值。
- 2：子树包含节点 [2,3] ，基因值分别为 [3,4] 。1 是缺失的最小基因值。
- 3：子树只包含节点 3 ，基因值为 4 。1是缺失的最小基因值。
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/08/23/case-2.png)

```
输入：parents = [-1,0,1,0,3,3], nums = [5,4,6,2,1,3]
输出：[7,1,1,4,2,1]
解释：每个子树答案计算结果如下：
- 0：子树内包含节点 [0,1,2,3,4,5] ，基因值分别为 [5,4,6,2,1,3] 。7 是缺失的最小基因值。
- 1：子树内包含节点 [1,2] ，基因值分别为 [4,6] 。 1 是缺失的最小基因值。
- 2：子树内只包含节点 2 ，基因值为 6 。1 是缺失的最小基因值。
- 3：子树内包含节点 [3,4,5] ，基因值分别为 [2,1,3] 。4 是缺失的最小基因值。
- 4：子树内只包含节点 4 ，基因值为 1 。2 是缺失的最小基因值。
- 5：子树内只包含节点 5 ，基因值为 3 。1 是缺失的最小基因值。
```

**示例 3：**

```
输入：parents = [-1,2,3,0,2,4,1], nums = [2,3,4,5,6,7,8]
输出：[1,1,1,1,1,1,1]
解释：所有子树都缺失基因值 1 。
```

**提示：**

-   `n == parents.length == nums.length`
-   `2 <= n <= 105`
-   对于 `i != 0` ，满足 `0 <= parents[i] <= n - 1`
-   `parents[0] == -1`
-   `parents` 表示一棵合法的树。
-   `1 <= nums[i] <= 105`
-   `nums[i]` 互不相同。

**解法笔记**

**解法一：启发式合并**
遍历整棵树，统计每棵子树包含的基因值集合以及缺失的最小基因值，记作 $\textit{mex}$。合并基因值集合时，总是从小的往大的合并（类似并查集的按秩合并），同时更新当前子树的 $\textit{mex}$ 的最大值。合并完成后再不断自增子树的 $\textit{mex}$ 直至其不在基因值集合中。

这一方法同时也适用于有相同基因值的情况。

时间复杂度：$O(n\log n)$。证明。

```go
func smallestMissingValueSubtree(parents []int, nums []int) []int {
	n := len(parents)
	g := make([][]int, n)
	for w := 1; w < n; w++ {
		v := parents[w]
		g[v] = append(g[v], w)
	}
	mex := make([]int, n)
	var f func(int) map[int]bool
	f = func(v int) map[int]bool {
		set := map[int]bool{}
		mex[v] = 1
		for _, w := range g[v] {
			s := f(w)
			// 保证总是从小集合合并到大集合上
			if len(s) > len(set) {
				set, s = s, set
			}
			for x := range s {
				set[x] = true
			}
			if mex[w] > mex[v] {
				mex[v] = mex[w]
			}
		}
		set[nums[v]] = true
		for set[mex[v]] {
			mex[v]++ // 不断自增 mex 直至其不在基因值集合中
		}
		return set
	}
	f(0)
	return mex
}

作者：endlesscheng
链接：https://leetcode-cn.com/problems/smallest-missing-genetic-value-in-each-subtree/solution/go-qi-fa-shi-he-bing-by-endlesscheng-kmff/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

**解法二：利用无重复基因值的性质**
由于没有重复基因值，若存在节点 $x$，其基因值等于 $1$，则从 $x$ 到根的这一条链上的所有节点的 $\textit{mex}$ 均超过 $1$，而其余节点的 $\textit{mex}$ 值均为 $1$。我们顺着 $x$ 往根上走，同时收集当前子树的基因值到集合中，然后更新当前子树的 $\textit{mex}$ 值。

时间复杂度：$O(n)$。

```go
func smallestMissingValueSubtree(parents []int, nums []int) []int {
	n := len(parents)
	g := make([][]int, n)
	for w := 1; w < n; w++ {
		v := parents[w]
		g[v] = append(g[v], w)
	}

	mex := make([]int, n)
	for i := range mex {
		mex[i] = 1
	}
	
	set := map[int]bool{}
	vis := make([]bool, n)
	var f func(int)
	f = func(v int) {
		set[nums[v]] = true // 收集基因值到 set 中
		for _, w := range g[v] {
			if !vis[w] {
				f(w)
			}
		}
	}
	
	// 找基因值等于 1 的节点 x
	x := -1
	for i, v := range nums {
		if v == 1 {
			x = i
			break
		}
	}
	// x 顺着父节点往上走
	for cur := 2; x >= 0; x = parents[x] {
		f(x)
		vis[x] = true // 这是保证时间复杂度的关键：之后遍历父节点子树时，就无需再次遍历 x 子树了
		for set[cur] {
			cur++ // 不断自增直至不在基因值集合中
		}
		mex[x] = cur
	}
	return mex
}

作者：endlesscheng
链接：https://leetcode-cn.com/problems/smallest-missing-genetic-value-in-each-subtree/solution/go-qi-fa-shi-he-bing-by-endlesscheng-kmff/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

**不一样的思路：主席树+dfs序（突破值互不相同的限制）**

-   看到很多题解是从基因值互不相同的条件入手写的，后来突然发现我的解法不需要考虑这个
-   可能我的思路比较繁琐，时间复杂度有没有O(n)的优秀，固在此也只是想提供一个与众不同的思路罢了（勿喷）
-   前置知识：主席树，dfs序
-   注意到，这题的本质就是求mex，但不同的是，其不是真正的mex，最小的数是1，mex最小的数是0，但不影响做这题。
-   将问题转化一下，求一个区间的mex，可以线段树、主席树，这里使用主席树做方便
-   主席树找mex的思路
    -   利用权值线段树在每个权值上记录该数最后出现的下标，再次基础上加上可持续化，便是主席树了
    -   最终查询 `[L, R]` 区间的mex时，则是在版本 `R` 的权值线段树中找下标小于 `L` 的最小的数即可
-   再来就是怎么将区间转化成树上查询，显然可以用到dfs序了

```c++
const int N = 1e5 + 5;
struct pii {
    int x, y;
};
pii p[N];

int head[N], cnt, tim;

//初始化
void init(int n) { fill_n(head, n + 5, -1); cnt = -1; tim = 0; }

struct edges {
    int to, next;
    void add(int t, int n) {
        to = t, next = n;
    }
}edge[N << 1]; //无向图则需要乘2

inline void add(int u, int v) {
    edge[++cnt].add(v, head[u]);
    head[u] = cnt;
}

int a[N], ram, root[N], vis[N];
void dfs(int u, vector<int>& num) {
    p[u].x = ++tim;
    a[tim] = num[u]; // 重置数组
    for (int i = head[u]; ~i; i = edge[i].next) {
        dfs(edge[i].to, num);
    }
    p[u].y = tim;
}


struct nodes {int l, r, minv; } hjt[N * 25];

int modify(int pre, int l, int r, int val, int pos) {
    int now = ++ram;
    hjt[now] = hjt[pre];
    if (l == r) {
        hjt[now].minv = pos;
        return now;
    }
    int mid = (l + r) >> 1;
    if (val <= mid) hjt[now].l = modify(hjt[now].l, l, mid, val, pos);
    else hjt[now].r = modify(hjt[now].r, mid + 1, r, val, pos);
    hjt[now].minv = min(hjt[hjt[now].l].minv, hjt[hjt[now].r].minv);
    return now;
}

int query(int tr, int l, int r, int ql) {
    if (l == r) return l;
    int mid = (l + r) >> 1;
    if (hjt[hjt[tr].l].minv < ql) return query(hjt[tr].l, l, mid, ql);
    return query(hjt[tr].r, mid + 1, r, ql);
}


class Solution {
public:
    vector<int> smallestMissingValueSubtree(vector<int>& pa, vector<int>& nums) {
        int n = pa.size();
        init(n);
        ram = 0;
        for (int i = 0; i < n; ++i) {
            if (pa[i] != -1) {
                add(pa[i], i);
            }
        }
        dfs(0, nums); // 预处理dfs序
        vector<int> ans(n, 0);
        int len = *max_element(nums.begin(), nums.end()) + 1; // 找到最大值，记得+1
        for (int i = 1; i <= n; ++i) {
            root[i] = modify(root[i - 1], 1, len, a[i], i); // 主席树插入，注意此时用到的数组是a，不是nums了
        }
        for (int i = 0; i < n; ++i) {
            // cout << p[i].x << ' ' << p[i].y << endl;
            ans[i] = query(root[p[i].y], 1, len, p[i].x);// 查询结点i的子树，意味在查询dfs序中区间[p[i].x, p[i].y]的答案
        }
        return ans;
    }
};

作者：haoboy
链接：https://leetcode-cn.com/problems/smallest-missing-genetic-value-in-each-subtree/solution/zhu-xi-shu-dfsxu-tu-po-zhi-hu-bu-xiang-t-7nh8/
来源：力扣（LeetCode）
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。
```

