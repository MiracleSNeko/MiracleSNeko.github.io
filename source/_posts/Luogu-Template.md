---
title: 洛谷模板题集合
date: 2021-10-10 00:01:12
tags: 模板
---

---------

# 洛谷模板题集合

> 参考[这篇文章](https://www.cnblogs.com/Ender-hz/p/15018563.html)列出的题目

## 0. 工具宏 & 包导入声明

```rust
/// Dummy Luogu/LeetCode Playground
use std::cell::RefCell;
use std::cmp::{max, min, Reverse};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque, BTreeMap};
use std::convert::{From, Into, TryFrom, TryInto};
use std::io;
use std::marker::PhantomData;
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
            Some(v) => {
                $fn(v);
            }
            None => {
                $map.insert($key, $val);
            }
        }
    }};
}
```

## 1. 普及-

### P3367 并查集

```rust
struct DisjointSet<T>
where
    T: Sized + Eq + Copy + TryInto<usize>,
{
    parent: Vec<usize>,
    rank: Vec<usize>,
    phantom: PhantomData<T>,
}

impl<T> DisjointSet<T>
where
    T: Sized + Eq + Copy + TryInto<usize>,
{
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            parent: vec![usize::MAX; cap],
            rank: vec![0; cap],
            phantom: PhantomData,
        }
    }
    pub fn modify(&mut self, x: T, p: usize) {
        match x.try_into() {
            Ok(x) => { self.parent[x] = p; },
            Err(_) => panic!()
        }
    }
    pub fn union(&mut self, x: T, y: T) -> bool {
        let (mut fx, mut fy) = (self.find(x), self.find(y));
        if fx == fy || fx == usize::MAX || fy == usize::MAX {
            false
        } else {
            if self.rank[fx] > self.rank[fy] {
                std::mem::swap(&mut fx, &mut fy);
            }
            self.parent[fx] = fy;
            if self.rank[fx] == self.rank[fy] {
                self.rank[fy] += 1;
            }
            true
        }
    }
    pub fn find(&mut self, x: T) -> usize {
        match x.try_into() {
            Ok(x) => self.find_wrapper(x),
            Err(_) => usize::MAX
        }
    }
    fn find_wrapper(&mut self, x: usize) -> usize {
        if x != self.parent[x] {
            self.parent[x] = self.find_wrapper(self.parent[x]);
        }
        self.parent[x]
    }
}

fn main() -> io::Result<()> {
    let cin = io::stdin();
    let mut buf = String::new();
    getline!(cin, buf);
    // n m
    if let (Some(n), Some(mut m)) = scanf!(buf, char::is_whitespace, i32, i32) {
        let mut dsj = DisjointSet::<i32>::with_capacity(n as usize + 1);
        (1..=n).for_each(|i| dsj.modify(i, i as usize));
        while m > 0 {
            getline!(cin, buf);
            if let (Some(op), Some(arg1), Some(arg2)) =
                scanf!(buf, char::is_whitespace, i32, i32, i32)
            {
                match op {
                    1 => { dsj.union(arg1, arg2); },
                    2 => { println!("{:}", if dsj.find(arg1) == dsj.find(arg2) && dsj.find(arg1) != usize::MAX {"Y"} else {"N"})},
                    _ => unreachable!()
                }
            }
            m -= 1;
        }
    }
    Ok(())
}
```



### P3371 单源最短路（非随机数据）

> 看描述似乎允许 SPFA ，但是原版 SPFA 还是被卡掉了三个用例

```rust
mod my {
    #[derive(Debug, Clone)]
    struct LFSNode {
        to: usize,
        next: usize,
        w: i32,
    }
    pub struct LinkedForwardStar {
        edges: Vec<LFSNode>,
        head: Vec<usize>,
        tot: usize,
    }
    impl LinkedForwardStar {
        pub fn with_capacity(nodes: usize, edges: usize) -> Self {
            Self {
                edges: vec![
                    LFSNode {
                        to: usize::MAX,
                        next: usize::MAX,
                        w: 0
                    };
                    edges
                ],
                head: vec![usize::MAX; nodes],
                tot: 0,
            }
        }
        pub fn add(&mut self, src: usize, dst: usize, weight: i32) {
            self.edges[self.tot].next = self.head[src];
            self.edges[self.tot].to = dst;
            self.edges[self.tot].w = weight;
            self.head[src] = self.tot;
            self.tot += 1;
        }
    }
    pub fn spfa_solver(lfs: LinkedForwardStar, src: usize) -> Vec<i32> {
        let mut vis = vec![false; lfs.tot];
        let mut dist = vec![i32::MAX; lfs.head.len()];
        let mut queue = vec![src];
        dist[src] = 0;
        vis[src] = true;
        while !queue.is_empty() {
            let x = queue.pop().unwrap();
            vis[x] = false;
            let mut i = lfs.head[x];
            while i != usize::MAX {
                let y = lfs.edges[i].to;
                if dist[y] - lfs.edges[i].w > dist[x] {
                    dist[y] = dist[x] + lfs.edges[i].w;
                    if !vis[y] {
                        vis[y] = true;
                        queue.push(y);
                    }
                }
                i = lfs.edges[i].next;
            }
        }
        dist
    }
    pub fn dijkstra_solver(lfs: LinkedForwardStar, src: usize) -> Vec<i32> {
        let mut vis = vec![false; lfs.head.len()];
        let mut dist = vec![i32::MAX; lfs.head.len()];
        let mut x = src;
        dist[src] = 0;
        while !vis[x] {
            let mut curr = i32::MAX;
            vis[x] = true;
            let mut y = lfs.head[x];
            while y != usize::MAX {
                if !vis[lfs.edges[y].to] && dist[lfs.edges[y].to] - lfs.edges[y].w > dist[x] {
                    dist[lfs.edges[y].to] = dist[x] + lfs.edges[y].w;
                }
                y = lfs.edges[y].next;
            }
            for i in 1..lfs.head.len() {
                if dist[i] < curr && !vis[i] {
                    curr = dist[i];
                    x = i;
                }
            }
        }
        dist
    }
}
use my::*;

fn main() -> io::Result<()> {
    let (cin, mut buf) = init_cin!();
    getline!(cin, buf);
    if let (Some(n), Some(mut m), Some(src)) = scanf!(buf, char::is_whitespace, usize, usize, usize)
    {
        let mut lfs = LinkedForwardStar::with_capacity(n + 1, m + 1);
        while m > 0 {
            m -= 1;
            getline!(cin, buf);
            if let (Some(u), Some(v), Some(w)) = scanf!(buf, char::is_whitespace, usize, usize, i32)
            {
                lfs.add(u, v, w);
            }
        }
        // spfa_solver(lfs, src)
        dijkstra_solver(lfs, src)
            .iter()
            .skip(1)
            .for_each(|i| print!("{:} ", i))
    }
    Ok(())
}
```



### P1226 快速幂

```rust
fn fast_pow(x: i64, n: i64, m: i64) -> i64 {
    let (mut x, mut n, mut ret) = (x, n, 1);
    while n > 0 {
        if (n & 1) == 1 {
            ret *= x;
            ret %= m;
        }
        x *= x;
        x %= m;
        n >>= 1;
    }
    ret % m
}

fn main() -> io::Result<()> {
    let (cin, mut buf) = init_cin!();
    getline!(cin, buf);
    if let (Some(x), Some(n), Some(m)) = scanf!(buf, char::is_whitespace, i64, i64, i64) {
        println!("{0}^{1} mod {2}={3}", x, n, m, fast_pow(x, n, m));
    }
    Ok(())
}
```



### P4779 单源最短路（标准）

> 不优化的 Dijkstra 也寄了，堆优化的 AC

```rust
#[derive(Default, Debug, Clone)]
struct LFSNode {
    to: usize,
    next: usize,
    w: i32,
}
pub struct LinkedForwardStar {
    edges: Vec<LFSNode>,
    head: Vec<usize>,
    tot: usize,
}
impl LinkedForwardStar {
    pub fn with_capacity(node_cap: usize, edge_cap: usize) -> Self {
        Self {
            edges: vec![LFSNode::default(); edge_cap],
            head: vec![usize::MAX; node_cap],
            tot: 0,
        }
    }
    pub fn add(&mut self, src: usize, dst: usize, weight: i32) {
        self.edges[self.tot].next = self.head[src];
        self.edges[self.tot].to = dst;
        self.edges[self.tot].w = weight;
        self.head[src] = self.tot;
        self.tot += 1;
    }
}
pub fn spfa_solver(lfs: LinkedForwardStar, src: usize) -> Vec<i32> {
    let mut vis = vec![false; lfs.tot];
    let mut dist = vec![i32::MAX; lfs.head.len()];
    let mut queue = vec![src];
    dist[src] = 0;
    vis[src] = true;
    while !queue.is_empty() {
        let x = queue.pop().unwrap();
        vis[x] = false;
        let mut i = lfs.head[x];
        while i != usize::MAX {
            let y = lfs.edges[i].to;
            if dist[y] - lfs.edges[i].w > dist[x] {
                dist[y] = dist[x] + lfs.edges[i].w;
                if !vis[y] {
                    vis[y] = true;
                    queue.push(y);
                }
            }
            i = lfs.edges[i].next;
        }
    }
    dist
}
/// Dijkstra with Heap optimization
pub fn dijkstra_solver(lfs: LinkedForwardStar, src: usize) -> Vec<i32> {
    let mut vis = vec![false; lfs.head.len()];
    let mut dist = vec![i32::MAX; lfs.head.len()];
    let mut pq = BinaryHeap::new();
    dist[src] = 0;
    pq.push(Reverse((dist[src], src)));
    while let Some(Reverse((_, u))) = pq.pop() {
        if vis[u] {
            continue;
        }
        vis[u] = true;
        let mut e = lfs.head[u];
        while e != usize::MAX {
            let v = lfs.edges[e].to;
            let w = lfs.edges[e].w;
            if dist[v] > dist[u] + w {
                dist[v] = dist[u] + w;
                pq.push(Reverse((dist[v], v)));
            }
            e = lfs.edges[e].next;
        }
    }
    dist
}

fn main() -> io::Result<()> {
    let (cin, mut buf) = init_cin!();
    getline!(cin, buf);
    if let (Some(n), Some(mut m), Some(src)) = scanf!(buf, char::is_whitespace, usize, usize, usize)
    {
        let mut lfs = LinkedForwardStar::with_capacity(n + 1, m + 1);
        while m > 0 {
            m -= 1;
            getline!(cin, buf);
            if let (Some(u), Some(v), Some(w)) = scanf!(buf, char::is_whitespace, usize, usize, i32)
            {
                lfs.add(u, v, w);
            }
        }
        // spfa_solver(lfs, src)
        dijkstra_solver(lfs, src)
            .iter()
            .skip(1)
            .for_each(|i| print!("{:} ", i))
    }
    Ok(())
}
```

### P3383 【模板】线性筛素数

```rust
fn euler_sieve(n: usize) -> Vec<i32> {
    let mut valid = vec![true; n + 1];
    let mut ans = vec![0; n + 1];
    let mut tot = 0;
    for i in 2..=n {
        if valid[i] {
            tot += 1;
            ans[tot] = i;
        }
        let mut j = 1;
        while j <= tot && i * ans[j] <= n {
            valid[i * ans[j]] = false;
            if i % ans[j] == 0 {
                break;
            }
            j += 1;
        }
    }
    ans.into_iter().skip(1).take(tot).map(|i| i as i32).collect()
}

fn main() -> io::Result<()> {
    let (cin, mut buf) = init_cin!();
    getline!(cin, buf);
    if let (Some(n), Some(mut qs)) = scanf!(buf, char::is_whitespace, usize, i32) {
        let primes = euler_sieve(n);
        while qs > 0 {
            getline!(cin, buf);
            if let (Some(q),) = scanf!(buf, char::is_whitespace, usize) {
                println!("{0}", primes[q-1]);
            }
            qs -= 1;
        }
    }
    Ok(())
}
```

### P3366 【模板】最小生成树

```rust
struct Edge {
    u: usize,
    v: usize,
    w: i32,
}

/// Kruskal 算法
/// 复杂度 `M \log M`
/// 思想：每次取剩下的边权最小的边，如果加上这条边后图中出现了一个环，就不选这条边（可以通过并查集）判断
fn kruskal(edges: &mut Vec<Edge>, nodes_num: usize) -> i32 {
    let mut parent = (0..=nodes_num).collect::<Vec<_>>();
    fn find(u: usize, parent: &mut Vec<usize>) -> usize {
        if u == parent[u] {
            u
        } else {
            parent[u] = find(parent[u], parent);
            parent[u]
        }
    }
    edges.sort_by(|lhs, rhs| lhs.w.cmp(&rhs.w));
    let mut ans = 0;
    let mut cnt = nodes_num;
    for i in 0..edges.len() {
        let (p1, p2) = (find(edges[i].u, &mut parent), find(edges[i].v, &mut parent));
        if p1 != p2 {
            parent[p1] = p2;
            ans += edges[i].w;
            cnt -= 1;
            if cnt == 1 {
                break;
            }
        }
    }
    if cnt == 1 {
        ans
    } else {
        i32::MAX
    }
}

fn main() -> io::Result<()> {
    let (cin, mut buf) = init_cin!();
    getline!(cin, buf);
    if let (Some(n), Some(mut m)) = scanf!(buf, char::is_whitespace, usize, usize) {
        let mut edges = Vec::with_capacity(m);
        while m > 0 {
            getline!(cin, buf);
            if let (Some(u), Some(v), Some(w)) = scanf!(buf, char::is_whitespace, usize, usize, i32)
            {
                edges.push(Edge { u, v, w });
            }
            m -= 1;
        }
        let ans = kruskal(&mut edges, n);
        println!("{0}", if ans == i32::MAX { "orz".to_string() } else { ans.to_string() });
    }
    Ok(())
}
```

### P3378 【模板】堆

>   不想手写，直接用了 std

### P3370 【模板】字符串哈希

>   交了半天都是 TLE，一开 O2 过了，烦死

```rust
fn string_hash(s: &String) -> i32 {
    let mut hash = s.chars()
        .fold(0, |mut hash: i32, ch| {
            hash = hash.wrapping_add(ch as i32);
            hash = hash.wrapping_add(hash.wrapping_shl(10));
            hash ^= hash.wrapping_shr(6);
            hash
        });
    hash = hash.wrapping_add(hash.wrapping_shl(3));
    hash ^= hash.wrapping_shr(11);
    hash = hash.wrapping_add(hash.wrapping_shl(15));
    hash
}

/*
const MOD: usize = 2147483647;
const POW: usize = 257;

fn string_hash(s: &String) -> usize {
    s.chars()
        .fold(0, |hash, ch| (hash * POW + (ch as usize)) % MOD)
}
*/

fn main() -> io::Result<()> {
    let (cin, mut buf) = init_cin!();
    getline!(cin, buf);
    if let (Some(n),) = scanf!(buf, char::is_whitespace, usize) {
        let mut bucket = vec![0; n];
        let mut i = 0usize;
        let mut ans = n;
        while i < n {
            getline!(cin, buf);
            bucket[i] = string_hash(&buf);
            i += 1;
        }
        bucket.sort();
        (1..n).for_each(|i| if bucket[i] == bucket[i - 1] { ans -= 1 } );
        println!("{0}", ans);
    }
    Ok(())
}
```

### P1177 【模板】快速排序

```rust
/*
fn sort(vals: &mut Vec<i32>, l: usize, r: usize) {
    if r - l > 10 {
        quick_sort(vals, l, r);
    } else {
        insert_sort(vals, l, r);
    }
}
*/

fn quick_sort(vals: &mut Vec<i32>, l: usize, r: usize) {
    if l >= r {
        return;
    }
    let mut i = l;
    let mut j = r;
    let pivot = vals[((l + r) >> 1) as usize];
    while i <= j {
        while pivot > vals[i] {
            i += 1;
        }
        while pivot < vals[j] {
            j -= 1;
        }
        if i <= j {
            let tmp = vals[i];
            vals[i] = vals[j];
            vals[j] = tmp;
            i += 1;
            j -= 1;
        }
    }
    quick_sort(vals, l, j);
    quick_sort(vals, i, r);
}

/*
fn insert_sort(vals: &mut Vec<i32>, l: usize, r: usize) {
    for i in l..=r {
        let mut j = i;
        let pivot = vals[i];
        while j >= 1 && vals[j - 1] > pivot {
            vals[j] = vals[j - 1];
            j -= 1;
        }
        vals[j] = pivot;
    }
}
*/

fn main() -> io::Result<()> {
    let (cin, mut buf) = init_cin!();
    getline!(cin, buf);
    if let (Some(n),) = scanf!(buf, char::is_whitespace, usize) {
        getline!(cin, buf);
        let mut vals = buf
            .split(char::is_whitespace)
            .map(|i| i.parse::<i32>())
            .filter(|i| i.is_ok())
            .map(|i| i.unwrap())
            .collect::<Vec<_>>();
        let len = vals.len();
        assert_eq!(len, n);
        let len = vals.len();
        quick_sort(&mut vals, 0, len - 1);
        vals.iter().for_each(|i| print!("{0} ", i));
    }
    Ok(())
}
```

普及 / 提高-
P1886 【模板】单调队列 / 滑动窗口
P3382 【模板】三分法
P3374 【模板】树状数组 1
P3811 【模板】乘法逆元
P3372 【模板】线段树 1
P3375 【模板】KMP字符串匹配
P3368 【模板】树状数组 2
P3379 【模板】最近公共祖先（LCA）
P1939 【模板】矩阵加速（数列）
P3385 【模板】负环
P3865 【模板】ST 表
P3390 【模板】矩阵快速幂
P4549 【模板】裴蜀定理
P4779 【模板】单源最短路径（标准版）
P5788 【模板】单调栈
普及+ / 提高
P5431 【模板】乘法逆元2
P5367 【模板】康托展开
P5960 【模板】差分约束算法
P2613 【模板】有理数取余
P2252 【模板】威佐夫博弈 / [SHOI2002] 取石子游戏
P3373 【模板】线段树 2
P5905 【模板】Johnson 全源最短路
P2197 【模板】nim 游戏
P3387 【模板】缩点
P1439 【模板】最长公共子序列
P3386 【模板】二分图最大匹配
P3388 【模板】割点（割顶）
P5656 【模板】二元一次不定方程 (exgcd)
提高+ / 省选-
P3377 【模板】左偏树（可并堆）
P3381 【模板】最小费用最大流
P3369 【模板】普通平衡树
P6091 【模板】原根
P4781 【模板】拉格朗日插值
P5903 【模板】树上 k 级祖先
P5854 【模板】笛卡尔树
P6086 【模板】Prufer 序列
P1368 【模板】最小表示法
P5632 【模板】Stoer-Wagner算法
P5490 【模板】扫描线
P3805 【模板】manacher 算法
P1495 【模板】中国剩余定理(CRT)/曹冲养猪
P2742 【模板】二维凸包 / [USACO5.1]圈奶牛Fencing the Cows
P3389 【模板】高斯消元法
P4783 【模板】矩阵求逆
P3796 【模板】AC自动机（加强版）
P3808 【模板】AC自动机（简单版）
P5826 【模板】子序列自动机
P3846 【模板】BSGS / [TJOI2007] 可爱的质数
P3376 【模板】网络最大流
P3391 【模板】文艺平衡树
P5091 【模板】扩展欧拉定理
P3807 【模板】卢卡斯定理/Lucas 定理
P3384 【模板】轻重链剖分/树链剖分
P3812 【模板】线性基
P3919 【模板】可持久化线段树 1（可持久化数组）
P3834 【模板】可持久化线段树 2（主席树）
P7112 【模板】行列式求值
P1919 【模板】A*B Problem升级版（FFT快速傅里叶）
P3803 【模板】多项式乘法（FFT）
P4525 【模板】自适应辛普森法1
省选 / NOI-
P4196 【模板】半平面交 / [CQOI2006]凸多边形
P1452 【模板】旋转卡壳 / [USACO03FALL]Beauty Contest G
P3809 【模板】后缀排序
P6177 【模板】树分块 / Count on a tree II
P5906 【模板】回滚莫队&不删除莫队
P4782 【模板】2-SAT 问题
P5357 【模板】AC自动机（二次加强版）
P4719 【模板】"动态 DP"&动态树分治
P5055 【模板】可持久化文艺平衡树
P5236 【模板】静态仙人掌
P5394 【模板】下降幂多项式乘法
P5496 【模板】回文自动机（PAM）
P5494 【模板】线段树分裂
P6136 【模板】普通平衡树（数据加强版）
P6114 【模板】Lyndon 分解
P5807 【模板】BEST 定理 / Which Dreamed It
P6139 【模板】广义后缀自动机（广义 SAM）
P6329 【模板】点分树 | 震波
P4213 【模板】杜教筛（Sum）
P4717 【模板】快速莫比乌斯/沃尔什变换 (FMT/FWT)
P5043 【模板】树同构 / [BJOI2015] 树的同构
P5192 【模板】有源汇上下界最大流 / Zoj3229 Shoot the Bullet|东方文花帖|
P5491 【模板】二次剩余
P3804 【模板】后缀自动机 (SAM)
P3810 【模板】三维偏序（陌上花开）
P4526 【模板】自适应辛普森法2
P4777 【模板】扩展中国剩余定理（EXCRT）
P4716 【模板】最小树形图
P4718 【模板】Pollard-Rho算法
P4725 【模板】多项式对数函数（多项式 ln）
P4929 【模板】舞蹈链（DLX）
P5787 【模板】线段树分治 / 二分图
P6113 【模板】一般图最大匹配
P3835 【模板】可持久化平衡树
P4195 【模板】扩展 BSGS/exBSGS
P6097 【模板】子集卷积
P6178 【模板】Matrix-Tree 定理
P6164 【模板】后缀平衡树
P6242 【模板】线段树 3
P3806 【模板】点分治1
P4556 【模板】线段树合并 / [Vani有约会]雨天的尾巴
P4751 【模板】"动态DP"&动态树分治（加强版）
P4721 【模板】分治 FFT
P5410 【模板】扩展 KMP（Z 函数）
P5829 【模板】失配树
P6192 【模板】最小斯坦纳树
P4720 【模板】扩展卢卡斯定理/exLucas
P3690 【模板】动态树（Link Cut Tree）
P4722 【模板】最大流 加强版 / 预流推进
P3380 【模板】二逼平衡树（树套树）
P4245 【模板】任意模数多项式乘法
P7173 【模板】有负圈的费用流
P5245 【模板】多项式快速幂
P4238 【模板】多项式乘法逆
P4897 【模板】最小割树（Gomory-Hu Tree）
P4980 【模板】Pólya 定理
P6657 【模板】LGV 引理
P5205 【模板】多项式开根
P4512 【模板】多项式除法
P5277 【模板】多项式开根（加强版）
P4726 【模板】多项式指数函数（多项式 exp）
P5273 【模板】多项式幂函数 (加强版)
P6577 【模板】二分图最大权完美匹配
P6800 【模板】Chirp Z-Transform
NOI / NOI+ / CTSC
P4724 【模板】三维凸包
P5180 【模板】支配树
P4723 【模板】常系数齐次线性递推
P2483 【模板】k短路 / [SDOI2010]魔法猪学院
P5050 【模板】多项式多点求值
P5282 【模板】快速阶乘算法
P5325 【模板】Min_25筛
P5247 【模板】动态图完全连通性
P5809 【模板】多项式复合逆
P5056 【模板】插头dp
P5373 【模板】多项式复合函数
P5808 【模板】常系数非齐次线性递推
P5487 【模板】Berlekamp-Massey算法
P5668 【模板】N次剩余
P5158 【模板】多项式快速插值
P4887 【模板】莫队二次离线（第十四分块(前体)）
P5170 【模板】类欧几里得算法
P6699 【模板】一般图最大权匹配
P6656 【模板】Runs
P6115 【模板】整式递推
