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
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::io;
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

