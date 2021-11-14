---
title: LeetCode 周赛 263
date: 2021-11-14 23:22:04
tags: LeetCode 周赛总结
---

--------

# LeetCode 周赛 263

| 排名        | 用户名       | 得分 | 完成时间 | [题目1 (3)](https://leetcode-cn.com/contest/weekly-contest-267/problems/time-needed-to-buy-tickets/) | [题目2 (4)](https://leetcode-cn.com/contest/weekly-contest-267/problems/reverse-nodes-in-even-length-groups/) | [题目3 (5)](https://leetcode-cn.com/contest/weekly-contest-267/problems/decode-the-slanted-ciphertext/) | [题目4 (6)](https://leetcode-cn.com/contest/weekly-contest-267/problems/process-restricted-friend-requests/) |
| ----------- | ------------ | ---- | -------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 1279 / 4364 | MiracleSNeko | 8    | 1:10:34  | 0:07:14 2                                                    |                                                              | 1:00:34                                                      |                                                              |

>   以后不强行要求的时候再原地玩链表我就是伞兵

## T1 5926. 买票需要的时间

-   **通过的用户数**3160
-   **尝试过的用户数**3336
-   **用户总通过次数**3204
-   **用户总提交次数**5384
-   **题目难度** **Easy**

有 `n` 个人前来排队买票，其中第 `0` 人站在队伍 **最前方** ，第 `(n - 1)` 人站在队伍 **最后方** 。

给你一个下标从 **0** 开始的整数数组 `tickets` ，数组长度为 `n` ，其中第 `i` 人想要购买的票数为 `tickets[i]` 。

每个人买票都需要用掉 **恰好 1 秒** 。一个人 **一次只能买一张票** ，如果需要购买更多票，他必须走到 **队尾** 重新排队（**瞬间** 发生，不计时间）。如果一个人没有剩下需要买的票，那他将会 **离开** 队伍。

返回位于位置 `k`（下标从 **0** 开始）的人完成买票需要的时间（以秒为单位）。

**示例 1：**

```
输入：tickets = [2,3,2], k = 2
输出：6
解释： 
- 第一轮，队伍中的每个人都买到一张票，队伍变为 [1, 2, 1] 。
- 第二轮，队伍中的每个都又都买到一张票，队伍变为 [0, 1, 0] 。
位置 2 的人成功买到 2 张票，用掉 3 + 3 = 6 秒。
```

**示例 2：**

```
输入：tickets = [5,1,1,1], k = 0
输出：8
解释：
- 第一轮，队伍中的每个人都买到一张票，队伍变为 [4, 0, 0, 0] 。
- 接下来的 4 轮，只有位置 0 的人在买票。
位置 0 的人成功买到 5 张票，用掉 4 + 1 + 1 + 1 + 1 = 8 秒。 
```

**提示：**

-   `n == tickets.length`
-   `1 <= n <= 100`
-   `1 <= tickets[i] <= 100`
-   `0 <= k < n`

**提交：**

>   目标后面的 >= 应该少算一次，吃 WA

```rust
impl Solution {
    pub fn time_required_to_buy(tickets: Vec<i32>, k: i32) -> i32 {
        let kth = tickets[k as usize];
        tickets.iter().filter(|&&i| i < kth).sum::<i32>() + kth * tickets.iter().filter(|&&i| i >= kth).count() as i32 - tickets.iter().enumerate().filter(|&(id, i)| id > k as usize && *i >= kth).count() as i32
    }
}
```

## T2 5927. 反转偶数长度组的节点

-   **通过的用户数**1679
-   **尝试过的用户数**2097
-   **用户总通过次数**1699
-   **用户总提交次数**5004
-   **题目难度****Medium**

给你一个链表的头节点 `head` 。

链表中的节点 **按顺序** 划分成若干 **非空** 组，这些非空组的长度构成一个自然数序列（`1, 2, 3, 4, ...`）。一个组的 **长度** 就是组中分配到的节点数目。换句话说：

-   节点 `1` 分配给第一组
-   节点 `2` 和 `3` 分配给第二组
-   节点 `4`、`5` 和 `6` 分配给第三组，以此类推

注意，最后一组的长度可能小于或者等于 `1 + 倒数第二组的长度` 。

**反转** 每个 **偶数** 长度组中的节点，并返回修改后链表的头节点 `head` 。

 

**示例 1：**

![img](https://assets.leetcode.com/uploads/2021/10/25/eg1.png)

```
输入：head = [5,2,6,3,9,1,7,3,8,4]
输出：[5,6,2,3,9,1,4,8,3,7]
解释：
- 第一组长度为 1 ，奇数，没有发生反转。
- 第二组长度为 2 ，偶数，节点反转。
- 第三组长度为 3 ，奇数，没有发生反转。
- 最后一组长度为 4 ，偶数，节点反转。
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/10/25/eg2.png)

```
输入：head = [1,1,0,6]
输出：[1,0,1,6]
解释：
- 第一组长度为 1 ，没有发生反转。
- 第二组长度为 2 ，节点反转。
- 最后一组长度为 1 ，没有发生反转。
```

**示例 3：**

![img](https://assets.leetcode.com/uploads/2021/10/28/eg3.png)

```
输入：head = [2,1]
输出：[2,1]
解释：
- 第一组长度为 1 ，没有发生反转。
- 最后一组长度为 1 ，没有发生反转。
```

**示例 4：**

```
输入：head = [8]
输出：[8]
解释：只有一个长度为 1 的组，没有发生反转。
```

 

**提示：**

-   链表中节点数目范围是 `[1, 105]`
-   `0 <= Node.val <= 105`

**答案：**

>   等我抽空给 `Option<Box<ListNode>>` 写个封装和 IntoIter。我再原地就是伞兵

```rust

```

## T3 5928. 解码斜向换位密码

-   **通过的用户数**1380
-   **尝试过的用户数**1575
-   **用户总通过次数**1412
-   **用户总提交次数**3238
-   **题目难度** **Medium**

字符串 `originalText` 使用 **斜向换位密码** ，经由 **行数固定** 为 `rows` 的矩阵辅助，加密得到一个字符串 `encodedText` 。

`originalText` 先按从左上到右下的方式放置到矩阵中。

![img](https://assets.leetcode.com/uploads/2021/11/07/exa11.png)

先填充蓝色单元格，接着是红色单元格，然后是黄色单元格，以此类推，直到到达 `originalText` 末尾。箭头指示顺序即为单元格填充顺序。所有空单元格用 `' '` 进行填充。矩阵的列数需满足：用 `originalText` 填充之后，最右侧列 **不为空** 。

接着按行将字符附加到矩阵中，构造 `encodedText` 。

![img](https://assets.leetcode.com/uploads/2021/11/07/exa12.png)

先把蓝色单元格中的字符附加到 `encodedText` 中，接着是红色单元格，最后是黄色单元格。箭头指示单元格访问顺序。

例如，如果 `originalText = "cipher"` 且 `rows = 3` ，那么我们可以按下述方法将其编码：

![img](https://assets.leetcode.com/uploads/2021/10/25/desc2.png)

蓝色箭头标识 `originalText` 是如何放入矩阵中的，红色箭头标识形成 `encodedText` 的顺序。在上述例子中，`encodedText = "ch  ie  pr"` 。

给你编码后的字符串 `encodedText` 和矩阵的行数 `rows` ，返回源字符串 `originalText` 。

**注意：**`originalText` **不** 含任何尾随空格 `' '` 。生成的测试用例满足 **仅存在一个** 可能的 `originalText` 。

 

**示例 1：**

```
输入：encodedText = "ch   ie   pr", rows = 3
输出："cipher"
解释：此示例与问题描述中的例子相同。
```

**示例 2：**

![img](https://assets.leetcode.com/uploads/2021/10/26/exam1.png)

```
输入：encodedText = "iveo    eed   l te   olc", rows = 4
输出："i love leetcode"
解释：上图标识用于编码 originalText 的矩阵。 
蓝色箭头展示如何从 encodedText 找到 originalText 。
```

**示例 3：**

![img](https://assets.leetcode.com/uploads/2021/10/26/eg2.png)

```
输入：encodedText = "coding", rows = 1
输出："coding"
解释：由于只有 1 行，所以 originalText 和 encodedText 是相同的。
```

**示例 4：**

![img](https://assets.leetcode.com/uploads/2021/10/26/exam3.png)

```
输入：encodedText = " b  ac", rows = 2
输出：" abc"
解释：originalText 不能含尾随空格，但它可能会有一个或者多个前置空格。
```

 

**提示：**

-   `0 <= encodedText.length <= 106`
-   `encodedText` 仅由小写英文字母和 `' '` 组成
-   `encodedText` 是对某个 **不含** 尾随空格的 `originalText` 的一个有效编码
-   `1 <= rows <= 1000`
-   生成的测试用例满足 **仅存在一个** 可能的 `originalText`

**提交：**

>   推下公式很容易发现，rows 个一组，每组两个相邻字母隔着 cols + 1，去掉末尾空格就行了

```rust
impl Solution {
    pub fn decode_ciphertext(encoded_text: String, rows: i32) -> String {
        if rows == 1 {
            encoded_text
        } else {
            let rows = rows as usize;
            let cols = encoded_text.len() / rows;
            let mut text = Vec::<u8>::new();
            let enc = encoded_text.bytes().collect::<Vec<_>>();
            let len = enc.len();
            let mut i = 0usize;
            let mut r = 0usize;
            while i + r * (cols + 1) < len {
                text.push(enc[i + r * (cols + 1)]);
                i += (r + 1) / rows;
                r = (r + 1) % rows;
            }
            text = text.into_iter().rev().skip_while(|&ch| ch == b' ').collect();
            text = text.into_iter().rev().collect();
            String::from_utf8(text).unwrap_or(String::from(""))
        }
    }
}
```

## T4 5929. 处理含限制条件的好友请求

-   **通过的用户数**452
-   **尝试过的用户数**663
-   **用户总通过次数**493
-   **用户总提交次数**1216
-   **题目难度** **Hard**

给你一个整数 `n` ，表示网络上的用户数目。每个用户按从 `0` 到 `n - 1` 进行编号。

给你一个下标从 **0** 开始的二维整数数组 `restrictions` ，其中 `restrictions[i] = [xi, yi]` 意味着用户 `xi` 和用户 `yi` **不能** 成为 **朋友** ，不管是 **直接** 还是通过其他用户 **间接** 。

最初，用户里没有人是其他用户的朋友。给你一个下标从 **0** 开始的二维整数数组 `requests` 表示好友请求的列表，其中 `requests[j] = [uj, vj]` 是用户 `uj` 和用户 `vj` 之间的一条好友请求。

如果 `uj` 和 `vj` 可以成为 **朋友** ，那么好友请求将会 **成功** 。每个好友请求都会按列表中给出的顺序进行处理（即，`requests[j]` 会在 `requests[j + 1]` 前）。一旦请求成功，那么对所有未来的好友请求而言， `uj` 和 `vj` 将会 **成为直接朋友 。**

返回一个 **布尔数组** `result` ，其中元素遵循此规则：如果第 `j` 个好友请求 **成功** ，那么 `result[j]` 就是 `true` ；否则，为 `false` 。

**注意：**如果 `uj` 和 `vj` 已经是直接朋友，那么他们之间的请求将仍然 **成功** 。

 

**示例 1：**

```
输入：n = 3, restrictions = [[0,1]], requests = [[0,2],[2,1]]
输出：[true,false]
解释：
请求 0 ：用户 0 和 用户 2 可以成为朋友，所以他们成为直接朋友。 
请求 1 ：用户 2 和 用户 1 不能成为朋友，因为这会使 用户 0 和 用户 1 成为间接朋友 (1--2--0) 。
```

**示例 2：**

```
输入：n = 3, restrictions = [[0,1]], requests = [[1,2],[0,2]]
输出：[true,false]
解释：
请求 0 ：用户 1 和 用户 2 可以成为朋友，所以他们成为直接朋友。 
请求 1 ：用户 0 和 用户 2 不能成为朋友，因为这会使 用户 0 和 用户 1 成为间接朋友 (0--2--1) 。
```

**示例 3：**

```
输入：n = 5, restrictions = [[0,1],[1,2],[2,3]], requests = [[0,4],[1,2],[3,1],[3,4]]
输出：[true,false,true,false]
解释：
请求 0 ：用户 0 和 用户 4 可以成为朋友，所以他们成为直接朋友。 
请求 1 ：用户 1 和 用户 2 不能成为朋友，因为他们之间存在限制。
请求 2 ：用户 3 和 用户 1 可以成为朋友，所以他们成为直接朋友。 
请求 3 ：用户 3 和 用户 4 不能成为朋友，因为这会使 用户 0 和 用户 1 成为间接朋友 (0--4--3--1) 。
```

 

**提示：**

-   `2 <= n <= 1000`
-   `0 <= restrictions.length <= 1000`
-   `restrictions[i].length == 2`
-   `0 <= xi, yi <= n - 1`
-   `xi != yi`
-   `1 <= requests.length <= 1000`
-   `requests[j].length == 2`
-   `0 <= uj, vj <= n - 1`
-   `uj != vj`

**思路：**

并查集。吃饭去了没多想就没写，抽空补上。
