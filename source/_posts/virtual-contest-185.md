---
title: LeetCode 虚拟周赛 185
date: 2021-11-17 23:18:03
tags: LeetCode 周赛总结
---

-----

# LeetCode 虚拟周赛 185

## T1 1417. 重新格式化字符串

-   **通过的用户数**3297
-   **尝试过的用户数**3450
-   **用户总通过次数**3441
-   **用户总提交次数**6732
-   **题目难度** **Easy**

给你一个混合了数字和字母的字符串 `s`，其中的字母均为小写英文字母。

请你将该字符串重新格式化，使得任意两个相邻字符的类型都不同。也就是说，字母后面应该跟着数字，而数字后面应该跟着字母。

请你返回 **重新格式化后** 的字符串；如果无法按要求重新格式化，则返回一个 **空字符串** 。

**示例 1：**

```
输入：s = "a0b1c2"
输出："0a1b2c"
解释："0a1b2c" 中任意两个相邻字符的类型都不同。 "a0b1c2", "0a1b2c", "0c2a1b" 也是满足题目要求的答案。
```

**示例 2：**

```
输入：s = "leetcode"
输出：""
解释："leetcode" 中只有字母，所以无法满足重新格式化的条件。
```

**示例 3：**

```
输入：s = "1229857369"
输出：""
解释："1229857369" 中只有数字，所以无法满足重新格式化的条件。
```

**示例 4：**

```
输入：s = "covid2019"
输出："c2o0v1i9d"
```

**示例 5：**

```
输入：s = "ab123"
输出："1a2b3"
```

**提示：**

-   `1 <= s.length <= 500`
-   `s` 仅由小写英文字母和/或数字组成。

**提交：**

```Rust
impl Solution {
    pub fn reformat(s: String) -> String {
        let ch = s.bytes().filter(|ch| ch.is_ascii_alphabetic()).collect::<Vec<_>>();
        let dig = s.bytes().filter(|ch| ch.is_ascii_digit()).collect::<Vec<_>>();
        if (ch.len() as i32 - dig.len() as i32).abs() > 1 {
            String::from("")
        } else {
            let mut s = (0..ch.len().min(dig.len())).fold(Vec::new(), |mut s, i| {
                s.push(ch[i]);
                s.push(dig[i]);
                s
            });
            if ch.len() > dig.len() {
                s.push(ch[ch.len() - 1]);
            } else if ch.len() < dig.len() {
                s.insert(0, dig[dig.len() - 1]);
            }
            String::from_utf8(s).unwrap()
        }
    }
}
```

## T2 1418. 点菜展示表

-   **通过的用户数**1861
-   **尝试过的用户数**2042
-   **用户总通过次数**1885
-   **用户总提交次数**3567
-   **题目难度** **Medium**

给你一个数组 `orders`，表示客户在餐厅中完成的订单，确切地说， `orders[i]=[customerNamei,tableNumberi,foodItemi]` ，其中 `customerNamei` 是客户的姓名，`tableNumberi` 是客户所在餐桌的桌号，而 `foodItemi` 是客户点的餐品名称。

请你返回该餐厅的 **点菜展示表** *。*在这张表中，表中第一行为标题，其第一列为餐桌桌号 “Table” ，后面每一列都是按字母顺序排列的餐品名称。接下来每一行中的项则表示每张餐桌订购的相应餐品数量，第一列应当填对应的桌号，后面依次填写下单的餐品数量。

注意：客户姓名不是点菜展示表的一部分。此外，表中的数据行应该按餐桌桌号升序排列。

**示例 1：**

```
输入：orders = [["David","3","Ceviche"],["Corina","10","Beef Burrito"],["David","3","Fried Chicken"],["Carla","5","Water"],["Carla","5","Ceviche"],["Rous","3","Ceviche"]]
输出：[["Table","Beef Burrito","Ceviche","Fried Chicken","Water"],["3","0","2","1","0"],["5","0","1","0","1"],["10","1","0","0","0"]] 
解释：
点菜展示表如下所示：
Table,Beef Burrito,Ceviche,Fried Chicken,Water
3    ,0           ,2      ,1            ,0
5    ,0           ,1      ,0            ,1
10   ,1           ,0      ,0            ,0
对于餐桌 3：David 点了 "Ceviche" 和 "Fried Chicken"，而 Rous 点了 "Ceviche"
而餐桌 5：Carla 点了 "Water" 和 "Ceviche"
餐桌 10：Corina 点了 "Beef Burrito" 
```

**示例 2：**

```
输入：orders = [["James","12","Fried Chicken"],["Ratesh","12","Fried Chicken"],["Amadeus","12","Fried Chicken"],["Adam","1","Canadian Waffles"],["Brianna","1","Canadian Waffles"]]
输出：[["Table","Canadian Waffles","Fried Chicken"],["1","2","0"],["12","0","3"]] 
解释：
对于餐桌 1：Adam 和 Brianna 都点了 "Canadian Waffles"
而餐桌 12：James, Ratesh 和 Amadeus 都点了 "Fried Chicken"
```

**示例 3：**

```
输入：orders = [["Laura","2","Bean Burrito"],["Jhon","2","Beef Burrito"],["Melissa","2","Soda"]]
输出：[["Table","Bean Burrito","Beef Burrito","Soda"],["2","1","1","1"]] 
```

**提示：**

-   `1 <= orders.length <= 5 * 10^4`
-   `orders[i].length == 3`
-   `1 <= customerNamei.length, foodItemi.length <= 20`
-   `customerNamei` 和 `foodItemi` 由大小写英文字母及空格字符 `' '` 组成。
-   `tableNumberi` 是 `1` 到 `500` 范围内的整数。

**题解：**

>   不难，就是恶心，写了半个小时困得扛不住就一键放弃了

```C#
public List<List<String>> displayTable(List<List<String>> orders) {
        //构造表头
        List<String> header = orders.stream()
                .map(order -> order.get(2))
                .distinct()
                .sorted()
                .collect(Collectors.toCollection(() -> Stream.of("Table").collect(Collectors.toList())));

        //填写数据
        return orders.stream()
                .collect(Collectors.groupingBy(
                        order -> Integer.valueOf(order.get(1)),
                        (Supplier<Map<Integer, Map<String, Integer>>>) TreeMap::new,
                        Collectors.groupingBy(order -> order.get(2), Collectors.summingInt(e -> 1)))
                ).entrySet()
                .stream()
                .peek(entry -> entry.getValue().put("Table", entry.getKey()))
                .map(entry -> header.stream()
                        .map(column -> entry.getValue().getOrDefault(column, 0).toString())
                        .collect(Collectors.toList()))
                .collect(Collectors.toCollection(() -> Stream.of(header).collect(Collectors.toList())));
    }
```

>   干脆出道手写 SQL 数据库得了

## L3 1419. 数青蛙

-   **通过的用户数**1087
-   **尝试过的用户数**1882
-   **用户总通过次数**1105
-   **用户总提交次数**4362
-   **题目难度** **Medium**

给你一个字符串 `croakOfFrogs`，它表示不同青蛙发出的蛙鸣声（字符串 "croak" ）的组合。由于同一时间可以有多只青蛙呱呱作响，所以 `croakOfFrogs` 中会混合多个 “croak” *。*请你返回模拟字符串中所有蛙鸣所需不同青蛙的最少数目。

**注意：**要想发出蛙鸣 "croak"，青蛙必须 **依序** 输出 `‘c’, ’r’, ’o’, ’a’, ’k’` 这 5 个字母。如果没有输出全部五个字母，那么它就不会发出声音。

如果字符串 `croakOfFrogs` 不是由若干有效的 "croak" 字符混合而成，请返回 `-1` 。

**示例 1：**

```
输入：croakOfFrogs = "croakcroak"
输出：1 
解释：一只青蛙 “呱呱” 两次
```

**示例 2：**

```
输入：croakOfFrogs = "crcoakroak"
输出：2 
解释：最少需要两只青蛙，“呱呱” 声用黑体标注
第一只青蛙 "crcoakroak"
第二只青蛙 "crcoakroak"
```

**示例 3：**

```
输入：croakOfFrogs = "croakcrook"
输出：-1
解释：给出的字符串不是 "croak" 的有效组合。
```

**示例 4：**

```
输入：croakOfFrogs = "croakcroa"
输出：-1
```

**提示：**

-   `1 <= croakOfFrogs.length <= 10^5`
-   字符串中的字符只有 `'c'`, `'r'`, `'o'`, `'a'` 或者 `'k'`

**题解：**

>   状态机。记录在转移序列中的青蛙个数的最大值即可。如果状态转移任何一步出现问题，都应该返回 -1

```rust
impl Solution {
    pub fn min_number_of_frogs(croak_of_frogs: String) -> i32 {
        let mut stat = [0; 4];
        let mut cnt = 0;
        for ch in croak_of_frogs.bytes() {
            match ch {
                b'c' => {
                    stat[0] += 1;
                }
                b'r' => {
                    stat[0] -= 1;
                    stat[1] += 1;
                }
                b'o' => {
                    stat[1] -= 1;
                    stat[2] += 1;
                }
                b'a' => {
                    stat[2] -= 1;
                    stat[3] += 1;
                }
                b'k' => {
                    stat[3] -= 1;
                }
                _ => unreachable!()
            }
            if stat.iter().filter(|&&i| i < 0).count() != 0 {
                return -1;
            }
            cnt = cnt.max(stat.iter().sum::<i32>());
        }
        if stat.iter().filter(|&&i| i != 0).count() != 0 {
            return -1;
        }
        cnt
    }
}
```

## T4 1420. 生成数组

-   **通过的用户数**181
-   **尝试过的用户数**292
-   **用户总通过次数**200
-   **用户总提交次数**417
-   **题目难度** **Hard**

给你三个整数 `n`、`m` 和 `k` 。下图描述的算法用于找出正整数数组中最大的元素。

![img](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2020/04/19/e.png)

请你生成一个具有下述属性的数组 `arr` ：

-   `arr` 中有 `n` 个整数。
-   `1 <= arr[i] <= m` 其中 `(0 <= i < n)` 。
-   将上面提到的算法应用于 `arr` ，`search_cost` 的值等于 `k` 。

返回上述条件下生成数组 `arr` 的 **方法数** ，由于答案可能会很大，所以 **必须** 对 `10^9 + 7` 取余。

**示例 1：**

```
输入：n = 2, m = 3, k = 1
输出：6
解释：可能的数组分别为 [1, 1], [2, 1], [2, 2], [3, 1], [3, 2] [3, 3]
```

**示例 2：**

```
输入：n = 5, m = 2, k = 3
输出：0
解释：没有数组可以满足上述条件
```

**示例 3：**

```
输入：n = 9, m = 1, k = 1
输出：1
解释：可能的数组只有 [1, 1, 1, 1, 1, 1, 1, 1, 1]
```

**示例 4：**

```
输入：n = 50, m = 100, k = 25
输出：34549172
解释：不要忘了对 1000000007 取余
```

**示例 5：**

```
输入：n = 37, m = 17, k = 7
输出：418930126
```

**提示：**

-   `1 <= n <= 50`
-   `1 <= m <= 100`
-   `0 <= k <= n`

**题解：**

>   [动态规划]([生成数组 - 生成数组 - 力扣（LeetCode） (leetcode-cn.com)](https://leetcode-cn.com/problems/build-array-where-you-can-find-the-maximum-exactly-k-comparisons/solution/sheng-cheng-shu-zu-by-leetcode-solution-yswf/))

