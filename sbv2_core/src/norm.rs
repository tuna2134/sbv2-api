use once_cell::sync::Lazy;
use std::collections::HashMap;

static REPLACE_MAP: Lazy<HashMap<&str, &str>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert("：", ",");
    map.insert("；", ",");
    map.insert("，", ",");
    map.insert("。", ".");
    map.insert("！", "!");
    map.insert("？", "?");
    map.insert("\n", ".");
    map.insert("．", ".");
    map.insert("…", "...");
    map.insert("···", "...");
    map.insert("・・・", "...");
    map.insert("·", ",");
    map.insert("・", ",");
    map.insert("、", ",");
    map.insert("$", ".");
    map.insert("“", "'");
    map.insert("”", "'");
    map.insert("\"", "'");
    map.insert("‘", "'");
    map.insert("’", "'");
    map.insert("（", "'");
    map.insert("）", "'");
    map.insert("(", "'");
    map.insert(")", "'");
    map.insert("《", "'");
    map.insert("》", "'");
    map.insert("【", "'");
    map.insert("】", "'");
    map.insert("[", "'");
    map.insert("]", "'");
    // NFKC 正規化後のハイフン・ダッシュの変種を全て通常半角ハイフン - \u002d に変換
    map.insert("\u{02d7}", "\u{002d}"); // ˗, Modifier Letter Minus Sign
    map.insert("\u{2010}", "\u{002d}"); // ‐, Hyphen,
    map.insert("\u{2012}", "\u{002d}"); // ‒, Figure Dash
    map.insert("\u{2013}", "\u{002d}"); // –, En Dash
    map.insert("\u{2014}", "\u{002d}"); // —, Em Dash
    map.insert("\u{2015}", "\u{002d}"); // ―, Horizontal Bar
    map.insert("\u{2043}", "\u{002d}"); // ⁃, Hyphen Bullet
    map.insert("\u{2212}", "\u{002d}"); // −, Minus Sign
    map.insert("\u{23af}", "\u{002d}"); // ⎯, Horizontal Line Extension
    map.insert("\u{23e4}", "\u{002d}"); // ⏤, Straightness
    map.insert("\u{2500}", "\u{002d}"); // ─, Box Drawings Light Horizontal
    map.insert("\u{2501}", "\u{002d}"); // ━, Box Drawings Heavy Horizontal
    map.insert("\u{2e3a}", "\u{002d}"); // ⸺, Two-Em Dash
    map.insert("\u{2e3b}", "\u{002d}"); // ⸻, Three-Em Dash
    map.insert("「", "'");
    map.insert("」", "'");

    map
});
