# player-classic —— 2026-09-28 那天播放器的原样备份

这一份是**分镜查看器**版本的播放器，原样拷贝，没有改过一个字节。

**为什么留着**：同一天 `frontend/player/` 被改成了视频播放器——一条能拖的进度条、暂停、音量、全屏、下载，
拆掉了顶上那一行幕数/教学目标、右边一整列「教学节拍」、以及 ◀ ▶ 重置三个按键。
查看器那套东西（逐拍列表、跨幕单步、`?scene=N&step=N` 停在某一帧）以后可能还想用，所以留一份。

**它不是活的代码**：

- **不会被服务**。`api.py` 只挂了 `/player`、`/frontend/player`、`/data` 和 `/`，这里没有挂载点，
  打开 `/player-classic/index.html` 是 404。
- **不进测试**。`tests/unit/test_player_theme.py` 和 `test_render_registry.py` 是
  `frontend/player/*.js` 的 glob，隔壁目录扫不到。这正是它必须放在**外面**的原因：
  放进去会让「全项目只有一处 `fetch(`」和「不许写死颜色」两条检查凭空变红。
- **不会跟着上游改动**。`player-classic` 里的任何模块都和 `frontend/player/` 断开了，
  改那边不会同步过来。这份是快照，不是分支。

**想删就删**：整个目录直接删掉，不影响任何东西。
