---
name: local-scene-recognition
description: |
  本地启蒙学习助手。这是一款完全本地运行、基于摄像头的幼儿学习助手：将网络摄像头对准闪卡、单词卡、算术卡、美国硬币、玩具或物体，它会给出简短的语音回答，例如朗读英语单词、解答算术卡，或识别并计算硬币总额。

  当用户提到摄像头画面中的物品，并希望进行启蒙、词汇、算术或硬币练习时使用此技能，例如“启蒙/闪卡/单词卡/识字卡/算术卡/这个用英语怎么说/这是多少钱/帮我算一下这张卡片”，或 flashcard/word card/what is this in English/teach me this word/add up these coins/how much money is this。
---

# local-scene-recognition

## 运行约束

- 使用 **PowerShell** 环境来运行 `scripts\run.ps1`。
- 不要从一个 shell 启动另一个 shell。
- 调用 `run.ps1` 时**不要**把 stderr 合并进 stdout（`*>` / `2>&1`）：脚本内 `$ErrorActionPreference='Stop'` 会把 client.py 写往 stderr 的 INFO 日志判为致命错误并 exit 99。
- `run.ps1` 退出既任务完成, 请查看其退出码以确认任务是否成功, 并根据打印的日志形成报告, **不要反复调用且无需验证结果**。

## 使用

- PowerShell: `scripts\run.ps1`
- 前端退出不影响后台服务的运行, 使用 `--continue` 选项可以继续与后台服务的交互。
- 可选参数包括 `--language`、`--no-tts` 和 `--no-fast-mode`；使用 `--status`、`--stop` 和 `--restart` 管理实时服务。

这是需要跨请求保持设备和会话状态的实时交互 skill，因此**必须保留后台服务, 请勿主动关闭后台进程**，启动完成时通知用户使用类似"停止服务"的指示来关闭后台服务。
首次启动如需下载模型，应保持命令运行。

硬件不受支持或命令失败时，应把原始错误反馈给用户，不要改用云端服务。
