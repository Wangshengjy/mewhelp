-- =============================================================
-- ch07 · 会话上下文管理 · 建表 DDL(第一步,共两步)
-- 这一步给 conversations 表加两列:一段投影用的摘要,和摘要覆盖到哪条的锚点。
-- 分层要用的另一个锚点和分段摘要表在 ch07-layers.sql,两个文件都要 apply。
--
-- 仓库差异:conversations.id 本仓库是 VARCHAR(36) UUID(课程参考是 BIGINT),
-- 锚点两列仍用 BIGINT 对齐 messages.id(INT 自增)。
-- apply:容器 13306,凭据读 .env;无 mysql client 时用 python+pymysql 执行。
-- =============================================================

SET NAMES utf8mb4;

ALTER TABLE conversations
  ADD COLUMN summary             TEXT            NULL COMMENT '最近几段梗概拼成的投影,拼装时跟证据一起挂在用户那句之后' AFTER status,
  ADD COLUMN summary_upto_msg_id BIGINT UNSIGNED NULL COMMENT '摘要已覆盖到哪条消息,滑窗从其后接原文' AFTER summary;
