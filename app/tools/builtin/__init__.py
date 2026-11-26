"""ch08 · 内置工具包:每模块 = @tool 实现 + registry.register(...)。

注册即定义:服务启动(lifespan)时 registry.scan_builtin() 扫描导入整包完成登记;
新工具 = 往本包丢一个文件,重启服务即被注册,核心代码零改动。
运行期依赖(session_factory/settings/conversation_id/tool_call_id)一律声明成
InjectedToolArg 注入参数——模型可见口径(tool_call_schema)不含它们。
"""
