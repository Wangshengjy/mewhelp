"""ch08 · 自建业务 MCP Server(顶层独立进程,Streamable HTTP)。

物流(:8101)查轨迹、售后(:8102)查在保/退货进度。Server 内随机 mock 数据,
不接真实系统、不建表。起停:python scripts/mcp_up.py / scripts/mcp_down.py。
"""
