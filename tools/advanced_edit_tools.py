"""
Advanced Code Editing Tools - Production-Grade Implementation

这是产品级代码编辑工具的设计，参考了 Cursor、Claude Code、SWE-agent 的实现。

核心理念：
- str_replace 精确替换，而不是整文件覆写
- 支持diff应用
- 支持行级操作
- 完整的编辑历史和撤销能力
"""

import os
import re
import difflib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from langchain_core.tools import tool


# ============================================================================
# EDIT HISTORY MANAGEMENT
# ============================================================================

@dataclass
class EditRecord:
    """单次编辑记录"""
    file_path: str
    operation: str  # str_replace, insert, delete, apply_diff
    timestamp: str
    before_content: str
    after_content: str
    details: Dict[str, Any] = field(default_factory=dict)


class EditHistory:
    """编辑历史管理器，支持撤销操作"""
    
    def __init__(self, max_history: int = 100):
        self.history: List[EditRecord] = []
        self.max_history = max_history
    
    def record(self, record: EditRecord):
        """记录一次编辑"""
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def undo_last(self, file_path: str = None) -> Optional[EditRecord]:
        """撤销最近的编辑"""
        for i in range(len(self.history) - 1, -1, -1):
            if file_path is None or self.history[i].file_path == file_path:
                record = self.history.pop(i)
                # 恢复原内容
                Path(record.file_path).write_text(record.before_content, encoding='utf-8')
                return record
        return None
    
    def get_history(self, file_path: str = None, limit: int = 10) -> List[EditRecord]:
        """获取编辑历史"""
        if file_path:
            records = [r for r in self.history if r.file_path == file_path]
        else:
            records = self.history
        return records[-limit:]


# 全局编辑历史
_edit_history = EditHistory()


# ============================================================================
# CORE EDITING TOOLS
# ============================================================================

@tool
def str_replace(
    file_path: str,
    old_str: str,
    new_str: str,
    occurrence: int = 1
) -> str:
    """
    精确替换文件中的字符串 - 这是产品级代码编辑的核心工具。
    
    为什么使用str_replace而不是write_file：
    1. Token效率：只发送需要修改的部分，而不是整个文件
    2. 精确性：必须完全匹配才会替换，避免意外修改
    3. 可审计：清楚记录了修改前后的内容
    4. 可撤销：保留历史记录，支持撤销
    
    Args:
        file_path: 文件路径
        old_str: 要替换的原始字符串（必须完全匹配，包括空格和缩进）
        new_str: 替换后的新字符串
        occurrence: 替换第几个匹配（1表示第一个，0表示全部替换）
        
    Returns:
        JSON格式的结果，包含成功/失败状态和详细信息
        
    Example:
        # 修复函数名拼写错误
        str_replace(
            "utils.py",
            old_str="def caculate_total(items):",
            new_str="def calculate_total(items):"
        )
        
        # 添加导入语句（在现有import后）
        str_replace(
            "main.py",
            old_str="import os",
            new_str="import os\\nimport json"
        )
    
    IMPORTANT:
        - old_str 必须完全匹配文件中的内容，包括空格、换行符、缩进
        - 如果匹配失败，会返回文件中最相似的片段供参考
        - 建议在使用前先用 read_file_with_lines 查看确切内容
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({
                "success": False,
                "error": f"文件不存在: {file_path}"
            })
        
        original_content = path.read_text(encoding='utf-8')
        
        # 检查old_str是否存在
        count = original_content.count(old_str)
        
        if count == 0:
            # 找到最相似的片段帮助用户调试
            similar = _find_similar_text(original_content, old_str)
            return json.dumps({
                "success": False,
                "error": "未找到匹配的文本",
                "hint": "old_str必须完全匹配文件内容，包括空格和缩进",
                "similar_text": similar,
                "suggestion": "请使用 read_file_with_lines 查看文件确切内容"
            }, ensure_ascii=False, indent=2)
        
        # 执行替换
        if occurrence == 0:
            # 替换所有
            new_content = original_content.replace(old_str, new_str)
            replaced_count = count
        else:
            # 替换指定的第N个
            if occurrence > count:
                return json.dumps({
                    "success": False,
                    "error": f"只找到 {count} 个匹配，但要求替换第 {occurrence} 个"
                })
            
            # 找到第N个匹配的位置
            pos = -1
            for i in range(occurrence):
                pos = original_content.find(old_str, pos + 1)
            
            new_content = (
                original_content[:pos] + 
                new_str + 
                original_content[pos + len(old_str):]
            )
            replaced_count = 1
        
        # 保存文件
        path.write_text(new_content, encoding='utf-8')
        
        # 记录历史
        _edit_history.record(EditRecord(
            file_path=str(path.absolute()),
            operation="str_replace",
            timestamp=datetime.now().isoformat(),
            before_content=original_content,
            after_content=new_content,
            details={
                "old_str": old_str[:100] + "..." if len(old_str) > 100 else old_str,
                "new_str": new_str[:100] + "..." if len(new_str) > 100 else new_str,
                "replaced_count": replaced_count
            }
        ))
        
        # 生成diff预览
        diff = _generate_diff(original_content, new_content, file_path)
        
        return json.dumps({
            "success": True,
            "file": file_path,
            "replaced_count": replaced_count,
            "diff_preview": diff[:2000]  # 限制长度
        }, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


@tool
def insert_at_line(
    file_path: str,
    line_number: int,
    content: str,
    position: str = "after"
) -> str:
    """
    在指定行插入内容。
    
    Args:
        file_path: 文件路径
        line_number: 行号（1-indexed）
        content: 要插入的内容（可以是多行）
        position: "before" 在该行之前插入，"after" 在该行之后插入
        
    Returns:
        JSON格式的结果
        
    Example:
        # 在第5行后添加一个新函数
        insert_at_line(
            "utils.py",
            line_number=5,
            content="\\ndef new_helper():\\n    pass\\n",
            position="after"
        )
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"success": False, "error": f"文件不存在: {file_path}"})
        
        original_content = path.read_text(encoding='utf-8')
        lines = original_content.splitlines(keepends=True)
        
        # 验证行号
        if line_number < 1 or line_number > len(lines) + 1:
            return json.dumps({
                "success": False,
                "error": f"行号 {line_number} 超出范围 (1-{len(lines)})"
            })
        
        # 确保content以换行符结尾
        if content and not content.endswith('\n'):
            content += '\n'
        
        # 插入内容
        insert_index = line_number - 1 if position == "before" else line_number
        lines.insert(insert_index, content)
        
        new_content = ''.join(lines)
        path.write_text(new_content, encoding='utf-8')
        
        # 记录历史
        _edit_history.record(EditRecord(
            file_path=str(path.absolute()),
            operation="insert_at_line",
            timestamp=datetime.now().isoformat(),
            before_content=original_content,
            after_content=new_content,
            details={
                "line_number": line_number,
                "position": position,
                "content_preview": content[:100]
            }
        ))
        
        return json.dumps({
            "success": True,
            "file": file_path,
            "inserted_at": insert_index + 1,
            "lines_added": content.count('\n')
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def delete_lines(
    file_path: str,
    start_line: int,
    end_line: int
) -> str:
    """
    删除指定范围的行。
    
    Args:
        file_path: 文件路径
        start_line: 起始行号（1-indexed，包含）
        end_line: 结束行号（1-indexed，包含）
        
    Returns:
        JSON格式的结果
        
    Example:
        # 删除第10-15行
        delete_lines("main.py", start_line=10, end_line=15)
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"success": False, "error": f"文件不存在: {file_path}"})
        
        original_content = path.read_text(encoding='utf-8')
        lines = original_content.splitlines(keepends=True)
        
        # 验证行号
        if start_line < 1 or end_line > len(lines) or start_line > end_line:
            return json.dumps({
                "success": False,
                "error": f"无效的行范围: {start_line}-{end_line} (文件共 {len(lines)} 行)"
            })
        
        # 记录被删除的内容
        deleted_lines = lines[start_line-1:end_line]
        deleted_content = ''.join(deleted_lines)
        
        # 删除行
        new_lines = lines[:start_line-1] + lines[end_line:]
        new_content = ''.join(new_lines)
        
        path.write_text(new_content, encoding='utf-8')
        
        # 记录历史
        _edit_history.record(EditRecord(
            file_path=str(path.absolute()),
            operation="delete_lines",
            timestamp=datetime.now().isoformat(),
            before_content=original_content,
            after_content=new_content,
            details={
                "start_line": start_line,
                "end_line": end_line,
                "deleted_preview": deleted_content[:200]
            }
        ))
        
        return json.dumps({
            "success": True,
            "file": file_path,
            "deleted_lines": end_line - start_line + 1,
            "deleted_content_preview": deleted_content[:200]
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def apply_diff(
    file_path: str,
    diff_content: str
) -> str:
    """
    应用unified diff格式的补丁到文件。
    
    支持标准的unified diff格式，可以一次性应用多处修改。
    
    Args:
        file_path: 文件路径
        diff_content: unified diff格式的补丁内容
        
    Returns:
        JSON格式的结果
        
    Example:
        apply_diff("main.py", '''
--- a/main.py
+++ b/main.py
@@ -10,7 +10,7 @@
 def process_data(data):
-    result = data * 2
+    result = data * 3
     return result
''')
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return json.dumps({"success": False, "error": f"文件不存在: {file_path}"})
        
        original_content = path.read_text(encoding='utf-8')
        original_lines = original_content.splitlines(keepends=True)
        
        # 解析diff
        patched_lines = _apply_unified_diff(original_lines, diff_content)
        
        if patched_lines is None:
            return json.dumps({
                "success": False,
                "error": "无法应用diff补丁，可能与当前文件内容不匹配"
            })
        
        new_content = ''.join(patched_lines)
        path.write_text(new_content, encoding='utf-8')
        
        # 记录历史
        _edit_history.record(EditRecord(
            file_path=str(path.absolute()),
            operation="apply_diff",
            timestamp=datetime.now().isoformat(),
            before_content=original_content,
            after_content=new_content,
            details={"diff_preview": diff_content[:500]}
        ))
        
        return json.dumps({
            "success": True,
            "file": file_path,
            "changes_applied": True
        }, ensure_ascii=False)
        
    except Exception as e:
        return json.dumps({"success": False, "error": str(e)})


@tool
def read_file_with_lines(
    file_path: str,
    start_line: int = None,
    end_line: int = None
) -> str:
    """
    读取文件内容，带行号显示。
    
    这个工具对于str_replace很重要，因为你需要看到确切的内容（包括空格）才能正确匹配。
    
    Args:
        file_path: 文件路径
        start_line: 起始行号（可选，1-indexed）
        end_line: 结束行号（可选，1-indexed）
        
    Returns:
        带行号的文件内容
        
    Example:
        # 读取整个文件
        read_file_with_lines("main.py")
        
        # 只读取第10-20行
        read_file_with_lines("main.py", start_line=10, end_line=20)
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return f"ERROR: 文件不存在: {file_path}"
        
        content = path.read_text(encoding='utf-8')
        lines = content.splitlines()
        
        # 处理行范围
        if start_line is not None:
            start_idx = max(0, start_line - 1)
        else:
            start_idx = 0
            
        if end_line is not None:
            end_idx = min(len(lines), end_line)
        else:
            end_idx = len(lines)
        
        # 格式化输出
        output_lines = []
        output_lines.append(f"=== FILE: {file_path} ===")
        output_lines.append(f"=== LINES: {start_idx + 1}-{end_idx} of {len(lines)} ===")
        output_lines.append("")
        
        # 计算行号宽度
        line_num_width = len(str(end_idx))
        
        for i in range(start_idx, end_idx):
            line_num = str(i + 1).rjust(line_num_width)
            output_lines.append(f"{line_num} | {lines[i]}")
        
        return '\n'.join(output_lines)
        
    except Exception as e:
        return f"ERROR: {str(e)}"


@tool
def undo_edit(file_path: str = None) -> str:
    """
    撤销最近的编辑操作。
    
    Args:
        file_path: 可选，只撤销指定文件的编辑
        
    Returns:
        撤销结果
    """
    record = _edit_history.undo_last(file_path)
    
    if record:
        return json.dumps({
            "success": True,
            "undone_operation": record.operation,
            "file": record.file_path,
            "timestamp": record.timestamp
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "success": False,
            "error": "没有可撤销的编辑记录"
        })


@tool
def get_edit_history(file_path: str = None, limit: int = 10) -> str:
    """
    获取编辑历史记录。
    
    Args:
        file_path: 可选，只获取指定文件的历史
        limit: 返回的最大记录数
        
    Returns:
        编辑历史列表
    """
    records = _edit_history.get_history(file_path, limit)
    
    history = []
    for r in records:
        history.append({
            "file": r.file_path,
            "operation": r.operation,
            "timestamp": r.timestamp,
            "details": r.details
        })
    
    return json.dumps(history, ensure_ascii=False, indent=2)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _find_similar_text(content: str, search_text: str, context: int = 100) -> str:
    """找到文件中最相似的文本片段"""
    # 简化搜索文本用于模糊匹配
    search_simplified = ' '.join(search_text.split())
    
    best_match = None
    best_ratio = 0
    
    # 滑动窗口搜索
    window_size = len(search_text) + context
    for i in range(0, len(content) - window_size, 50):
        chunk = content[i:i + window_size]
        chunk_simplified = ' '.join(chunk.split())
        
        ratio = difflib.SequenceMatcher(None, search_simplified, chunk_simplified).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = chunk
    
    if best_match and best_ratio > 0.5:
        return f"(相似度 {best_ratio:.0%}):\n{best_match}"
    
    return "未找到相似内容"


def _generate_diff(old_content: str, new_content: str, file_path: str) -> str:
    """生成unified diff格式的差异"""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{file_path}",
        tofile=f"b/{file_path}"
    )
    
    return ''.join(diff)


def _apply_unified_diff(original_lines: List[str], diff_content: str) -> Optional[List[str]]:
    """应用unified diff补丁"""
    try:
        # 简化实现：解析diff并应用
        # 实际生产环境应使用专门的patch库
        patched = list(original_lines)
        
        # 解析diff块
        hunks = re.findall(
            r'@@ -(\d+),?\d* \+(\d+),?\d* @@\n((?:[ +-].*\n)*)',
            diff_content
        )
        
        offset = 0
        for old_start, new_start, hunk_content in hunks:
            old_start = int(old_start) - 1 + offset
            lines = hunk_content.splitlines(keepends=True)
            
            # 处理hunk中的每一行
            new_hunk_lines = []
            for line in lines:
                if line.startswith('+'):
                    new_hunk_lines.append(line[1:])
                elif line.startswith('-'):
                    pass  # 删除的行
                else:
                    new_hunk_lines.append(line[1:] if line.startswith(' ') else line)
            
            # 计算要替换的行数
            old_lines_count = sum(1 for l in lines if l.startswith('-') or l.startswith(' '))
            
            # 应用修改
            patched = patched[:old_start] + new_hunk_lines + patched[old_start + old_lines_count:]
            offset += len(new_hunk_lines) - old_lines_count
        
        return patched
    except Exception:
        return None


# ============================================================================
# TOOL COLLECTION
# ============================================================================

ADVANCED_EDIT_TOOLS = [
    str_replace,
    insert_at_line,
    delete_lines,
    apply_diff,
    read_file_with_lines,
    undo_edit,
    get_edit_history
]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

"""
使用示例：

1. 精确替换（最常用）
-----------------------
result = str_replace(
    "main.py",
    old_str='def old_function():',
    new_str='def new_function():'
)

2. 添加新代码
-----------------------
# 方法1：使用str_replace在现有代码后添加
str_replace(
    "main.py",
    old_str="import os",
    new_str="import os\\nimport json  # 新添加"
)

# 方法2：使用insert_at_line在指定位置插入
insert_at_line(
    "main.py",
    line_number=10,
    content="# 这是新添加的注释\\n",
    position="after"
)

3. 删除代码
-----------------------
# 删除第20-25行
delete_lines("main.py", start_line=20, end_line=25)

4. 复杂修改（使用diff）
-----------------------
apply_diff("main.py", '''
--- a/main.py
+++ b/main.py
@@ -15,8 +15,10 @@
 def process():
-    old_logic()
+    new_logic()
+    additional_step()
     return result
''')

5. 查看文件内容（准备str_replace）
-----------------------
# 先查看确切内容
content = read_file_with_lines("main.py", start_line=10, end_line=20)
print(content)  # 看到确切的缩进和空格

# 然后精确替换
str_replace("main.py", old_str="...", new_str="...")

6. 撤销操作
-----------------------
undo_edit("main.py")  # 撤销对main.py的最近编辑
undo_edit()           # 撤销任意文件的最近编辑
"""

