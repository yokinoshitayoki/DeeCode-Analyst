"""
Code Parser Module
使用 tree-sitter 解析源代码并提取结构化信息
"""
import os
import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from loguru import logger


class CodeParser:
    """代码解析器，用于提取代码结构信息"""
    
    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust'
    }
    
    def __init__(self):
        """初始化代码解析器"""
        self.parsers = {}
        self._setup_parsers()
        logger.info("CodeParser 初始化完成")
    
    def _setup_parsers(self):
        """设置不同语言的解析器"""
        try:
            # 设置 Python 解析器
            PY_LANGUAGE = Language(tspython.language(), "python")
            py_parser = Parser()
            py_parser.set_language(PY_LANGUAGE)
            self.parsers['python'] = py_parser
            logger.info("Python 解析器设置完成")
        except Exception as e:
            logger.warning(f"设置 Python 解析器失败: {str(e)}")
        
        # TODO: 添加其他语言的解析器
        # 当前版本专注于 Python 解析
    
    def parse_repository(self, repo_path: str) -> Dict[str, Any]:
        """
        解析整个仓库的代码结构
        
        Args:
            repo_path: 仓库根目录路径
            
        Returns:
            包含节点和边信息的字典
        """
        repo_path = Path(repo_path)
        
        if not repo_path.exists():
            raise ValueError(f"仓库路径不存在: {repo_path}")
        
        logger.info(f"开始解析仓库: {repo_path}")
        
        all_nodes = []
        all_edges = []
        file_count = 0
        error_count = 0
        
        # 遍历所有支持的源代码文件
        for file_path in self._find_source_files(repo_path):
            try:
                nodes, edges = self.parse_file(file_path)
                all_nodes.extend(nodes)
                all_edges.extend(edges)
                file_count += 1
                
                if file_count % 50 == 0:
                    logger.info(f"已解析 {file_count} 个文件...")
                    
            except Exception as e:
                error_count += 1
                logger.warning(f"解析文件 {file_path} 时出错: {str(e)}")
        
        result = {
            "nodes": all_nodes,
            "edges": all_edges,
            "statistics": {
                "total_files": file_count,
                "error_files": error_count,
                "total_nodes": len(all_nodes),
                "total_edges": len(all_edges),
                "success_rate": (file_count - error_count) / file_count if file_count > 0 else 0
            }
        }
        
        logger.success(f"仓库解析完成: {file_count} 个文件, {len(all_nodes)} 个节点, {len(all_edges)} 条边")
        return result
    
    def parse_file(self, file_path: Path) -> Tuple[List[Dict], List[Dict]]:
        """
        解析单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            (节点列表, 边列表)
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            return [], []
        
        language = self.SUPPORTED_EXTENSIONS[file_ext]
        
        if language not in self.parsers:
            logger.warning(f"不支持的语言: {language}")
            return [], []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                source_code = f.read()
        except Exception as e:
            logger.warning(f"读取文件 {file_path} 失败: {str(e)}")
            return [], []
        
        if language == 'python':
            return self._parse_python_file(file_path, source_code)
        else:
            # TODO: 实现其他语言的解析
            return [], []
    
    def _parse_python_file(self, file_path: Path, source_code: str) -> Tuple[List[Dict], List[Dict]]:
        """
        解析 Python 文件
        
        Args:
            file_path: 文件路径
            source_code: 源代码内容
            
        Returns:
            (节点列表, 边列表)
        """
        nodes = []
        edges = []
        
        try:
            # 使用 tree-sitter 解析
            parser = self.parsers['python']
            tree = parser.parse(bytes(source_code, "utf8"))
            root_node = tree.root_node
            
            # 提取函数定义
            functions = self._extract_functions(root_node, source_code, file_path)
            nodes.extend(functions)
            
            # 提取类定义
            classes = self._extract_classes(root_node, source_code, file_path)
            nodes.extend(classes)
            
            # 提取函数调用关系
            function_calls = self._extract_function_calls(root_node, source_code, file_path)
            edges.extend(function_calls)
            
            # 提取导入关系
            imports = self._extract_imports(root_node, source_code, file_path)
            edges.extend(imports)
            
        except Exception as e:
            logger.warning(f"使用 tree-sitter 解析 {file_path} 失败: {str(e)}")
            # 回退到 AST 解析
            try:
                nodes, edges = self._parse_python_with_ast(file_path, source_code)
            except Exception as ast_e:
                logger.error(f"AST 解析 {file_path} 也失败: {str(ast_e)}")
                return [], []
        
        return nodes, edges
    
    def _extract_functions(self, root_node: Node, source_code: str, file_path: Path) -> List[Dict]:
        """提取函数定义"""
        functions = []
        
        def traverse(node):
            if node.type == 'function_definition':
                # 获取函数名
                name_node = node.child_by_field_name('name')
                if name_node:
                    func_name = source_code[name_node.start_byte:name_node.end_byte]
                    
                    # 获取参数
                    params_node = node.child_by_field_name('parameters')
                    params = []
                    if params_node:
                        for child in params_node.children:
                            if child.type == 'identifier':
                                params.append(source_code[child.start_byte:child.end_byte])
                    
                    # 获取函数体的开始和结束行
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    
                    functions.append({
                        'type': 'function',
                        'name': func_name,
                        'file_path': str(file_path),
                        'start_line': start_line,
                        'end_line': end_line,
                        'parameters': params,
                        'id': f"{file_path}:{func_name}:{start_line}"
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return functions
    
    def _extract_classes(self, root_node: Node, source_code: str, file_path: Path) -> List[Dict]:
        """提取类定义"""
        classes = []
        
        def traverse(node):
            if node.type == 'class_definition':
                # 获取类名
                name_node = node.child_by_field_name('name')
                if name_node:
                    class_name = source_code[name_node.start_byte:name_node.end_byte]
                    
                    # 获取父类
                    superclasses = []
                    superclasses_node = node.child_by_field_name('superclasses')
                    if superclasses_node:
                        for child in superclasses_node.children:
                            if child.type == 'identifier':
                                superclasses.append(source_code[child.start_byte:child.end_byte])
                    
                    start_line = node.start_point[0] + 1
                    end_line = node.end_point[0] + 1
                    
                    classes.append({
                        'type': 'class',
                        'name': class_name,
                        'file_path': str(file_path),
                        'start_line': start_line,
                        'end_line': end_line,
                        'superclasses': superclasses,
                        'id': f"{file_path}:{class_name}:{start_line}"
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return classes
    
    def _extract_function_calls(self, root_node: Node, source_code: str, file_path: Path) -> List[Dict]:
        """提取函数调用关系"""
        calls = []
        
        def traverse(node):
            if node.type == 'call':
                # 获取被调用的函数名
                function_node = node.child_by_field_name('function')
                if function_node:
                    if function_node.type == 'identifier':
                        func_name = source_code[function_node.start_byte:function_node.end_byte]
                    elif function_node.type == 'attribute':
                        # 处理方法调用 (object.method)
                        func_name = source_code[function_node.start_byte:function_node.end_byte]
                    else:
                        func_name = source_code[function_node.start_byte:function_node.end_byte]
                    
                    call_line = node.start_point[0] + 1
                    
                    calls.append({
                        'type': 'function_call',
                        'caller_file': str(file_path),
                        'caller_line': call_line,
                        'called_function': func_name,
                        'id': f"{file_path}:call:{call_line}:{func_name}"
                    })
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return calls
    
    def _extract_imports(self, root_node: Node, source_code: str, file_path: Path) -> List[Dict]:
        """提取导入关系"""
        imports = []
        
        def traverse(node):
            if node.type in ['import_statement', 'import_from_statement']:
                import_line = node.start_point[0] + 1
                import_text = source_code[node.start_byte:node.end_byte]
                
                imports.append({
                    'type': 'import',
                    'file_path': str(file_path),
                    'line': import_line,
                    'import_statement': import_text.strip(),
                    'id': f"{file_path}:import:{import_line}"
                })
            
            for child in node.children:
                traverse(child)
        
        traverse(root_node)
        return imports
    
    def _parse_python_with_ast(self, file_path: Path, source_code: str) -> Tuple[List[Dict], List[Dict]]:
        """
        使用 Python AST 作为回退解析方案
        """
        nodes = []
        edges = []
        
        try:
            tree = ast.parse(source_code)
            
            class ASTVisitor(ast.NodeVisitor):
                def visit_FunctionDef(self, node):
                    nodes.append({
                        'type': 'function',
                        'name': node.name,
                        'file_path': str(file_path),
                        'start_line': node.lineno,
                        'end_line': node.end_lineno or node.lineno,
                        'parameters': [arg.arg for arg in node.args.args],
                        'id': f"{file_path}:{node.name}:{node.lineno}"
                    })
                    self.generic_visit(node)
                
                def visit_ClassDef(self, node):
                    superclasses = []
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            superclasses.append(base.id)
                    
                    nodes.append({
                        'type': 'class',
                        'name': node.name,
                        'file_path': str(file_path),
                        'start_line': node.lineno,
                        'end_line': node.end_lineno or node.lineno,
                        'superclasses': superclasses,
                        'id': f"{file_path}:{node.name}:{node.lineno}"
                    })
                    self.generic_visit(node)
                
                def visit_Call(self, node):
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    else:
                        func_name = "unknown"
                    
                    edges.append({
                        'type': 'function_call',
                        'caller_file': str(file_path),
                        'caller_line': node.lineno,
                        'called_function': func_name,
                        'id': f"{file_path}:call:{node.lineno}:{func_name}"
                    })
                    self.generic_visit(node)
            
            visitor = ASTVisitor()
            visitor.visit(tree)
            
        except SyntaxError as e:
            logger.warning(f"Python 语法错误 in {file_path}: {str(e)}")
        
        return nodes, edges
    
    def _find_source_files(self, repo_path: Path) -> List[Path]:
        """
        查找仓库中的所有源代码文件
        
        Args:
            repo_path: 仓库根目录
            
        Returns:
            源代码文件路径列表
        """
        source_files = []
        
        # 需要忽略的目录
        ignore_dirs = {
            '.git', '__pycache__', '.pytest_cache', 'node_modules', 
            '.venv', 'venv', 'env', '.env', 'build', 'dist', 
            '.idea', '.vscode', 'target', 'bin', 'obj'
        }
        
        def should_ignore_file(file_path: Path) -> bool:
            """判断是否应该忽略文件"""
            # 忽略隐藏文件
            if file_path.name.startswith('.'):
                return True
            
            # 忽略特定文件
            ignore_files = {'__init__.py'} # 可以根据需要添加更多
            if file_path.name in ignore_files and file_path.stat().st_size == 0:
                return True
            
            # 忽略过大的文件 (>1MB)
            try:
                if file_path.stat().st_size > 1024 * 1024:
                    return True
            except:
                return True
            
            return False
        
        for root, dirs, files in os.walk(repo_path):
            # 修改 dirs 来跳过不需要的目录
            dirs[:] = [d for d in dirs if d not in ignore_dirs]
            
            for file in files:
                file_path = Path(root) / file
                
                if should_ignore_file(file_path):
                    continue
                
                if file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    source_files.append(file_path)
        
        logger.info(f"找到 {len(source_files)} 个源代码文件")
        return source_files
