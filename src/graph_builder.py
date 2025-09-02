"""
Knowledge Graph Builder Module
构建和管理代码知识图谱
"""
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import networkx as nx
from loguru import logger


class KnowledgeGraph:
    """代码知识图谱构建器和管理器"""
    
    def __init__(self):
        """初始化知识图谱"""
        self.graph = nx.DiGraph()
        self.metadata = {
            'created_at': None,
            'repo_info': {},
            'statistics': {}
        }
        logger.info("KnowledgeGraph 初始化完成")
    
    def build_graph(self, nodes: List[Dict], edges: List[Dict], repo_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        从解析的节点和边构建知识图谱
        
        Args:
            nodes: 节点列表
            edges: 边列表
            repo_info: 仓库信息
            
        Returns:
            构建结果信息
        """
        logger.info(f"开始构建知识图谱: {len(nodes)} 个节点, {len(edges)} 条边")
        
        try:
            # 清空现有图谱
            self.graph.clear()
            
            # 添加节点
            self._add_nodes(nodes)
            
            # 添加边
            self._add_edges(edges, nodes)
            
            # 计算图谱统计信息
            stats = self._calculate_statistics()
            
            # 更新元数据
            from datetime import datetime
            self.metadata = {
                'created_at': datetime.now().isoformat(),
                'repo_info': repo_info or {},
                'statistics': stats
            }
            
            result = {
                'success': True,
                'message': '知识图谱构建完成',
                'statistics': stats
            }
            
            logger.success(f"知识图谱构建完成: {stats}")
            return result
            
        except Exception as e:
            error_msg = f"构建知识图谱时发生错误: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'message': '知识图谱构建失败'
            }
    
    def _add_nodes(self, nodes: List[Dict]):
        """添加节点到图谱"""
        for node in nodes:
            node_id = node.get('id', f"{node['type']}_{len(self.graph.nodes)}")
            
            # 添加节点属性
            self.graph.add_node(node_id, **node)
            
        logger.info(f"已添加 {len(nodes)} 个节点")
    
    def _add_edges(self, edges: List[Dict], nodes: List[Dict]):
        """添加边到图谱"""
        # 创建节点名称到ID的映射
        node_mapping = self._create_node_mapping(nodes)
        
        edge_count = 0
        
        for edge in edges:
            edge_type = edge.get('type')
            
            if edge_type == 'function_call':
                source_id = self._find_caller_node(edge, node_mapping)
                target_id = self._find_called_node(edge, node_mapping)
                
                if source_id and target_id:
                    self.graph.add_edge(source_id, target_id, 
                                      edge_type='calls',
                                      **edge)
                    edge_count += 1
                    
            elif edge_type == 'import':
                # 处理导入关系
                self._add_import_edge(edge, node_mapping)
                edge_count += 1
        
        logger.info(f"已添加 {edge_count} 条边")
    
    def _create_node_mapping(self, nodes: List[Dict]) -> Dict[str, List[str]]:
        """创建节点名称到ID的映射"""
        mapping = {}
        
        for node in nodes:
            name = node.get('name', '')
            node_id = node.get('id', '')
            
            if name not in mapping:
                mapping[name] = []
            mapping[name].append(node_id)
        
        return mapping
    
    def _find_caller_node(self, edge: Dict, node_mapping: Dict) -> Optional[str]:
        """查找调用者节点"""
        caller_file = edge.get('caller_file', '')
        caller_line = edge.get('caller_line', 0)
        
        # 查找包含此调用行的函数或类
        for node_id, node_data in self.graph.nodes(data=True):
            if (node_data.get('file_path') == caller_file and
                node_data.get('start_line', 0) <= caller_line <= node_data.get('end_line', 0)):
                return node_id
        
        return None
    
    def _find_called_node(self, edge: Dict, node_mapping: Dict) -> Optional[str]:
        """查找被调用的节点"""
        called_function = edge.get('called_function', '')
        
        # 简单名称匹配
        if called_function in node_mapping:
            # 如果有多个同名函数，返回第一个
            return node_mapping[called_function][0]
        
        # 处理方法调用 (object.method)
        if '.' in called_function:
            method_name = called_function.split('.')[-1]
            if method_name in node_mapping:
                return node_mapping[method_name][0]
        
        return None
    
    def _add_import_edge(self, edge: Dict, node_mapping: Dict):
        """添加导入关系边"""
        # 这里可以添加更复杂的导入关系处理逻辑
        # 目前暂时跳过，因为需要更详细的模块分析
        pass
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        """计算图谱统计信息"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': {},
            'edge_types': {},
            'graph_metrics': {}
        }
        
        # 统计节点类型
        for _, node_data in self.graph.nodes(data=True):
            node_type = node_data.get('type', 'unknown')
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # 统计边类型
        for _, _, edge_data in self.graph.edges(data=True):
            edge_type = edge_data.get('edge_type', 'unknown')
            stats['edge_types'][edge_type] = stats['edge_types'].get(edge_type, 0) + 1
        
        # 计算图谱指标
        try:
            if self.graph.number_of_nodes() > 0:
                stats['graph_metrics'] = {
                    'density': nx.density(self.graph),
                    'is_connected': nx.is_weakly_connected(self.graph),
                    'number_of_components': nx.number_weakly_connected_components(self.graph)
                }
                
                # 计算中心性指标（如果图不太大）
                if self.graph.number_of_nodes() < 1000:
                    try:
                        centrality = nx.degree_centrality(self.graph)
                        stats['graph_metrics']['max_degree_centrality'] = max(centrality.values()) if centrality else 0
                        stats['graph_metrics']['avg_degree_centrality'] = sum(centrality.values()) / len(centrality) if centrality else 0
                    except:
                        pass
                        
        except Exception as e:
            logger.warning(f"计算图谱指标时出错: {str(e)}")
        
        return stats
    
    def save_graph(self, file_path: str, format: str = 'pickle') -> Dict[str, Any]:
        """
        保存知识图谱
        
        Args:
            file_path: 保存路径
            format: 保存格式 ('pickle', 'graphml', 'json')
            
        Returns:
            保存结果信息
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format.lower() == 'pickle':
                graph_data = {
                    'graph': self.graph,
                    'metadata': self.metadata
                }
                with open(file_path, 'wb') as f:
                    pickle.dump(graph_data, f)
                    
            elif format.lower() == 'graphml':
                nx.write_graphml(self.graph, file_path)
                
                # 单独保存元数据
                metadata_path = file_path.with_suffix('.metadata.json')
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(self.metadata, f, indent=2, ensure_ascii=False)
                    
            elif format.lower() == 'json':
                graph_data = {
                    'nodes': [
                        {**data, 'id': node_id} 
                        for node_id, data in self.graph.nodes(data=True)
                    ],
                    'edges': [
                        {'source': u, 'target': v, **data}
                        for u, v, data in self.graph.edges(data=True)
                    ],
                    'metadata': self.metadata
                }
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(graph_data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            result = {
                'success': True,
                'file_path': str(file_path),
                'format': format,
                'message': f'知识图谱已保存到 {file_path}'
            }
            
            logger.success(f"知识图谱已保存: {file_path}")
            return result
            
        except Exception as e:
            error_msg = f"保存知识图谱时发生错误: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'message': '知识图谱保存失败'
            }
    
    def load_graph(self, file_path: str, format: str = 'pickle') -> Dict[str, Any]:
        """
        加载知识图谱
        
        Args:
            file_path: 文件路径
            format: 文件格式
            
        Returns:
            加载结果信息
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileNotFoundError(f"文件不存在: {file_path}")
            
            if format.lower() == 'pickle':
                with open(file_path, 'rb') as f:
                    graph_data = pickle.load(f)
                    self.graph = graph_data['graph']
                    self.metadata = graph_data['metadata']
                    
            elif format.lower() == 'graphml':
                self.graph = nx.read_graphml(file_path)
                
                # 尝试加载元数据
                metadata_path = file_path.with_suffix('.metadata.json')
                if metadata_path.exists():
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
                        
            elif format.lower() == 'json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    graph_data = json.load(f)
                    
                # 重建图谱
                self.graph.clear()
                
                # 添加节点
                for node in graph_data['nodes']:
                    node_id = node.pop('id')
                    self.graph.add_node(node_id, **node)
                
                # 添加边
                for edge in graph_data['edges']:
                    source = edge.pop('source')
                    target = edge.pop('target')
                    self.graph.add_edge(source, target, **edge)
                
                self.metadata = graph_data.get('metadata', {})
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            result = {
                'success': True,
                'file_path': str(file_path),
                'format': format,
                'statistics': self._calculate_statistics(),
                'message': f'知识图谱已从 {file_path} 加载'
            }
            
            logger.success(f"知识图谱已加载: {file_path}")
            return result
            
        except Exception as e:
            error_msg = f"加载知识图谱时发生错误: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'message': '知识图谱加载失败'
            }
    
    def query_nodes(self, node_type: Optional[str] = None, 
                   file_path: Optional[str] = None,
                   name_pattern: Optional[str] = None) -> List[Dict]:
        """
        查询节点
        
        Args:
            node_type: 节点类型过滤
            file_path: 文件路径过滤
            name_pattern: 名称模式过滤
            
        Returns:
            匹配的节点列表
        """
        results = []
        
        for node_id, node_data in self.graph.nodes(data=True):
            # 类型过滤
            if node_type and node_data.get('type') != node_type:
                continue
            
            # 文件路径过滤
            if file_path and node_data.get('file_path') != file_path:
                continue
            
            # 名称模式过滤
            if name_pattern and name_pattern.lower() not in node_data.get('name', '').lower():
                continue
            
            results.append({
                'id': node_id,
                **node_data
            })
        
        return results
    
    def get_node_connections(self, node_id: str, direction: str = 'both') -> Dict[str, List]:
        """
        获取节点的连接关系
        
        Args:
            node_id: 节点ID
            direction: 方向 ('in', 'out', 'both')
            
        Returns:
            连接关系信息
        """
        if node_id not in self.graph:
            return {'incoming': [], 'outgoing': []}
        
        result = {'incoming': [], 'outgoing': []}
        
        if direction in ['in', 'both']:
            # 获取入边
            for pred in self.graph.predecessors(node_id):
                edge_data = self.graph.get_edge_data(pred, node_id)
                result['incoming'].append({
                    'source': pred,
                    'source_data': self.graph.nodes[pred],
                    'edge_data': edge_data
                })
        
        if direction in ['out', 'both']:
            # 获取出边
            for succ in self.graph.successors(node_id):
                edge_data = self.graph.get_edge_data(node_id, succ)
                result['outgoing'].append({
                    'target': succ,
                    'target_data': self.graph.nodes[succ],
                    'edge_data': edge_data
                })
        
        return result
    
    def find_call_chains(self, start_node: str, max_depth: int = 3) -> List[List[str]]:
        """
        查找函数调用链
        
        Args:
            start_node: 起始节点
            max_depth: 最大深度
            
        Returns:
            调用链列表
        """
        if start_node not in self.graph:
            return []
        
        chains = []
        
        def dfs(current, path, depth):
            if depth >= max_depth:
                return
            
            for successor in self.graph.successors(current):
                edge_data = self.graph.get_edge_data(current, successor)
                if edge_data and edge_data.get('edge_type') == 'calls':
                    new_path = path + [successor]
                    chains.append(new_path)
                    dfs(successor, new_path, depth + 1)
        
        dfs(start_node, [start_node], 0)
        return chains
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取图谱统计信息"""
        return {
            'metadata': self.metadata,
            'current_statistics': self._calculate_statistics()
        }
