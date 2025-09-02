"""
DeepCode-Analyst 主程序
基于多智能体与图谱推理的开源项目深度解析与技术问答系统
"""
import os
import sys
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import click
from loguru import logger
from dotenv import load_dotenv

# 添加src目录到Python路径
sys.path.append(str(Path(__file__).parent / "src"))

from src.data_ingestion import RepoCloner, CodeParser
from src.graph_builder import KnowledgeGraph


class DeepCodeAnalyst:
    """DeepCode-Analyst 主控制器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化 DeepCode-Analyst
        
        Args:
            config_path: 配置文件路径
        """
        # 加载环境变量
        load_dotenv()
        
        # 初始化组件
        self.repo_cloner = RepoCloner()
        self.code_parser = CodeParser()
        self.knowledge_graph = KnowledgeGraph()
        
        # 创建必要的目录
        self._setup_directories()
        
        logger.info("DeepCode-Analyst 初始化完成")
    
    def _setup_directories(self):
        """设置必要的目录结构"""
        directories = [
            "./data",
            "./data/repos", 
            "./data/graphs",
            "./outputs",
            "./logs"
        ]
        
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def analyze_repository(self, repo_url: str, 
                         local_path: Optional[str] = None,
                         save_graph: bool = True,
                         graph_format: str = 'pickle') -> Dict[str, Any]:
        """
        分析整个仓库的完整流程
        
        Args:
            repo_url: GitHub 仓库 URL
            local_path: 可选的本地克隆路径
            save_graph: 是否保存知识图谱
            graph_format: 图谱保存格式
            
        Returns:
            分析结果
        """
        logger.info(f"开始分析仓库: {repo_url}")
        
        try:
            # 步骤1: 克隆仓库
            logger.info("步骤1: 克隆仓库...")
            clone_result = self.repo_cloner.clone(repo_url, local_path)
            
            if not clone_result['success']:
                return {
                    'success': False,
                    'stage': 'clone',
                    'error': clone_result['error'],
                    'message': '仓库克隆失败'
                }
            
            repo_path = clone_result['local_path']
            repo_info = clone_result['repo_info']
            
            # 步骤2: 解析代码
            logger.info("步骤2: 解析代码结构...")
            parse_result = self.code_parser.parse_repository(repo_path)
            
            nodes = parse_result['nodes']
            edges = parse_result['edges']
            parse_stats = parse_result['statistics']
            
            # 步骤3: 构建知识图谱
            logger.info("步骤3: 构建知识图谱...")
            graph_result = self.knowledge_graph.build_graph(nodes, edges, repo_info)
            
            if not graph_result['success']:
                return {
                    'success': False,
                    'stage': 'graph_building',
                    'error': graph_result['error'],
                    'message': '知识图谱构建失败'
                }
            
            # 步骤4: 保存图谱
            graph_path = None
            if save_graph:
                logger.info("步骤4: 保存知识图谱...")
                repo_name = Path(repo_path).name
                graph_filename = f"{repo_name}_graph.{graph_format}"
                graph_path = f"./data/graphs/{graph_filename}"
                
                save_result = self.knowledge_graph.save_graph(graph_path, graph_format)
                
                if not save_result['success']:
                    logger.warning(f"保存图谱失败: {save_result['error']}")
            
            # 整理结果
            result = {
                'success': True,
                'repo_url': repo_url,
                'repo_path': repo_path,
                'graph_path': graph_path,
                'statistics': {
                    'parsing': parse_stats,
                    'graph': graph_result['statistics']
                },
                'repo_info': repo_info,
                'message': '仓库分析完成'
            }
            
            logger.success(f"仓库分析完成: {repo_url}")
            return result
            
        except Exception as e:
            error_msg = f"分析仓库时发生未知错误: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'stage': 'unknown',
                'error': error_msg,
                'message': '仓库分析失败'
            }
    
    def load_graph(self, graph_path: str) -> Dict[str, Any]:
        """
        加载已保存的知识图谱
        
        Args:
            graph_path: 图谱文件路径
            
        Returns:
            加载结果
        """
        return self.knowledge_graph.load_graph(graph_path)
    
    def query_graph(self, **kwargs) -> Dict[str, Any]:
        """
        查询知识图谱
        
        Args:
            **kwargs: 查询参数
            
        Returns:
            查询结果
        """
        return {
            'nodes': self.knowledge_graph.query_nodes(**kwargs),
            'statistics': self.knowledge_graph.get_statistics()
        }


@click.group()
def cli():
    """DeepCode-Analyst: 基于多智能体与图谱推理的开源项目深度解析与技术问答系统"""
    pass


@cli.command()
@click.argument('repo_url')
@click.option('--local-path', '-l', help='本地克隆路径')
@click.option('--output-dir', '-o', default='./outputs', help='输出目录')
@click.option('--graph-format', '-f', default='pickle', 
              type=click.Choice(['pickle', 'graphml', 'json']),
              help='图谱保存格式')
@click.option('--verbose', '-v', is_flag=True, help='详细输出')
def analyze(repo_url, local_path, output_dir, graph_format, verbose):
    """分析指定的GitHub仓库"""
    
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # 初始化分析器
    analyst = DeepCodeAnalyst()
    
    # 执行分析
    result = analyst.analyze_repository(
        repo_url=repo_url,
        local_path=local_path,
        save_graph=True,
        graph_format=graph_format
    )
    
    if result['success']:
        click.echo(f"✅ {result['message']}")
        click.echo(f"仓库路径: {result['repo_path']}")
        
        if result['graph_path']:
            click.echo(f"图谱路径: {result['graph_path']}")
        
        # 显示统计信息
        stats = result['statistics']
        click.echo("\n📊 统计信息:")
        click.echo(f"  解析文件数: {stats['parsing']['total_files']}")
        click.echo(f"  成功率: {stats['parsing']['success_rate']:.2%}")
        click.echo(f"  图谱节点数: {stats['graph']['total_nodes']}")
        click.echo(f"  图谱边数: {stats['graph']['total_edges']}")
        
        # 保存详细结果到文件
        import json
        from datetime import datetime
        
        output_file = Path(output_dir) / f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        click.echo(f"详细结果已保存到: {output_file}")
    else:
        click.echo(f"❌ {result['message']}")
        click.echo(f"错误阶段: {result.get('stage', 'unknown')}")
        click.echo(f"错误信息: {result.get('error', 'unknown')}")
        sys.exit(1)


@cli.command()
@click.argument('graph_path')
@click.option('--node-type', help='节点类型过滤')
@click.option('--file-path', help='文件路径过滤')
@click.option('--name-pattern', help='名称模式过滤')
def query(graph_path, node_type, file_path, name_pattern):
    """查询知识图谱"""
    
    analyst = DeepCodeAnalyst()
    
    # 加载图谱
    load_result = analyst.load_graph(graph_path)
    
    if not load_result['success']:
        click.echo(f"❌ 加载图谱失败: {load_result['error']}")
        sys.exit(1)
    
    click.echo(f"✅ 图谱加载成功: {load_result['message']}")
    
    # 执行查询
    query_params = {}
    if node_type:
        query_params['node_type'] = node_type
    if file_path:
        query_params['file_path'] = file_path
    if name_pattern:
        query_params['name_pattern'] = name_pattern
    
    result = analyst.query_graph(**query_params)
    
    # 显示结果
    nodes = result['nodes']
    click.echo(f"\n🔍 查询结果: 找到 {len(nodes)} 个节点")
    
    for i, node in enumerate(nodes[:10]):  # 只显示前10个
        click.echo(f"\n{i+1}. {node['name']} ({node['type']})")
        click.echo(f"   文件: {node.get('file_path', 'N/A')}")
        click.echo(f"   行号: {node.get('start_line', 'N/A')}")
    
    if len(nodes) > 10:
        click.echo(f"\n... 还有 {len(nodes) - 10} 个结果")


@cli.command()
@click.option('--host', default='localhost', help='服务器主机')
@click.option('--port', default=8000, help='服务器端口')
def serve(host, port):
    """启动Web服务器 (TODO: 未来版本实现)"""
    click.echo("Web服务器功能将在未来版本中实现")
    click.echo("当前可以使用 analyze 和 query 命令进行操作")


if __name__ == '__main__':
    cli()
