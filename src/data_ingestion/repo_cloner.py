"""
Repository Cloner Module
用于克隆 GitHub 仓库到本地的模块
"""
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
from git import Repo, GitCommandError
from loguru import logger


class RepoCloner:
    """Git 仓库克隆器"""
    
    def __init__(self, base_path: str = "./data/repos"):
        """
        初始化 RepoCloner
        
        Args:
            base_path: 仓库存储的基础路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"RepoCloner 初始化完成，基础路径: {self.base_path}")
    
    def clone(self, repo_url: str, local_path: Optional[str] = None) -> Dict[str, Any]:
        """
        克隆 Git 仓库到本地
        
        Args:
            repo_url: Git 仓库 URL
            local_path: 可选的本地路径，如果不提供将自动生成
            
        Returns:
            包含克隆信息的字典
        """
        try:
            # 如果没有提供本地路径，则根据仓库名生成
            if local_path is None:
                repo_name = self._extract_repo_name(repo_url)
                local_path = self.base_path / repo_name
            else:
                local_path = Path(local_path)
            
            # 如果目录已存在，删除它
            if local_path.exists():
                logger.warning(f"目录 {local_path} 已存在，将删除并重新克隆")
                shutil.rmtree(local_path)
            
            logger.info(f"开始克隆仓库: {repo_url} -> {local_path}")
            
            # 克隆仓库
            repo = Repo.clone_from(repo_url, local_path)
            
            # 获取仓库信息
            repo_info = self._get_repo_info(repo)
            
            result = {
                "success": True,
                "repo_url": repo_url,
                "local_path": str(local_path),
                "repo_info": repo_info,
                "message": f"成功克隆仓库到 {local_path}"
            }
            
            logger.success(f"仓库克隆完成: {repo_url}")
            return result
            
        except GitCommandError as e:
            error_msg = f"Git 命令执行失败: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "repo_url": repo_url,
                "local_path": str(local_path) if 'local_path' in locals() else None,
                "error": error_msg,
                "message": "仓库克隆失败"
            }
        except Exception as e:
            error_msg = f"克隆仓库时发生未知错误: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "repo_url": repo_url,
                "local_path": str(local_path) if 'local_path' in locals() else None,
                "error": error_msg,
                "message": "仓库克隆失败"
            }
    
    def _extract_repo_name(self, repo_url: str) -> str:
        """
        从仓库 URL 中提取仓库名称
        
        Args:
            repo_url: Git 仓库 URL
            
        Returns:
            仓库名称
        """
        # 处理不同格式的 Git URL
        if repo_url.endswith('.git'):
            repo_url = repo_url[:-4]
        
        # 提取最后的路径部分作为仓库名
        repo_name = repo_url.split('/')[-1]
        
        # 清理仓库名，确保它是有效的目录名
        repo_name = "".join(c for c in repo_name if c.isalnum() or c in ('-', '_', '.'))
        
        return repo_name
    
    def _get_repo_info(self, repo: Repo) -> Dict[str, Any]:
        """
        获取仓库的基本信息
        
        Args:
            repo: GitPython Repo 对象
            
        Returns:
            仓库信息字典
        """
        try:
            # 获取最新提交信息
            latest_commit = repo.head.commit
            
            # 获取分支信息
            branches = [ref.name for ref in repo.references if ref.name.startswith('origin/')]
            
            # 获取远程 URL
            remote_urls = [remote.url for remote in repo.remotes]
            
            return {
                "latest_commit": {
                    "hash": latest_commit.hexsha,
                    "message": latest_commit.message.strip(),
                    "author": str(latest_commit.author),
                    "date": latest_commit.committed_datetime.isoformat()
                },
                "branches": branches,
                "remote_urls": remote_urls,
                "active_branch": repo.active_branch.name if repo.active_branch else None
            }
        except Exception as e:
            logger.warning(f"获取仓库信息时出错: {str(e)}")
            return {"error": str(e)}
    
    def update_repo(self, local_path: str) -> Dict[str, Any]:
        """
        更新本地仓库
        
        Args:
            local_path: 本地仓库路径
            
        Returns:
            更新结果信息
        """
        try:
            repo = Repo(local_path)
            origin = repo.remotes.origin
            
            logger.info(f"开始更新仓库: {local_path}")
            
            # 拉取最新更改
            origin.pull()
            
            # 获取更新后的仓库信息
            repo_info = self._get_repo_info(repo)
            
            result = {
                "success": True,
                "local_path": local_path,
                "repo_info": repo_info,
                "message": f"仓库更新完成: {local_path}"
            }
            
            logger.success(f"仓库更新完成: {local_path}")
            return result
            
        except Exception as e:
            error_msg = f"更新仓库时发生错误: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "local_path": local_path,
                "error": error_msg,
                "message": "仓库更新失败"
            }
    
    def list_repos(self) -> Dict[str, Any]:
        """
        列出所有已克隆的仓库
        
        Returns:
            仓库列表信息
        """
        repos = []
        
        try:
            for item in self.base_path.iterdir():
                if item.is_dir() and (item / '.git').exists():
                    try:
                        repo = Repo(str(item))
                        repo_info = self._get_repo_info(repo)
                        repos.append({
                            "name": item.name,
                            "path": str(item),
                            "info": repo_info
                        })
                    except Exception as e:
                        logger.warning(f"读取仓库 {item} 信息时出错: {str(e)}")
            
            return {
                "success": True,
                "repos": repos,
                "count": len(repos),
                "message": f"找到 {len(repos)} 个仓库"
            }
            
        except Exception as e:
            error_msg = f"列出仓库时发生错误: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "列出仓库失败"
            }
