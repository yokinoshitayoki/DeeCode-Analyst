"""
Multi-Agent System using LangGraph
基于 LangGraph 的多智能体代码分析系统
"""
import os
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolInvocation
from loguru import logger


class AgentState(TypedDict):
    """定义智能体图的状态"""
    # 输入
    user_question: str
    knowledge_graph: Any  # KnowledgeGraph 实例
    
    # 中间状态
    sub_tasks: List[Dict[str, str]]
    current_task_index: int
    analysis_results: List[Dict[str, Any]]
    
    # 输出
    final_report: str
    messages: Annotated[List[BaseMessage], operator.add]


class AgentGraph:
    """多智能体分析图"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", temperature: float = 0.1):
        """
        初始化智能体图
        
        Args:
            model_name: 使用的语言模型
            temperature: 模型温度参数
        """
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.graph = self._build_graph()
        logger.info(f"AgentGraph 初始化完成，使用模型: {model_name}")
    
    def _build_graph(self) -> StateGraph:
        """构建智能体图"""
        
        # 创建状态图
        workflow = StateGraph(AgentState)
        
        # 添加节点
        workflow.add_node("task_decomposer", self.task_decomposer_node)
        workflow.add_node("code_analyst", self.code_analyst_node)
        workflow.add_node("graph_rag", self.graph_rag_node)
        workflow.add_node("synthesizer", self.synthesizer_node)
        
        # 设置入口点
        workflow.set_entry_point("task_decomposer")
        
        # 添加边
        workflow.add_edge("task_decomposer", "code_analyst")
        workflow.add_conditional_edges(
            "code_analyst",
            self.should_continue_analysis,
            {
                "continue": "graph_rag",
                "next_task": "code_analyst",
                "finish": "synthesizer"
            }
        )
        workflow.add_edge("graph_rag", "code_analyst")
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()
    
    def task_decomposer_node(self, state: AgentState) -> AgentState:
        """
        任务分解节点：将用户问题分解为具体的分析子任务
        """
        logger.info("执行任务分解...")
        
        prompt = PromptTemplate.from_template("""
        你是一个专业的代码分析任务分解专家。请将用户的问题分解为具体的、可执行的分析子任务。

        用户问题: {user_question}

        请将问题分解为2-5个具体的子任务，每个子任务应该：
        1. 目标明确，可以通过代码结构分析来回答
        2. 具有一定的独立性
        3. 按照逻辑顺序排列

        请以JSON格式返回，格式如下：
        [
            {{
                "task_id": "task_1",
                "description": "具体的任务描述",
                "focus_area": "关注的代码领域（如：函数调用关系、类继承结构等）"
            }},
            ...
        ]

        只返回JSON数组，不要包含其他文本。
        """)
        
        try:
            response = self.llm.invoke(prompt.format(user_question=state["user_question"]))
            
            # 解析响应
            import json
            sub_tasks = json.loads(response.content)
            
            # 验证和清理任务
            validated_tasks = []
            for i, task in enumerate(sub_tasks):
                if isinstance(task, dict) and "description" in task:
                    validated_tasks.append({
                        "task_id": task.get("task_id", f"task_{i+1}"),
                        "description": task["description"],
                        "focus_area": task.get("focus_area", "general"),
                        "status": "pending"
                    })
            
            logger.info(f"分解出 {len(validated_tasks)} 个子任务")
            
            return {
                **state,
                "sub_tasks": validated_tasks,
                "current_task_index": 0,
                "analysis_results": [],
                "messages": [AIMessage(content=f"已分解为 {len(validated_tasks)} 个子任务")]
            }
            
        except Exception as e:
            logger.error(f"任务分解失败: {str(e)}")
            # 创建默认任务
            default_tasks = [{
                "task_id": "task_1",
                "description": state["user_question"],
                "focus_area": "general",
                "status": "pending"
            }]
            
            return {
                **state,
                "sub_tasks": default_tasks,
                "current_task_index": 0,
                "analysis_results": [],
                "messages": [AIMessage(content="使用默认任务分解")]
            }
    
    def code_analyst_node(self, state: AgentState) -> AgentState:
        """
        代码分析节点：分析当前子任务并生成图谱查询
        """
        current_index = state["current_task_index"]
        
        if current_index >= len(state["sub_tasks"]):
            return state
        
        current_task = state["sub_tasks"][current_index]
        logger.info(f"分析任务 {current_index + 1}: {current_task['description']}")
        
        prompt = PromptTemplate.from_template("""
        你是一个专业的代码分析师。请分析以下任务，并设计相应的代码结构查询策略。

        任务描述: {task_description}
        关注领域: {focus_area}
        
        请分析这个任务需要查询代码知识图谱的哪些方面，并生成查询策略：

        1. 需要查找的节点类型（如：function, class）
        2. 需要分析的关系类型（如：calls, inherits）
        3. 关键的搜索关键词或模式
        4. 分析的深度和范围

        请以JSON格式返回查询策略：
        {{
            "query_strategy": {{
                "node_types": ["function", "class"],
                "relationship_types": ["calls", "inherits"],
                "search_patterns": ["关键词1", "关键词2"],
                "analysis_depth": "描述分析深度",
                "expected_insights": "期望获得的洞察"
            }}
        }}
        
        只返回JSON，不要包含其他文本。
        """)
        
        try:
            response = self.llm.invoke(prompt.format(
                task_description=current_task["description"],
                focus_area=current_task["focus_area"]
            ))
            
            import json
            query_strategy = json.loads(response.content)
            
            # 更新任务状态
            updated_tasks = state["sub_tasks"].copy()
            updated_tasks[current_index] = {
                **current_task,
                "status": "analyzing",
                "query_strategy": query_strategy["query_strategy"]
            }
            
            logger.info(f"生成查询策略: {query_strategy['query_strategy'].get('expected_insights', 'N/A')}")
            
            return {
                **state,
                "sub_tasks": updated_tasks,
                "messages": [AIMessage(content=f"分析任务 {current_index + 1}，生成查询策略")]
            }
            
        except Exception as e:
            logger.error(f"代码分析失败: {str(e)}")
            
            # 创建默认查询策略
            default_strategy = {
                "node_types": ["function", "class"],
                "relationship_types": ["calls"],
                "search_patterns": [current_task["description"]],
                "analysis_depth": "基础分析",
                "expected_insights": "代码结构概述"
            }
            
            updated_tasks = state["sub_tasks"].copy()
            updated_tasks[current_index] = {
                **current_task,
                "status": "analyzing",
                "query_strategy": default_strategy
            }
            
            return {
                **state,
                "sub_tasks": updated_tasks,
                "messages": [AIMessage(content="使用默认查询策略")]
            }
    
    def graph_rag_node(self, state: AgentState) -> AgentState:
        """
        图谱RAG节点：执行知识图谱查询并检索相关信息
        """
        current_index = state["current_task_index"]
        current_task = state["sub_tasks"][current_index]
        
        logger.info(f"执行图谱查询: 任务 {current_index + 1}")
        
        try:
            knowledge_graph = state["knowledge_graph"]
            query_strategy = current_task.get("query_strategy", {})
            
            # 执行多种查询
            results = {
                "nodes": [],
                "connections": [],
                "statistics": {}
            }
            
            # 1. 按节点类型查询
            node_types = query_strategy.get("node_types", ["function"])
            for node_type in node_types:
                nodes = knowledge_graph.query_nodes(node_type=node_type)
                results["nodes"].extend(nodes[:20])  # 限制结果数量
            
            # 2. 按搜索模式查询
            search_patterns = query_strategy.get("search_patterns", [])
            for pattern in search_patterns:
                nodes = knowledge_graph.query_nodes(name_pattern=pattern)
                results["nodes"].extend(nodes[:10])
            
            # 3. 分析连接关系
            for node in results["nodes"][:5]:  # 只分析前5个节点的连接
                connections = knowledge_graph.get_node_connections(node["id"])
                results["connections"].append({
                    "node_id": node["id"],
                    "node_name": node.get("name", "unknown"),
                    "incoming": len(connections["incoming"]),
                    "outgoing": len(connections["outgoing"]),
                    "connections": connections
                })
            
            # 4. 获取统计信息
            results["statistics"] = knowledge_graph.get_statistics()
            
            # 更新任务状态
            updated_tasks = state["sub_tasks"].copy()
            updated_tasks[current_index] = {
                **current_task,
                "status": "completed",
                "results": results
            }
            
            # 添加到分析结果
            analysis_results = state["analysis_results"].copy()
            analysis_results.append({
                "task_id": current_task["task_id"],
                "task_description": current_task["description"],
                "results": results,
                "insights": self._extract_insights(results, current_task)
            })
            
            logger.info(f"图谱查询完成: 找到 {len(results['nodes'])} 个节点，{len(results['connections'])} 个连接")
            
            return {
                **state,
                "sub_tasks": updated_tasks,
                "analysis_results": analysis_results,
                "messages": [AIMessage(content=f"任务 {current_index + 1} 查询完成")]
            }
            
        except Exception as e:
            logger.error(f"图谱查询失败: {str(e)}")
            
            # 创建空结果
            empty_results = {
                "nodes": [],
                "connections": [],
                "statistics": {},
                "error": str(e)
            }
            
            updated_tasks = state["sub_tasks"].copy()
            updated_tasks[current_index] = {
                **current_task,
                "status": "failed",
                "results": empty_results
            }
            
            analysis_results = state["analysis_results"].copy()
            analysis_results.append({
                "task_id": current_task["task_id"],
                "task_description": current_task["description"],
                "results": empty_results,
                "insights": ["查询失败，无法获取结果"]
            })
            
            return {
                **state,
                "sub_tasks": updated_tasks,
                "analysis_results": analysis_results,
                "messages": [AIMessage(content=f"任务 {current_index + 1} 查询失败")]
            }
    
    def synthesizer_node(self, state: AgentState) -> AgentState:
        """
        综合报告生成节点：汇总所有分析结果生成最终报告
        """
        logger.info("生成综合分析报告...")
        
        prompt = PromptTemplate.from_template("""
        你是一个专业的技术文档撰写专家。请根据以下代码分析结果，撰写一份详尽的技术分析报告。

        原始问题: {user_question}

        分析结果:
        {analysis_results}

        请撰写一份结构清晰、内容详实的技术分析报告，包括：

        1. **执行摘要** - 问题的简要总结和主要发现
        2. **详细分析** - 对每个子任务的详细分析结果
        3. **代码结构洞察** - 从代码结构角度得出的重要发现
        4. **技术建议** - 基于分析结果的技术建议（如果适用）
        5. **结论** - 总结性结论

        请确保：
        - 使用专业的技术语言
        - 结构清晰，逻辑连贯
        - 基于实际的分析数据
        - 提供有价值的洞察

        报告应该在1000-2000字之间，使用Markdown格式。
        """)
        
        try:
            # 格式化分析结果
            formatted_results = []
            for i, result in enumerate(state["analysis_results"]):
                formatted_result = f"""
                任务 {i+1}: {result['task_description']}
                - 找到节点数: {len(result['results'].get('nodes', []))}
                - 连接关系数: {len(result['results'].get('connections', []))}
                - 关键洞察: {', '.join(result.get('insights', ['无']))}
                """
                formatted_results.append(formatted_result)
            
            response = self.llm.invoke(prompt.format(
                user_question=state["user_question"],
                analysis_results='\n'.join(formatted_results)
            ))
            
            final_report = response.content
            
            logger.success("综合分析报告生成完成")
            
            return {
                **state,
                "final_report": final_report,
                "messages": [AIMessage(content="综合分析报告已生成")]
            }
            
        except Exception as e:
            logger.error(f"报告生成失败: {str(e)}")
            
            # 生成简单的报告
            simple_report = f"""
            # 代码分析报告
            
            ## 问题
            {state["user_question"]}
            
            ## 分析概述
            完成了 {len(state["analysis_results"])} 个分析任务。
            
            ## 结果
            {"由于技术问题，无法生成详细报告。请查看原始分析数据。" if str(e) else "分析完成。"}
            
            ## 错误信息
            {str(e) if str(e) else "无"}
            """
            
            return {
                **state,
                "final_report": simple_report,
                "messages": [AIMessage(content="生成了简化版报告")]
            }
    
    def should_continue_analysis(self, state: AgentState) -> str:
        """
        条件边：判断是否继续分析下一个任务
        """
        current_index = state["current_task_index"]
        total_tasks = len(state["sub_tasks"])
        
        if current_index >= total_tasks:
            return "finish"
        
        current_task = state["sub_tasks"][current_index]
        
        if current_task["status"] in ["pending", "analyzing"]:
            return "continue"
        elif current_task["status"] in ["completed", "failed"]:
            # 移动到下一个任务
            state["current_task_index"] = current_index + 1
            if state["current_task_index"] >= total_tasks:
                return "finish"
            else:
                return "next_task"
        
        return "finish"
    
    def _extract_insights(self, results: Dict[str, Any], task: Dict[str, str]) -> List[str]:
        """
        从查询结果中提取关键洞察
        """
        insights = []
        
        try:
            nodes = results.get("nodes", [])
            connections = results.get("connections", [])
            
            # 节点统计洞察
            if nodes:
                node_types = {}
                for node in nodes:
                    node_type = node.get("type", "unknown")
                    node_types[node_type] = node_types.get(node_type, 0) + 1
                
                insights.append(f"发现 {len(nodes)} 个代码元素，包括: {', '.join([f'{k}({v})' for k, v in node_types.items()])}")
            
            # 连接关系洞察
            if connections:
                total_incoming = sum(conn["incoming"] for conn in connections)
                total_outgoing = sum(conn["outgoing"] for conn in connections)
                insights.append(f"分析了 {len(connections)} 个节点的连接关系，总入度: {total_incoming}，总出度: {total_outgoing}")
                
                # 找出连接最多的节点
                if connections:
                    max_conn = max(connections, key=lambda x: x["incoming"] + x["outgoing"])
                    insights.append(f"连接最密集的节点: {max_conn['node_name']} (入度:{max_conn['incoming']}, 出度:{max_conn['outgoing']})")
            
            if not insights:
                insights.append("未发现明显的代码结构模式")
                
        except Exception as e:
            insights.append(f"洞察提取失败: {str(e)}")
        
        return insights
    
    def analyze(self, user_question: str, knowledge_graph: Any) -> Dict[str, Any]:
        """
        执行完整的多智能体分析流程
        
        Args:
            user_question: 用户问题
            knowledge_graph: 知识图谱实例
            
        Returns:
            分析结果
        """
        logger.info(f"开始多智能体分析: {user_question}")
        
        try:
            # 初始化状态
            initial_state = AgentState(
                user_question=user_question,
                knowledge_graph=knowledge_graph,
                sub_tasks=[],
                current_task_index=0,
                analysis_results=[],
                final_report="",
                messages=[]
            )
            
            # 执行图
            result = self.graph.invoke(initial_state)
            
            # 整理返回结果
            return {
                "success": True,
                "user_question": user_question,
                "sub_tasks": result["sub_tasks"],
                "analysis_results": result["analysis_results"],
                "final_report": result["final_report"],
                "message": "分析完成"
            }
            
        except Exception as e:
            error_msg = f"多智能体分析失败: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "error": error_msg,
                "message": "分析失败"
            }
