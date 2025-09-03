import json
import time
import os
from typing import List, Dict, Any, Optional, Union
from openai import OpenAI

class BatchProcessor:
    """
    用于处理OpenAI API的批处理请求
    """
    
    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv('OPENAI_API_KEY'),
        )
        
    def process_batch(self, tasks: List[Dict[str, Any]], batch_size: int = 5, model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
        """
        处理批量请求
        
        Args:
            tasks: 任务列表，每个任务包含消息和自定义ID
            batch_size: 每批处理的任务数量
            model: 使用的模型名称
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 分批处理
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i+batch_size]
            batch_results = []
            
            # 处理每个批次
            for task in batch:
                try:
                    # 创建请求
                    completion = self.client.chat.completions.create(
                        model=model,
                        messages=[
                            {
                                "role": msg["role"],
                                "content": msg["content"]
                            } for msg in task["messages"]
                        ],
                        temperature=task.get("temperature", 0.7),
                        max_tokens=task.get("max_tokens", 500),
                    )
                    
                    # 处理响应
                    content = None
                    if completion.choices and completion.choices[0].message:
                        content = completion.choices[0].message.content
                        
                    batch_results.append({
                        "custom_id": task["custom_id"],
                        "content": content
                    })
                    
                    # 避免API限制
                    time.sleep(0.5)
                except Exception as e:
                    print(f"Error processing task {task['custom_id']}: {e}")
                    batch_results.append({
                        "custom_id": task["custom_id"],
                        "error": str(e)
                    })
            
            # 添加批次结果
            results.extend(batch_results)
            
            # 避免API限制
            if i + batch_size < len(tasks):
                time.sleep(1)
                
        return results
    
    def create_batch_file(self, tasks: List[Dict[str, Any]], file_name: str) -> str:
        """
        创建批处理文件
        
        Args:
            tasks: 任务列表
            file_name: 文件名
            
        Returns:
            文件路径
        """
        # 创建JSONL文件
        with open(file_name, 'w') as file:
            for task in tasks:
                file.write(json.dumps(task) + '\n')
                
        return file_name
    
    def process_with_batch_api(self, tasks: List[Dict[str, Any]], model: str = "gpt-4o-mini") -> List[Dict[str, Any]]:
        """
        使用OpenAI批处理API处理请求
        
        Args:
            tasks: 任务列表
            model: 使用的模型名称
            
        Returns:
            处理结果列表
        """
        # 创建批处理任务
        batch_tasks = []
        for i, task in enumerate(tasks):
            batch_task = {
                "custom_id": task["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": task["messages"],
                    "temperature": task.get("temperature", 0.7),
                    "max_tokens": task.get("max_tokens", 500),
                }
            }
            batch_tasks.append(batch_task)
            
        # 创建批处理文件
        file_name = "batch_tasks.jsonl"
        self.create_batch_file(batch_tasks, file_name)
        
        try:
            # 上传文件
            batch_file = self.client.files.create(
                file=open(file_name, "rb"),
                purpose="batch"
            )
            
            # 创建批处理作业
            batch_job = self.client.batches.create(
                input_file_id=batch_file.id,
                endpoint="/v1/chat/completions",
                completion_window="24h"
            )
            
            # 等待作业完成
            job_status = "processing"
            while job_status != "completed":
                batch_job = self.client.batches.retrieve(batch_job.id)
                job_status = batch_job.status
                
                if job_status == "failed":
                    raise Exception(f"Batch job failed: {batch_job}")
                    
                print(f"Job status: {job_status}")
                time.sleep(10)
                
            # 获取结果
            if batch_job.output_file_id:
                result_file_id = batch_job.output_file_id
                result = self.client.files.content(result_file_id).content
                
                # 解析结果
                results = []
                for line in result.decode('utf-8').strip().split('\n'):
                    json_result = json.loads(line)
                    results.append({
                        "custom_id": json_result["custom_id"],
                        "content": json_result["response"]["body"]["choices"][0]["message"]["content"]
                    })
                    
                return results
            else:
                raise Exception("No output file ID found in batch job")
        except Exception as e:
            print(f"Error processing batch: {e}")
            # 如果批处理API失败，回退到普通处理
            return self.process_batch(tasks, model=model)
        finally:
            # 清理文件
            if os.path.exists(file_name):
                os.remove(file_name) 