#!/usr/bin/env python3
"""
Antigravity MCP Server v1.5 - MLX Unified Edition
Gemini強化のためのModel Context Protocolサーバー + HTTP API
全AI推論は MLX (RyotaOS Runtime :9001) に統一。

提供ツール (19個):
- eck_read / eck_write: ECK履歴管理
- compress / expand / search_rag: 圧縮エンジン・RAG
- dod_audit: DoD監査
- pcc_encode: PCC座標生成
- local_ai: ローカルLLM呼び出し (MLX)
- titan_ai: Titan Core AI呼び出し (MLX)
- mothership_architect / creative / engineer: Mothership AI群
- mcp_ryotaos_run / metrics: RyotaOS Runtime
- titan_batch: 並列バッチ処理
- get_stats / slack_notify / simulation_run / ryota_core_memory

HTTP API:
- POST /api/generate  - テキスト生成 (MLX)
- POST /api/compress  - PCC圧縮
- POST /api/batch     - 並列バッチ処理
- GET  /api/health    - ヘルスチェック
- GET  /api/stats     - システム統計
- POST /api/tool/:name - 任意のツール呼び出し

Usage:
    python server.py              # MCPサーバーとして起動
    python server.py --http       # HTTPサーバーとして起動 (ポート9000)
    python server.py --http --port 8080  # カスタムポート
    python server.py --test       # ツールテスト実行
    python server.py --version    # バージョン表示

Author: Antigravity Team
Version: 1.5.0
"""

import json
import asyncio
import sys
import os
import argparse
import logging
from datetime import datetime
from typing import Any, Optional, Dict, List
from pathlib import Path

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# バージョン
VERSION = "1.5.0"

# MCP SDK インポート
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent, Resource, ReadResourceResult
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP SDK not installed. Run: pip install mcp")

# HTTP Server (aiohttp)
try:
    from aiohttp import web
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    logger.warning("aiohttp not installed. Run: pip install aiohttp")


# === 設定ロード ===

def load_config() -> Dict:
    """設定を読み込む"""
    return {
        "eck_base_path": os.environ.get(
            "ANTIGRAVITY_ECK_PATH",
            os.path.expanduser("~/.gemini/antigravity/knowledge")
        ),
        "compression_engine_path": os.environ.get(
            "ANTIGRAVITY_COMPRESSION_PATH",
            os.path.expanduser(
                "~/compression_engine"
            )
        ),
        "mlx_url": os.environ.get("MLX_URL", "http://localhost:9001"),
        "default_model": os.environ.get("MLX_MODEL", "phi4"),
        "reasoning_model": os.environ.get("MLX_REASONING_MODEL", "deepseek-r1:14b"),
        "slack_webhook_url": os.environ.get("SLACK_WEBHOOK_URL", ""),
    }


def validate_config(config: Dict) -> List[str]:
    """設定を検証する"""
    errors = []
    eck_path = Path(config["eck_base_path"])
    if not eck_path.exists():
        errors.append(f"ECK path not found: {eck_path}")
    engine_path = Path(config["compression_engine_path"])
    if not engine_path.exists():
        errors.append(f"Compression engine not found: {engine_path}")
    return errors


def run_subprocess(cmd: list, cwd: str = None, timeout: int = 60) -> Dict:
    """サブプロセスを実行 (Secure)"""
    import subprocess
    import shutil
    
    # SECURITY: Ensure command is not a shell injection
    if not cmd:
        return {"status": "error", "error": "Empty command"}
        
    executable = shutil.which(cmd[0])
    if not executable:
         return {"status": "error", "error": f"Command not found: {cmd[0]}"}
    
    # Use absolute path for executable
    safe_cmd = [executable] + cmd[1:]
    
    try:
        result = subprocess.run(safe_cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return {
            "status": "ok" if result.returncode == 0 else "error",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def extract_pcc_coord(text: str) -> str:
    """テキストからPCC座標を抽出"""
    import re
    pattern = r'I\d[F]\d[C]\d[B]\d[R]\d[M]\d[E]\d[N]\d[S]\d'
    match = re.search(pattern, text.upper())
    if match:
        return match.group()
    return "I5F5C5B5R5M5E5N5S5"


# === ツール実装クラス ===

class AntigravityTools:
    """Antigravity ツール集"""
    
    def __init__(self, config: Dict):
        self.config = config
        logger.info(f"AntigravityTools initialized with config: {list(config.keys())}")
    
    async def eck_read(self, path: str = "", limit: int = 10) -> Dict:
        """ECK/KI履歴を読み込む"""
        base = Path(self.config["eck_base_path"])
        target = base / path if path else base
        
        if not target.exists():
            return {"error": f"Path not found: {target}"}
        
        if target.is_file():
            content = target.read_text(encoding="utf-8", errors="ignore")
            return {"path": str(target), "content": content[:10000]}
        
        items = []
        for item in sorted(target.iterdir())[:limit]:
            items.append({
                "name": item.name,
                "type": "dir" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None
            })
        return {"path": str(target), "items": items}
    
    async def eck_write(self, path: str, content: str, append: bool = True) -> Dict:
        """ECKに追記する"""
        base = Path(self.config["eck_base_path"])
        target = base / path
        target.parent.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().isoformat()
        entry = f"\n---\n[{timestamp}]\n{content}\n"
        
        mode = "a" if append else "w"
        with open(target, mode, encoding="utf-8") as f:
            f.write(entry if append else content)
        
        logger.info(f"ECK write: {target}")
        return {"status": "ok", "path": str(target), "timestamp": timestamp}
    
    async def compress(self, file_path: str, offline: bool = True, use_deepseek: bool = False) -> Dict:
        """圧縮エンジンでファイルを圧縮"""
        engine_path = self.config["compression_engine_path"]
        cmd = ["python3", "compression_engine.py", "ingest", file_path]
        if use_deepseek:
            cmd.append("--deepseek")
        elif offline:
            cmd.append("--offline")
        
        result = run_subprocess(cmd, engine_path, timeout=180)
        logger.info(f"Compress: {file_path} -> {result['status']}")
        return result

    async def mcp_antigravity_harvest_codebase(self, target_path: str, repo_name: str) -> Dict:
        """指定されたディレクトリ全体をNeural Packetとして抽出し、圧縮空間に格納する"""
        engine_path = os.path.expanduser("~/fusion-gate")
        target_abs = os.path.abspath(os.path.expanduser(target_path))
        
        # ターゲットにJS/TS系（App/IDE）があればharvest_js_packetsを優先、
        # 他コードであればneural_packet.py --harvestが望ましいが、現在harvest_js_packets.pyが最新
        # 今回はharvest_js_packets.pyによる包括的な抽出スクリプトを実行する
        cmd = ["python3", "harvest_js_packets.py", target_abs, repo_name]
        
        logger.info(f"Starting Codebase Harvest for {repo_name} at {target_abs}")
        result = run_subprocess(cmd, engine_path, timeout=600)  # 最大10分
        
        if result["status"] == "ok":
            logger.info(f"Harvest Codebase Success: {repo_name}")
            return {"status": "ok", "repo_name": repo_name, "details": result["stdout"]}
        else:
            logger.error(f"Harvest Codebase Failed: {result.get('error') or result.get('stderr')}")
            return {"status": "error", "error": result.get("error") or result.get("stderr")}
    
    async def expand(self, entry_id: str) -> Dict:
        """圧縮エントリを展開"""
        engine_path = self.config["compression_engine_path"]
        cmd = ["python3", "compression_engine.py", "expand", entry_id]
        return run_subprocess(cmd, engine_path, timeout=30)
    
    async def search_rag(self, query: str, n: int = 5) -> Dict:
        """RAG検索"""
        engine_path = self.config["compression_engine_path"]
        cmd = ["python3", "compression_engine.py", "search", query, "-n", str(n)]
        return run_subprocess(cmd, engine_path, timeout=60)
    
    async def dod_audit(self, file_path: str) -> Dict:
        """DoD監査を実行"""
        mothership_path = os.path.expanduser("~/.gemini/antigravity/scratch/mothership_simulation")
        cmd = ["python3", "dod_audit_engine.py", file_path, "--json"]
        result = run_subprocess(cmd, mothership_path, timeout=60)
        
        if result["status"] == "ok" and result.get("stdout"):
            try:
                report = json.loads(result["stdout"])
                return {"status": "ok", "report": report}
            except json.JSONDecodeError:
                pass
        return result
    
    async def pcc_encode(self, text: str) -> Dict:
        """テキストをPCC 9軸座標に変換 (MLX)"""
        import requests
        
        url = f"{self.config['mlx_url']}/generate"
        full_prompt = f"System: PCC座標をI5F7C3B8R2M4E1N6S5形式で返してください。\n\nUser: PCC座標を生成 (I,F,C,B,R,M,E,N,S各0-9):\n\n{text[:1000]}"
        
        try:
            resp = requests.post(
                url,
                json={
                    "prompt": full_prompt,
                    "model": self.config["default_model"],
                    "max_tokens": 256,
                    "temp": 0.5
                },
                timeout=60
            )
            
            if resp.status_code == 200:
                content = resp.json().get("response", "")
                coord = extract_pcc_coord(content)
                return {"status": "ok", "pcc_coord": coord}
            else:
                return {"status": "error", "error": f"MLX error: {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def local_ai(self, prompt: str, model: Optional[str] = None, 
                       system: Optional[str] = None) -> Dict:
        """ローカルLLMを呼び出す (MLX via RyotaOS Runtime)"""
        import requests
        
        model = model or self.config["default_model"]
        url = f"{self.config['mlx_url']}/generate"
        
        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\nUser: {prompt}"
        
        payload = {
            "prompt": full_prompt,
            "model": model,
            "max_tokens": 1024,
            "temp": 0.7
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=120)
            
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": "ok",
                    "response": data.get("response", ""),
                    "model": model,
                    "tokens": data.get("tokens", 0),
                    "engine": "mlx"
                }
            else:
                return {"status": "error", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def get_stats(self) -> Dict:
        """システム統計を取得"""
        import shutil
        
        stats = {"timestamp": datetime.now().isoformat(), "version": VERSION, "engine": "mlx"}
        
        try:
            total, used, free = shutil.disk_usage("/")
            stats["disk_root"] = {
                "total_gb": round(total / (1024**3), 1),
                "used_gb": round(used / (1024**3), 1),
                "free_gb": round(free / (1024**3), 1)
            }
        except Exception:
            stats["disk_root"] = "error"
        
        try:
            total, used, free = shutil.disk_usage("/Volumes/SSD")
            stats["disk_ssd"] = {
                "total_gb": round(total / (1024**3), 1),
                "used_gb": round(used / (1024**3), 1),
                "free_gb": round(free / (1024**3), 1)
            }
        except Exception:
            stats["disk_ssd"] = "not_mounted"
        
        try:
            import requests
            resp = requests.get(f"{self.config['mlx_url']}/health", timeout=5)
            if resp.status_code == 200:
                stats["mlx_runtime"] = resp.json()
            else:
                stats["mlx_runtime"] = "unavailable"
        except Exception:
            stats["mlx_runtime"] = "unavailable"
        
        return stats
    
    async def slack_notify(self, message: str, channel: Optional[str] = None) -> Dict:
        """Slack通知を送信"""
        import requests
        
        webhook_url = self.config.get("slack_webhook_url", "")
        if not webhook_url:
            return {"status": "error", "error": "SLACK_WEBHOOK_URL not configured"}
        
        payload = {"text": message}
        if channel:
            payload["channel"] = channel
        
        try:
            resp = requests.post(webhook_url, json=payload, timeout=10)
            if resp.status_code == 200:
                return {"status": "ok", "message": "Notification sent"}
            else:
                return {"status": "error", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def mothership_audit(self, file_path: str, full_pipeline: bool = True) -> Dict:
        """Mothershipフルパイプラインを実行"""
        mothership_path = os.path.expanduser("~/.gemini/antigravity/scratch/mothership_simulation")
        
        if full_pipeline:
            cmd = ["python3", "run_full_pipeline_v2.py", file_path, "--json"]
        else:
            cmd = ["python3", "dod_audit_engine.py", file_path, "--json"]
        
        result = run_subprocess(cmd, mothership_path, timeout=120)
        
        if result["status"] == "ok" and result.get("stdout"):
            try:
                report = json.loads(result["stdout"])
                return {"status": "ok", "report": report}
            except json.JSONDecodeError:
                return {"status": "ok", "raw_output": result["stdout"]}
        
        return result
    
    async def simulation_run(self, command: str, context: Optional[str] = None) -> Dict:
        """AI-in-AIシミュレーションを実行"""
        mothership_path = os.path.expanduser("~/.gemini/antigravity/scratch/mothership_simulation")
        
        payload = {"command": command, "context": context or "", "mode": "simulation"}
        cmd = ["python3", "simulation_kernel.py", "--input", json.dumps(payload)]
        result = run_subprocess(cmd, mothership_path, timeout=60)
        
        if result["status"] == "ok" and result.get("stdout"):
            try:
                sim_result = json.loads(result["stdout"])
                return {"status": "ok", "simulation": sim_result, "command": command}
            except json.JSONDecodeError:
                pass
        
        # フォールバック
        dangerous_patterns = ["rm -rf /", "sudo rm", "> /dev/", "mkfs", "dd if="]
        is_dangerous = any(p in command for p in dangerous_patterns)
        
        return {
            "status": "ok",
            "simulation": {
                "safe": not is_dangerous,
                "confidence": 0.7,
                "warning": "Fallback mode - kernel not available"
            },
            "command": command
        }
    
    async def ryota_core_memory(self, query: str, section: Optional[str] = None) -> Dict:
        """Aladdin設計図にアクセス"""
        knowledge_path = Path(self.config["eck_base_path"]) / "ryota_core_os_2026"
        blueprint_path = knowledge_path / "artifacts" / "master_blueprint.md"
        mcp_strategy_path = knowledge_path / "artifacts" / "mcp_integration_strategy.md"
        
        result = {"query": query, "matches": []}
        
        for doc_path in [blueprint_path, mcp_strategy_path]:
            if doc_path.exists():
                content = doc_path.read_text(encoding="utf-8")
                for line in content.split("\n"):
                    if query.lower() in line.lower():
                        result["matches"].append({
                            "source": doc_path.name,
                            "content": line.strip()
                        })
        
        if section:
            section_map = {
                "cerebrum": ["Antigravity", "MLX", "vLLM", "OpenHands", "DSPy", "LangGraph"],
                "nervous": ["MCP", "Temporal", "n8n", "Dagster", "Ray", "LiveKit"],
                "senses": ["Computer Use", "AppleScript", "Playwright", "Firecrawl", "Tavily", "FFmpeg"],
                "memory": ["Mem0", "MinIO", "Supabase", "Redis", "VectorDB", "DuckDB", "LlamaIndex"],
                "infra": ["Coolify", "OrbStack", "Tailscale", "Cloudflare", "Grafana", "Modal"]
            }
            if section.lower() in section_map:
                result["section"] = section
                result["components"] = section_map[section.lower()]
        
        return {"status": "ok", **result}
    
    async def titan_ai(self, prompt: str, model: Optional[str] = None,
                       system: Optional[str] = None) -> Dict:
        """Mac Studio Titan Core AIを呼び出す (via RyotaOS MLX)"""
        import requests
        
        # RyotaOS Runtime (MLX) Endpoint
        url = f"{self.config['mlx_url']}/generate"
        model = model or "phi4"  # Default MLX model
        
        # Construct prompt with system if present
        full_prompt = prompt
        if system:
            full_prompt = f"System: {system}\n\nUser: {prompt}"
        
        payload = {
            "prompt": full_prompt,
            "model": model,
            "max_tokens": 1024,
            "temp": 0.7
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=120)
            
            if resp.status_code == 200:
                data = resp.json()
                return {
                    "status": "ok",
                    "response": data.get("response", ""),
                    "model": data.get("model", model),
                    "tokens": data.get("tokens", 0),
                    "titan_status": "connected_mlx",
                    "host": "mac-studio-m3-ultra-512gb"
                }
            else:
                return {"status": "error", "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def mothership_architect(self, query: str, context: Optional[str] = None) -> Dict:
        """Mothership Architect (DeepSeek-R1)"""
        prompt = f"Context:\n{context}\n\nProblem:\n{query}" if context else query
        return await self.titan_ai(
            prompt=prompt,
            model="/Users/ryyota/Models/DeepSeek-R1-0528-4bit",
            system="You are Mothership Architect. Analyze deeply and provide comprehensive system design."
        )

    async def mothership_creative(self, prompt: str, tone: Optional[str] = None) -> Dict:
        """Mothership Creative (Qwen3-235B)"""
        full_prompt = f"Tone: {tone}\n\n{prompt}" if tone else prompt
        return await self.titan_ai(
            prompt=full_prompt,
            model="/Users/ryyota/Models/Qwen3-235B-A22B-8bit",
            system="You are Mothership Scribe. Write extensive, nuanced content."
        )

    async def mothership_engineer(self, instruction: str, code_snippet: Optional[str] = None) -> Dict:
        """Mothership Engineer (Qwen3-235B - Upgraded)"""
        full_prompt = f"Task: {instruction}\n\nCode:\n```\n{code_snippet}\n```" if code_snippet else instruction
        return await self.titan_ai(
            prompt=full_prompt,
            model="/Users/ryyota/Models/Qwen3-235B-A22B-8bit",
            system="You are Mothership Engineer. Analyze deeply and output clean, correct, secure code. Provide comprehensive code reviews."
        )

    async def mcp_ryotaos_run(self, query: str, model: str = "phi4", 
                              policy_lock: bool = False, priority: str = "normal") -> Dict:
        """RyotaOS Runtime (Titan Core) でタスクを実行"""
        import requests
        
        # 安全性設定に基づいてエンドポイントを選択
        endpoint = "/run" if policy_lock else "/run/safe"
        url = f"http://localhost:9001{endpoint}"
        
        payload = {
            "query": query,
            "model": model,
            "priority": priority,
            "policy_lock": policy_lock
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                return {"status": "ok", "job": resp.json()}
            elif resp.status_code == 403:
                return {"status": "error", "error": "Policy Lock Denied (use policy_lock=True)"}
            else:
                return {"status": "error", "error": f"HTTP {resp.status_code}: {resp.text}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def mcp_ryotaos_metrics(self) -> Dict:
        """RyotaOS Runtime のシステムメトリクスを取得"""
        import requests
        
        url = "http://localhost:9001/metrics/json"
        
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                return {"status": "ok", "metrics": resp.json()}
            else:
                return {"status": "error", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def titan_batch(self, tasks: List[Dict], 
                          concurrency: int = 8,
                          model: Optional[str] = None) -> Dict:
        """
        並列バッチ処理 (M3 Ultra 512GB 最適化)
        
        Args:
            tasks: [{"prompt": "...", "system": "..."}] のリスト
            concurrency: 同時実行数 (デフォルト8、最大16)
            model: 使用モデル (デフォルト: phi4)
        
        Returns:
            {"status": "ok", "results": [...], "stats": {...}}
        """
        import time
        import aiohttp
        
        if not tasks:
            return {"status": "error", "error": "No tasks provided"}
        
        # 同時実行数を制限 (M3 Ultra: 8-16が最適)
        concurrency = min(max(1, concurrency), 16)
        model = model or "phi4"
        
        semaphore = asyncio.Semaphore(concurrency)
        start_time = time.time()
        
        async def process_task(idx: int, task: Dict) -> Dict:
            async with semaphore:
                prompt = task.get("prompt", "")
                system = task.get("system")
                
                if not prompt:
                    return {"idx": idx, "status": "error", "error": "Empty prompt"}
                
                # RyotaOS MLX endpoint
                url = "http://localhost:9001/generate"
                full_prompt = f"System: {system}\n\nUser: {prompt}" if system else prompt
                
                payload = {
                    "prompt": full_prompt,
                    "model": model,
                    "max_tokens": task.get("max_tokens", 1024),
                    "temp": task.get("temp", 0.7)
                }
                
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                return {
                                    "idx": idx,
                                    "status": "ok",
                                    "response": data.get("response", ""),
                                    "tokens": data.get("tokens", 0)
                                }
                            else:
                                return {"idx": idx, "status": "error", "error": f"HTTP {resp.status}"}
                except asyncio.TimeoutError:
                    return {"idx": idx, "status": "error", "error": "Timeout"}
                except Exception as e:
                    return {"idx": idx, "status": "error", "error": str(e)}
        
        # 全タスクを並列実行
        results = await asyncio.gather(*[
            process_task(i, task) for i, task in enumerate(tasks)
        ])
        
        # 結果をソート (idx順)
        results = sorted(results, key=lambda x: x.get("idx", 0))
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r.get("status") == "ok")
        total_tokens = sum(r.get("tokens", 0) for r in results)
        
        return {
            "status": "ok",
            "results": results,
            "stats": {
                "total_tasks": len(tasks),
                "success": success_count,
                "failed": len(tasks) - success_count,
                "elapsed_seconds": round(elapsed, 2),
                "tokens_total": total_tokens,
                "concurrency": concurrency,
                "model": model
            }
        }

    async def pcc_critic_run(self, prompt: str, preset: str = "探",
                              runtime: str = "gemini", model: str = "deep",
                              timeout: int = 120) -> Dict:
        """PCC制約注入でAIの本気を引き出すcriticパイプライン"""
        pcc_path = os.path.join(os.path.dirname(__file__), "pcc_critic.py")
        if not os.path.exists(pcc_path):
            return {"status": "error", "error": f"pcc_critic.py not found at {pcc_path}"}

        cmd = [
            "python3", pcc_path,
            "--preset", preset,
            "--runtime", runtime,
            "--model", model,
            "--timeout", str(timeout),
            "--json",
            prompt
        ]
        result = run_subprocess(cmd, os.path.dirname(pcc_path), timeout=timeout + 30)

        if result["status"] == "ok" and result.get("stdout"):
            try:
                return {"status": "ok", "result": json.loads(result["stdout"])}
            except json.JSONDecodeError:
                return {"status": "ok", "raw_output": result["stdout"]}
        return result

    async def pcc_critic_audit(self, text: str) -> Dict:
        """テキストの品質を監査する（LLM呼び出しなし）"""
        pcc_path = os.path.join(os.path.dirname(__file__), "pcc_critic.py")
        if not os.path.exists(pcc_path):
            return {"status": "error", "error": f"pcc_critic.py not found at {pcc_path}"}

        cmd = ["python3", pcc_path, "--audit-only", "--json", text]
        result = run_subprocess(cmd, os.path.dirname(pcc_path), timeout=10)

        if result["status"] == "ok" and result.get("stdout"):
            try:
                return {"status": "ok", "audit": json.loads(result["stdout"])}
            except json.JSONDecodeError:
                return {"status": "ok", "raw_output": result["stdout"]}
        return result


# === HTTP API ===


def create_http_app(tools: AntigravityTools) -> 'web.Application':
    """HTTP APIアプリケーションを作成"""
    
    async def health_handler(request):
        """GET /api/health"""
        return web.json_response({
            "status": "ok",
            "version": VERSION,
            "timestamp": datetime.now().isoformat(),
            "host": "mac-studio-titan-core"
        })
    
    async def stats_handler(request):
        """GET /api/stats"""
        result = await tools.get_stats()
        return web.json_response(result)
    
    async def generate_handler(request):
        """POST /api/generate"""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        prompt = data.get("prompt", "")
        model = data.get("model")
        system = data.get("system")
        
        if not prompt:
            return web.json_response({"error": "prompt is required"}, status=400)
        
        result = await tools.local_ai(prompt=prompt, model=model, system=system)
        return web.json_response(result)
    
    async def compress_handler(request):
        """POST /api/compress"""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        text = data.get("text", "")
        if not text:
            return web.json_response({"error": "text is required"}, status=400)
        
        result = await tools.pcc_encode(text=text)
        return web.json_response(result)
    
    async def batch_handler(request):
        """POST /api/batch - 並列バッチ処理"""
        try:
            data = await request.json()
        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON"}, status=400)
        
        tasks = data.get("tasks", [])
        if not tasks:
            return web.json_response({"error": "tasks array is required"}, status=400)
        
        concurrency = data.get("concurrency", 8)
        model = data.get("model")
        
        result = await tools.titan_batch(tasks=tasks, concurrency=concurrency, model=model)
        return web.json_response(result)
    
    async def tool_handler(request):
        """POST /api/tool/{name}"""
        tool_name = request.match_info.get("name")
        
        try:
            data = await request.json()
        except json.JSONDecodeError:
            data = {}
        
        handler = getattr(tools, tool_name, None)
        if not handler:
            return web.json_response({"error": f"Unknown tool: {tool_name}"}, status=404)
        
        try:
            result = await handler(**data)
            return web.json_response(result)
        except Exception as e:
            logger.error(f"Tool error: {tool_name} -> {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    # CORSミドルウェア
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            response = await handler(request)
        
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response
    
    app = web.Application(middlewares=[cors_middleware])
    
    # ルート登録
    app.router.add_get("/api/health", health_handler)
    app.router.add_get("/api/stats", stats_handler)
    app.router.add_post("/api/generate", generate_handler)
    app.router.add_post("/api/compress", compress_handler)
    app.router.add_post("/api/batch", batch_handler)
    app.router.add_post("/api/tool/{name}", tool_handler)
    
    # ルート一覧
    async def index_handler(request):
        return web.json_response({
            "name": "Antigravity MCP Server",
            "version": VERSION,
            "endpoints": [
                {"method": "GET", "path": "/api/health", "description": "ヘルスチェック"},
                {"method": "GET", "path": "/api/stats", "description": "システム統計"},
                {"method": "POST", "path": "/api/generate", "description": "テキスト生成", "body": {"prompt": "string", "model": "string?", "system": "string?"}},
                {"method": "POST", "path": "/api/compress", "description": "PCC圧縮", "body": {"text": "string"}},
                {"method": "POST", "path": "/api/batch", "description": "並列バッチ処理", "body": {"tasks": "[{prompt, system?}]", "concurrency": "int?", "model": "string?"}},
                {"method": "POST", "path": "/api/tool/{name}", "description": "任意のツール呼び出し"},
            ],
            "available_tools": [
                "eck_read", "eck_write", "compress", "expand", "search_rag",
                "dod_audit", "pcc_encode", "local_ai", "get_stats", "slack_notify",
                "mothership_audit", "simulation_run", "ryota_core_memory", "titan_ai",
                "mcp_ryotaos_run", "mcp_ryotaos_metrics", "titan_batch",
                "pcc_critic_run", "pcc_critic_audit"
            ]
        })
    
    app.router.add_get("/", index_handler)
    
    return app


async def run_http_server(tools: AntigravityTools, port: int = 9000) -> None:
    """HTTPサーバーを実行"""
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not installed. Run: pip install aiohttp")
    
    app = create_http_app(tools)
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    
    logger.info(f"HTTP API server running at http://0.0.0.0:{port}")
    logger.info(f"Endpoints: /api/health, /api/generate, /api/compress, /api/stats")
    
    # 永続的に実行
    while True:
        await asyncio.sleep(3600)


# === テスト関数 ===

async def run_tests(tools: AntigravityTools) -> None:
    """ツールのテストを実行"""
    print("=== Antigravity MCP Server Test ===\n")
    
    print("📊 Testing get_stats()...")
    stats = await tools.get_stats()
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    assert "timestamp" in stats
    print("✅ get_stats passed\n")
    
    print("📚 Testing eck_read()...")
    eck = await tools.eck_read(limit=5)
    print(json.dumps(eck, indent=2, ensure_ascii=False))
    assert "path" in eck or "error" in eck
    print("✅ eck_read passed\n")
    
    print("=== All tests passed ===")


# === CLIパーサー ===

def create_parser() -> argparse.ArgumentParser:
    """CLIパーサーを作成"""
    parser = argparse.ArgumentParser(
        description="Antigravity MCP Server - Gemini強化のためのMCPサーバー",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python server.py              # MCPサーバーとして起動
    python server.py --http       # HTTPサーバーとして起動
    python server.py --http --port 8080  # カスタムポート
    python server.py --test       # ツールテスト実行
        """
    )
    
    parser.add_argument("--test", "-t", action="store_true", help="ツールテストを実行")
    parser.add_argument("--version", "-v", action="store_true", help="バージョン表示")
    parser.add_argument("--validate", action="store_true", help="設定検証")
    parser.add_argument("--http", action="store_true", help="HTTPサーバーモードで起動")
    parser.add_argument("--port", "-p", type=int, default=9000, help="HTTPサーバーポート (default: 9000)")
    
    return parser


# === MCPサーバー定義 ===

def create_mcp_server(tools: AntigravityTools):
    """MCPサーバーを作成"""
    if not MCP_AVAILABLE:
        raise RuntimeError("MCP SDK not available")
    
    server = Server("antigravity")
    
    @server.list_tools()
    async def list_tools() -> List['Tool']:
        return [
            Tool(name="eck_read", description="ECK/KI履歴読み込み", inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "limit": {"type": "integer", "default": 10}}
            }),
            Tool(name="eck_write", description="ECKに追記", inputSchema={
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "append": {"type": "boolean", "default": True}},
                "required": ["path", "content"]
            }),
            Tool(name="compress", description="圧縮エンジン呼び出し", inputSchema={
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "offline": {"type": "boolean", "default": True}, "use_deepseek": {"type": "boolean", "default": False}},
                "required": ["file_path"]
            }),
            Tool(name="mcp_antigravity_harvest_codebase", description="ソースコードベース全体をNeural Packetsとして圧縮・抽出", inputSchema={
                "type": "object",
                "properties": {
                    "target_path": {"type": "string", "description": "圧縮対象のディレクトリパス"},
                    "repo_name": {"type": "string", "description": "保存先のレッジ/リポジトリ名"}
                },
                "required": ["target_path", "repo_name"]
            }),
            Tool(name="expand", description="圧縮エントリ展開", inputSchema={
                "type": "object",
                "properties": {"entry_id": {"type": "string"}},
                "required": ["entry_id"]
            }),
            Tool(name="search_rag", description="RAG検索", inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "n": {"type": "integer", "default": 5}},
                "required": ["query"]
            }),
            Tool(name="dod_audit", description="DoD監査", inputSchema={
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            }),
            Tool(name="pcc_encode", description="PCC座標生成", inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }),
            Tool(name="local_ai", description="ローカルLLM呼び出し", inputSchema={
                "type": "object",
                "properties": {"prompt": {"type": "string"}, "model": {"type": "string"}, "system": {"type": "string"}},
                "required": ["prompt"]
            }),
            Tool(name="get_stats", description="システム統計", inputSchema={"type": "object", "properties": {}}),
            Tool(name="slack_notify", description="Slack通知", inputSchema={
                "type": "object",
                "properties": {"message": {"type": "string"}, "channel": {"type": "string"}},
                "required": ["message"]
            }),
            Tool(name="mothership_audit", description="Mothershipフルパイプライン", inputSchema={
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "full_pipeline": {"type": "boolean", "default": True}},
                "required": ["file_path"]
            }),
            Tool(name="simulation_run", description="AI-in-AIシミュレーション", inputSchema={
                "type": "object",
                "properties": {"command": {"type": "string"}, "context": {"type": "string"}},
                "required": ["command"]
            }),
            Tool(name="ryota_core_memory", description="Ryota-Core設計図", inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "section": {"type": "string"}},
                "required": ["query"]
            }),
            Tool(name="titan_ai", description="Titan Core AI呼び出し", inputSchema={
                "type": "object",
                "properties": {"prompt": {"type": "string"}, "model": {"type": "string"}, "system": {"type": "string"}},
                "required": ["prompt"]
            }),
            Tool(name="mothership_architect", description="Mothership Brain (DeepSeek-R1)", inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "context": {"type": "string"}},
                "required": ["query"]
            }),
            Tool(name="mothership_creative", description="Mothership Scribe (Qwen3-235B)", inputSchema={
                "type": "object",
                "properties": {"prompt": {"type": "string"}, "tone": {"type": "string"}},
                "required": ["prompt"]
            }),
            Tool(name="mothership_engineer", description="Mothership Builder (Qwen3-Coder)", inputSchema={
                "type": "object",
                "properties": {"instruction": {"type": "string"}, "code_snippet": {"type": "string"}},
                "required": ["instruction"]
            }),
            Tool(name="mcp_ryotaos_run", description="RyotaOSタスク実行", inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "model": {"type": "string", "default": "phi4"},
                    "policy_lock": {"type": "boolean", "default": False},
                    "priority": {"type": "string", "default": "normal"}
                },
                "required": ["query"]
            }),
            Tool(name="mcp_ryotaos_metrics", description="RyotaOSメトリクス", inputSchema={
                "type": "object",
                "properties": {},
            }),
            Tool(name="titan_batch", description="並列バッチ処理 (M3 Ultra最適化)", inputSchema={
                "type": "object",
                "properties": {
                    "tasks": {"type": "array", "items": {"type": "object", "properties": {"prompt": {"type": "string"}, "system": {"type": "string"}}}},
                    "concurrency": {"type": "integer", "default": 8},
                    "model": {"type": "string"}
                },
                "required": ["tasks"]
            }),
            Tool(name="pcc_critic_run", description="PCC制約注入×マルチAI Criticパイプライン", inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "批評対象のプロンプト"},
                    "preset": {"type": "string", "enum": ["探", "極", "均", "監", "刃"], "default": "探"},
                    "runtime": {"type": "string", "enum": ["gemini", "claude"], "default": "gemini"},
                    "model": {"type": "string", "default": "deep"},
                    "timeout": {"type": "integer", "default": 120}
                },
                "required": ["prompt"]
            }),
            Tool(name="pcc_critic_audit", description="テキストの品質監査（迎合検知・evidence判定）", inputSchema={
                "type": "object",
                "properties": {"text": {"type": "string", "description": "監査対象のテキスト"}},
                "required": ["text"]
            }),
        ]
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> List['TextContent']:
        try:
            handler = getattr(tools, name, None)
            if handler:
                result = await handler(**arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}
            return [TextContent(type="text", text=json.dumps(result, ensure_ascii=False, indent=2))]
        except Exception as e:
            logger.error(f"Tool error: {name} -> {e}")
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]
    
    @server.list_resources()
    async def list_resources() -> List['Resource']:
        return [
            Resource(
                uri="antigravity://status",
                name="System Status",
                mimeType="application/json",
                description="Current Antigravity/Titan system status"
            ),
            Resource(
                uri="antigravity://queue/pending",
                name="Pending Jobs",
                mimeType="application/json",
                description="Titan API Pending Jobs"
            ),
            Resource(
                uri="antigravity://queue/running",
                name="Running Jobs",
                mimeType="application/json",
                description="Titan API Running Jobs"
            ),
            Resource(
                uri="antigravity://logs/latest",
                name="Server Logs",
                mimeType="text/plain",
                description="Latest MCP Server Logs"
            )
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> 'ReadResourceResult':
        import json
        
        if uri == "antigravity://status":
            stats = await tools.get_stats()
            return ReadResourceResult(contents=[
                TextContent(
                    uri=uri,
                    mimeType="application/json",
                    text=json.dumps(stats, ensure_ascii=False, indent=2)
                )
            ])
            
        elif uri == "antigravity://queue/pending":
            # Direct file read for speed, assuming SSD path
            queue_path = Path("/Volumes/SSD/ryota_runtime/queue/pending.json")
            if queue_path.exists():
                content = queue_path.read_text()
            else:
                content = "[]"
            return ReadResourceResult(contents=[
                TextContent(uri=uri, mimeType="application/json", text=content)
            ])
            
        elif uri == "antigravity://queue/running":
            queue_path = Path("/Volumes/SSD/ryota_runtime/queue/running.json")
            if queue_path.exists():
                content = queue_path.read_text()
            else:
                content = "[]"
            return ReadResourceResult(contents=[
                TextContent(uri=uri, mimeType="application/json", text=content)
            ])
            
        elif uri == "antigravity://logs/latest":
            # Read local server.log if exists
            log_path = Path("server.log")
            if log_path.exists():
                # Read last 5KB
                size = log_path.stat().st_size
                with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                    if size > 5000:
                        f.seek(size - 5000)
                    content = f.read()
            else:
                content = "(No log file found)"
                
            return ReadResourceResult(contents=[
                TextContent(uri=uri, mimeType="text/plain", text=content)
            ])
            
        else:
            raise ValueError(f"Unknown resource: {uri}")

    return server


async def run_mcp_server(server) -> None:
    """MCPサーバーを実行"""
    from mcp.server.models import InitializationOptions
    from mcp.types import ServerCapabilities
    
    init_options = InitializationOptions(
        server_name="antigravity",
        server_version=VERSION,
        capabilities=ServerCapabilities(tools={})
    )
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, init_options)


def main() -> int:
    """メインエントリポイント"""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.version:
        print(f"Antigravity MCP Server v{VERSION}")
        return 0
    
    config = load_config()
    tools = AntigravityTools(config)
    
    if args.validate:
        errors = validate_config(config)
        if errors:
            for e in errors:
                print(f"❌ {e}")
            return 1
        print("✅ Configuration valid")
        return 0
    
    if args.test:
        asyncio.run(run_tests(tools))
        return 0
    
    # HTTPサーバーモード
    if args.http:
        if not AIOHTTP_AVAILABLE:
            print("Error: aiohttp not installed. Run: pip install aiohttp")
            return 1
        logger.info(f"Starting HTTP API Server on port {args.port}...")
        asyncio.run(run_http_server(tools, port=args.port))
        return 0
    
    # MCPサーバーモード
    if MCP_AVAILABLE:
        server = create_mcp_server(tools)
        logger.info("Starting Antigravity MCP Server...")
        asyncio.run(run_mcp_server(server))
    else:
        print("Error: MCP SDK not installed. Run: pip install mcp")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
