"""
Discord Webhook Notifier Utility for ML Training & Evaluation Pipelines.
Provides real-time execution progress updates, model-by-model metrics ("1 train/model" logging),
and overall experiment summaries directly to a Discord channel.
"""

import os
import json
import time
import urllib.request
import urllib.parse
from typing import Dict, List, Any, Optional


def load_env_file(env_path: str = ".env"):
    """
    Parses a local .env file and sets environment variables if not already set.
    Falls back gracefully if file is missing.
    """
    if not os.path.exists(env_path):
        return
        
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'").strip('"')
                    if key and key not in os.environ:
                        os.environ[key] = val
    except Exception as e:
        print(f"[DiscordNotifier] Warning: Failed to parse .env file: {e}")


# Load .env automatically upon module import
load_env_file()


class DiscordNotifier:
    """
    Discord Webhook Manager for real-time task progress logging.
    """
    
    # Discord embed colors (decimal)
    COLOR_INFO = 0x3498DB     # Blue
    COLOR_SUCCESS = 0x2ECC71  # Green
    COLOR_WARNING = 0xF1C40F  # Yellow
    COLOR_ERROR = 0xE74C3C    # Red
    COLOR_PURPLE = 0x9B59B6   # Purple
    
    def __init__(self, webhook_url: Optional[str] = None):
        """
        Initializes DiscordNotifier.
        URL priority: passed parameter > DISCORD_WEBHOOK_URL env var.
        """
        load_env_file()
        self.webhook_url = webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "").strip()
        self.enabled = bool(self.webhook_url and self.webhook_url.startswith("http"))
        
        if not self.enabled:
            print("[DiscordNotifier] Webhook URL not provided or invalid. Discord notifications are DISABLED.")
        else:
            print(f"[DiscordNotifier] Webhook initialized. Directing notifications to Discord.")
            
    def send_embed(
        self,
        title: str,
        description: str = "",
        color: int = COLOR_INFO,
        fields: Optional[List[Dict[str, Any]]] = None,
        footer: str = "Disaster NLP ML Benchmark"
    ) -> bool:
        """
        Posts a rich Discord Embed payload via HTTP POST.
        """
        if not self.enabled:
            return False
            
        embed = {
            "title": title,
            "description": description,
            "color": color,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "footer": {"text": footer}
        }
        
        if fields:
            embed["fields"] = fields
            
        payload = {
            "username": "ML Experiment Bot",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/2103/2103633.png",
            "embeds": [embed]
        }
        
        try:
            req = urllib.request.Request(
                self.webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "User-Agent": "Disaster-ML-Bot/1.0"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status in (200, 204)
        except Exception as e:
            print(f"[DiscordNotifier] Error sending webhook payload: {e}")
            return False

    def notify_experiment_start(
        self,
        run_id: str,
        task: str,
        model: str,
        n_trials: int,
        cv_folds: int,
        use_gpu: bool
    ):
        """Logs experiment launch banner."""
        fields = [
            {"name": "Run ID", "value": f"`{run_id}`", "inline": True},
            {"name": "Task", "value": f"`{task}`", "inline": True},
            {"name": "Model Filter", "value": f"`{model}`", "inline": True},
            {"name": "Optuna Trials", "value": str(n_trials), "inline": True},
            {"name": "CV Folds", "value": str(cv_folds), "inline": True},
            {"name": "GPU Accelerated", "value": "⚡ True" if use_gpu else "🖥️ False", "inline": True},
        ]
        self.send_embed(
            title="🚀 Experiment Pipeline Started",
            description=f"Beginning ML experiment pipeline for run **{run_id}**.",
            color=self.COLOR_PURPLE,
            fields=fields
        )

    def notify_step_complete(
        self,
        task_name: str,
        step_name: str,
        metrics: Optional[Dict[str, Any]] = None,
        elapsed_sec: Optional[float] = None
    ):
        """
        Logs progress notification whenever 1 model training / trial step completes.
        ("เมื่อทำครบ 1 train/model ให้ log 1 ครั้ง")
        """
        fields = [
            {"name": "Task Scope", "value": f"`{task_name}`", "inline": True},
            {"name": "Model / Step", "value": f"**{step_name}**", "inline": True},
        ]
        
        if elapsed_sec is not None:
            fields.append({"name": "Time Elapsed", "value": f"{elapsed_sec:.2f}s", "inline": True})
            
        if metrics:
            metric_lines = []
            for k, v in metrics.items():
                if isinstance(v, float):
                    metric_lines.append(f"• **{k}**: `{v:.4f}`")
                else:
                    metric_lines.append(f"• **{k}**: `{v}`")
            fields.append({
                "name": "📊 Results & Metrics",
                "value": "\n".join(metric_lines) if metric_lines else "N/A",
                "inline": False
            })
            
        self.send_embed(
            title=f"✅ Finished Step: {step_name}",
            description=f"Completed training and evaluation for **{step_name}** in `{task_name}`.",
            color=self.COLOR_SUCCESS,
            fields=fields
        )

    def notify_task_complete(self, task_name: str, summary_info: str = ""):
        """Logs completion of an entire task."""
        self.send_embed(
            title=f"🏆 Task Completed: {task_name}",
            description=summary_info or f"All models and benchmarks for **{task_name}** completed successfully.",
            color=self.COLOR_INFO
        )

    def notify_experiment_complete(self, run_id: str, output_dir: str):
        """Logs final experiment completion banner."""
        fields = [
            {"name": "Run ID", "value": f"`{run_id}`", "inline": True},
            {"name": "Output Directory", "value": f"`{output_dir}`", "inline": False}
        ]
        self.send_embed(
            title="🎉 ALL EXPERIMENTS COMPLETED SUCCESSFULLY",
            description="The entire experiment pipeline finished without errors.",
            color=self.COLOR_SUCCESS,
            fields=fields
        )

    def notify_error(self, task_name: str, error_msg: str):
        """Logs error alert."""
        self.send_embed(
            title=f"❌ Error in {task_name}",
            description=f"An exception occurred during execution:\n```\n{error_msg[:1500]}\n```",
            color=self.COLOR_ERROR
        )
