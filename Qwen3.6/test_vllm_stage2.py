#!/usr/bin/env python3
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from datetime import datetime


# =========================
# 固定配置区：自用只改这里
# =========================

BASE_URL = "http://localhost:8081"
ENDPOINT = "/v1/completions"

# 本地模型路径：给 vllm bench 加载 tokenizer 用
MODEL_PATH = "/mnt/workspace/users/han.li/models/Qwen--Qwen3.6-35B-A3B"

# vLLM server 对外暴露的模型名
# 不确定就保持 None，脚本会自动从 /v1/models 获取
SERVED_MODEL_NAME = None

# 控制本次运行哪些测试
# 可选项：
# "normal"  = 常规聊天负载：512 -> 128
# "prefill" = 长输入 Prefill 测试：8192 -> 128
# "decode"  = 长输出 Decode 测试：512 -> 1024
RUN_CASES = ["normal", "prefill", "decode"]

# 每类 workload 对应的 request_rate 扫描范围
REQUEST_RATE_SWEEPS = {
    "normal": [1, 2, 4, 6, 8],
    "prefill": [0.5, 1.0, 1.5, 2.0],
    "decode": [0.25, 0.5, 0.75, 1.0, 1.25],
}

# 结果目录：固定保存到脚本同目录下的 bench_results
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "bench_results"

# 每次运行只输出一个 result.txt，重复运行会覆盖旧文件
RESULT_TXT = RESULT_DIR / "result.txt"

TIME_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


# =========================
# workload 配置
# =========================

ALL_TESTS = {
    "normal": {
        "name": "常规聊天负载",
        "input_len": 512,
        "output_len": 128,
        "num_prompts": 300,
    },
    "prefill": {
        "name": "长输入 Prefill 测试",
        "input_len": 8192,
        "output_len": 128,
        "num_prompts": 100,
    },
    "decode": {
        "name": "长输出 Decode 测试",
        "input_len": 512,
        "output_len": 1024,
        "num_prompts": 100,
    },
}


def get_served_model_name() -> str:
    """从 /v1/models 自动获取当前 vLLM 服务暴露的模型名。"""
    if SERVED_MODEL_NAME:
        return SERVED_MODEL_NAME

    url = f"{BASE_URL.rstrip('/')}/v1/models"
    with urllib.request.urlopen(url, timeout=10) as response:
        data = json.loads(response.read().decode("utf-8"))

    models = data.get("data", [])
    if not models:
        raise RuntimeError("没有从 /v1/models 获取到模型名，请确认 vLLM 服务已经启动完成。")

    return models[0]["id"]


def get_jobs() -> list[dict]:
    """根据 RUN_CASES 和 REQUEST_RATE_SWEEPS 生成测试任务。"""
    jobs = []

    for case in RUN_CASES:
        if case not in ALL_TESTS:
            raise ValueError(f"未知测试项: {case}，可选项为: {list(ALL_TESTS.keys())}")

        if case not in REQUEST_RATE_SWEEPS:
            raise ValueError(f"没有为 {case} 配置 request_rate sweep。")

        test = ALL_TESTS[case]

        for rate in REQUEST_RATE_SWEEPS[case]:
            jobs.append({
                "case": case,
                "case_name": test["name"],
                "input_len": test["input_len"],
                "output_len": test["output_len"],
                "num_prompts": test["num_prompts"],
                "request_rate": rate,
            })

    if not jobs:
        raise ValueError("没有生成任何测试任务，请检查 RUN_CASES。")

    return jobs


def build_cmd(served_model_name: str, job: dict) -> list[str]:
    """构建单组 vllm bench serve 命令。"""
    return [
        "vllm", "bench", "serve",
        "--backend", "vllm",

        # 注意：这里是本地模型路径，不是 served model name
        "--model", MODEL_PATH,

        # 这里是服务端对外暴露的模型名
        "--served-model-name", served_model_name,

        "--base-url", BASE_URL,
        "--endpoint", ENDPOINT,
        "--dataset-name", "random",
        "--random-input-len", str(job["input_len"]),
        "--random-output-len", str(job["output_len"]),
        "--num-prompts", str(job["num_prompts"]),
        "--request-rate", str(job["request_rate"]),
        "--trust-remote-code",
    ]


def write_log(text: str = "") -> None:
    """写入 result.txt。"""
    with open(RESULT_TXT, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def reset_log() -> None:
    """每次运行开始时清空旧的 result.txt。"""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write("")


def run_cmd_and_log(cmd: list[str]) -> tuple[int, float]:
    """运行命令；终端同步显示，同时把完整输出写入 result.txt。"""
    start = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None

    with open(RESULT_TXT, "a", encoding="utf-8") as f:
        for line in process.stdout:
            print(line, end="")
            f.write(line)

    returncode = process.wait()
    elapsed = time.time() - start

    return returncode, elapsed


def main() -> None:
    reset_log()

    served_model_name = get_served_model_name()
    jobs = get_jobs()

    header = [
        "=" * 100,
        "vLLM Benchmark 阶段 2：request_rate sweep 测试记录",
        "=" * 100,
        f"测试时间: {TIME_TAG}",
        f"BASE_URL: {BASE_URL}",
        f"ENDPOINT: {ENDPOINT}",
        f"MODEL_PATH: {MODEL_PATH}",
        f"SERVED_MODEL_NAME: {served_model_name}",
        f"RUN_CASES: {RUN_CASES}",
        f"REQUEST_RATE_SWEEPS: {REQUEST_RATE_SWEEPS}",
        f"RESULT_TXT: {RESULT_TXT}",
        "",
    ]

    for line in header:
        print(line)
        write_log(line)

    summary = []

    for i, job in enumerate(jobs, start=1):
        total = len(jobs)

        title = [
            "",
            "=" * 100,
            f"开始测试 {i}/{total}: {job['case_name']}",
            f"case={job['case']}, input_len={job['input_len']}, "
            f"output_len={job['output_len']}, num_prompts={job['num_prompts']}, "
            f"request_rate={job['request_rate']}",
            "=" * 100,
        ]

        for line in title:
            print(line)
            write_log(line)

        cmd = build_cmd(served_model_name, job)

        write_log("命令:")
        write_log(" ".join(cmd))
        write_log("")

        returncode, elapsed = run_cmd_and_log(cmd)

        status = "成功" if returncode == 0 else "失败"

        summary.append({
            "case": job["case"],
            "case_name": job["case_name"],
            "input_len": job["input_len"],
            "output_len": job["output_len"],
            "num_prompts": job["num_prompts"],
            "request_rate": job["request_rate"],
            "returncode": returncode,
            "status": status,
            "elapsed": elapsed,
        })

        write_log("")
        write_log(f"本组测试状态: {status}")
        write_log(f"本组测试耗时: {elapsed:.2f} 秒")
        write_log("")

    print("\n" + "=" * 100)
    print("测试统一汇总")
    print("=" * 100)

    write_log("")
    write_log("=" * 100)
    write_log("测试统一汇总")
    write_log("=" * 100)

    for i, item in enumerate(summary, start=1):
        lines = [
            f"测试 {i}: {item['case_name']}",
            f"case: {item['case']}",
            f"输入长度: {item['input_len']}",
            f"输出长度: {item['output_len']}",
            f"请求数: {item['num_prompts']}",
            f"请求速率: {item['request_rate']}",
            f"运行状态: {item['status']}",
            f"耗时: {item['elapsed']:.2f} 秒",
            "",
        ]

        for line in lines:
            print(line)
            write_log(line)

    print("=" * 100)
    print("全部测试完成。")
    print(f"完整结果文件: {RESULT_TXT}")
    print("=" * 100)

    write_log("=" * 100)
    write_log("全部测试完成。")
    write_log(f"完整结果文件: {RESULT_TXT}")
    write_log("=" * 100)


if __name__ == "__main__":
    main()