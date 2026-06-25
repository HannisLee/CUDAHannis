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
# "normal"  = 常规聊天负载
# "prefill" = 长输入 Prefill 测试
# "decode"  = 长输出 Decode 测试
RUN_CASES = ["normal", "prefill", "decode"]

# 结果目录：固定保存到脚本同目录下的 bench_results
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_DIR = SCRIPT_DIR / "bench_results"

# 每次运行只输出一个 result.txt，重复运行会覆盖旧的 result.txt
RUN_LOG_FILE = RESULT_DIR / "result.txt"

# JSON 文件名保留时间后缀，避免多次运行互相覆盖
TIME_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULT_PREFIX = f"qwen36_35b_single_{TIME_TAG}"


# =========================
# 测试配置
# =========================

ALL_TESTS = {
    "normal": {
        "name": "常规聊天负载",
        "input_len": 512,
        "output_len": 128,
        "num_prompts": 200,
        "request_rate": 4,
        "json_name": f"{RESULT_PREFIX}_normal_512in_128out_qps4.json",
    },
    "prefill": {
        "name": "长输入 Prefill 测试",
        "input_len": 8192,
        "output_len": 128,
        "num_prompts": 100,
        "request_rate": 2,
        "json_name": f"{RESULT_PREFIX}_prefill_8192in_128out_qps2.json",
    },
    "decode": {
        "name": "长输出 Decode 测试",
        "input_len": 512,
        "output_len": 1024,
        "num_prompts": 100,
        "request_rate": 2,
        "json_name": f"{RESULT_PREFIX}_decode_512in_1024out_qps2.json",
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


def get_selected_tests() -> list[dict]:
    """根据 RUN_CASES 选择本次要运行的测试项。"""
    selected_tests = []

    for case in RUN_CASES:
        if case not in ALL_TESTS:
            raise ValueError(f"未知测试项: {case}，可选项为: {list(ALL_TESTS.keys())}")
        selected_tests.append(ALL_TESTS[case])

    if not selected_tests:
        raise ValueError("RUN_CASES 为空，本次没有任何测试会被运行。")

    return selected_tests


def build_cmd(served_model_name: str, test: dict) -> list[str]:
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
        "--random-input-len", str(test["input_len"]),
        "--random-output-len", str(test["output_len"]),
        "--num-prompts", str(test["num_prompts"]),
        "--request-rate", str(test["request_rate"]),
        "--result-dir", str(RESULT_DIR),
        "--result-filename", test["json_name"],
        "--save-result",
        "--trust-remote-code",
    ]


def write_log(text: str = "") -> None:
    """把测试记录写入本次运行的 result.txt。"""
    with open(RUN_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(text)
        f.write("\n")


def reset_log() -> None:
    """每次运行开始时清空旧的 result.txt。"""
    with open(RUN_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("")


def run_cmd(cmd: list[str]) -> tuple[int, float]:
    """运行命令；不捕获 stdout/stderr，这样 tqdm 进度条可以正常显示。"""
    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start
    return result.returncode, elapsed


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    reset_log()

    selected_tests = get_selected_tests()
    served_model_name = get_served_model_name()

    write_log("=" * 100)
    write_log("vLLM Benchmark 测试记录")
    write_log("=" * 100)
    write_log(f"测试时间: {TIME_TAG}")
    write_log(f"BASE_URL: {BASE_URL}")
    write_log(f"ENDPOINT: {ENDPOINT}")
    write_log(f"MODEL_PATH: {MODEL_PATH}")
    write_log(f"SERVED_MODEL_NAME: {served_model_name}")
    write_log(f"RUN_CASES: {RUN_CASES}")
    write_log(f"RESULT_DIR: {RESULT_DIR}")
    write_log("")

    print("开始运行 vLLM benchmark 测试。")
    print(f"本次测试项: {RUN_CASES}")
    print(f"JSON 目录: {RESULT_DIR}")
    print(f"日志文件: {RUN_LOG_FILE}")

    summary = []

    for i, test in enumerate(selected_tests, start=1):
        total = len(selected_tests)

        print("\n" + "=" * 100)
        print(f"开始测试 {i}/{total}：{test['name']}")
        print(
            f"输入长度={test['input_len']}，"
            f"输出长度={test['output_len']}，"
            f"请求数={test['num_prompts']}，"
            f"请求速率={test['request_rate']}"
        )
        print("=" * 100)

        cmd = build_cmd(served_model_name, test)

        write_log("=" * 100)
        write_log(f"测试 {i}: {test['name']}")
        write_log("=" * 100)
        write_log("命令:")
        write_log(" ".join(cmd))
        write_log("")

        returncode, elapsed = run_cmd(cmd)

        json_path = RESULT_DIR / test["json_name"]
        json_exists = json_path.exists()

        write_log(f"耗时: {elapsed:.2f} 秒")
        write_log(f"返回码: {returncode}")
        write_log(f"JSON 文件: {json_path}")
        write_log(f"JSON 是否存在: {json_exists}")
        write_log("")

        summary.append({
            "name": test["name"],
            "returncode": returncode,
            "elapsed": elapsed,
            "json_path": json_path,
            "json_exists": json_exists,
        })

    print("\n" + "=" * 100)
    print("测试统一汇总")
    print("=" * 100)

    write_log("=" * 100)
    write_log("测试统一汇总")
    write_log("=" * 100)

    for i, item in enumerate(summary, start=1):
        status = "成功" if item["returncode"] == 0 else "失败"
        save_status = "已保存" if item["json_exists"] else "未找到"

        print(f"\n测试 {i}: {item['name']}")
        print(f"运行状态: {status}")
        print(f"耗时: {item['elapsed']:.2f} 秒")
        print(f"JSON 状态: {save_status}")
        print(f"JSON 路径: {item['json_path']}")

        write_log(f"测试 {i}: {item['name']}")
        write_log(f"运行状态: {status}")
        write_log(f"耗时: {item['elapsed']:.2f} 秒")
        write_log(f"JSON 状态: {save_status}")
        write_log(f"JSON 路径: {item['json_path']}")
        write_log("")

    print("\n全部测试完成。")
    print(f"日志文件: {RUN_LOG_FILE}")


if __name__ == "__main__":
    main()