"""用 Qwen-VL（阿里云百炼 DashScope OpenAI 兼容接口）估计物体级 metalness，输出 scene_meta.json。

只做离线语义判断：VLM 只输出物体类别 / 是否金属 / metal_prob / confidence，
不输出发射率、温度或任何物理量。生成的 scene_meta.json 交给
script/prepare_object_metalness.py --scene_meta 使用。

默认读取 <scene>/visible/ 下的可见光照片（可用 --vis_dir 指定）；
该目录不存在或为空时，按 --missing 处理：nonmetal 写出 metallic=0 的
scene_meta.json（等价于不启用金属先验），error 直接报错。

用法：
    set DASHSCOPE_API_KEY=sk-xxx          # Windows；Linux 用 export
    python script/query_vlm_metalness.py -s dataset --model qwen3-vl-flash
    python script/query_vlm_metalness.py -s dataset --dry_run

只依赖 Python 标准库，不导入 torch。
"""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request

# ============ 在这里直接填写（命令行参数 / 环境变量会覆盖这里）============
API_KEY = "sk-98cf14e2048346769aba43a2f0347e75"                  # 百炼 API key；留空则读环境变量 DASHSCOPE_API_KEY
MODEL = "qwen3-vl-flash"      # 备选：qwen3-vl-plus、qwen-vl-plus
# ==========================================================================

DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = MODEL
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
VIS_DIR_CANDIDATES = ("visible", "rgb", "color", "vis", "visible_rgb")

PROMPT = """你是一名材质分析助手。请观察图片中的主体物体，判断它主要表面的材质是金属还是非金属。
只依据图像中可见的外观线索（颜色、光泽、高光形状、反射纹理、氧化或涂层痕迹）。
不要输出发射率、温度、反射率或任何物理数值，也不要给训练建议。
只返回一个 JSON 对象，不要输出 markdown 代码块或其它文字：
{"object": "物体的简短名称（中文）",
 "is_metal": true 或 false,
 "metal_prob": 0.0 到 1.0 之间的小数，表示主体金属感的强度,
 "confidence": 0.0 到 1.0 之间的小数，表示你对该判断的把握,
 "reason": "不超过 30 字的主要依据"}
判断参考：抛光不锈钢/铝/黄铜等裸露金属取 0.8~0.95；明显氧化的金属 0.6~0.8；
镜面镀铬 0.8~0.95；金属漆或强反光塑料 0.2~0.5；木材、塑料、橡胶、陶瓷、织物取 0~0.2。
如果主体被大面积遮挡或看不清材质，降低 confidence。"""


def find_visible_dir(scene, vis_dir):
    if vis_dir:
        return vis_dir
    for name in VIS_DIR_CANDIDATES:
        candidate = os.path.join(scene, name)
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(scene, VIS_DIR_CANDIDATES[0])


def list_images(directory):
    if not os.path.isdir(directory):
        return []
    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in IMAGE_EXTS]
    return [os.path.join(directory, f) for f in sorted(files)]


def pick_evenly(paths, k):
    if len(paths) <= k:
        return paths
    if k <= 1:
        return [paths[len(paths) // 2]]
    step = (len(paths) - 1) / (k - 1)
    return [paths[round(i * step)] for i in range(k)]


def encode_image(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return mime, data


def parse_json_block(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("response contains no JSON object")
    obj = json.loads(text[start:end + 1])
    metal_prob = min(max(float(obj.get("metal_prob", 0.0)), 0.0), 1.0)
    confidence = min(max(float(obj.get("confidence", 0.0)), 0.0), 1.0)
    is_metal = bool(obj.get("is_metal", metal_prob >= 0.5))
    return {
        "object": str(obj.get("object", "")).strip(),
        "is_metal": is_metal,
        "metal_prob": metal_prob,
        "confidence": confidence,
        "reason": str(obj.get("reason", "")).strip(),
    }


def post_json(url, api_key, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key,
    })
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_vlm(path, args, api_key):
    mime, b64 = encode_image(path)
    payload = {
        "model": args.model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": PROMPT},
            ],
        }],
        "temperature": 0.1,
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
    }
    if args.dry_run:
        return None, {"prompt_chars": len(PROMPT), "image_bytes": len(b64) * 3 // 4}

    last_err = None
    for attempt in range(args.retries + 1):
        try:
            resp = post_json(args.base_url, api_key, payload, args.timeout)
            content = resp["choices"][0]["message"]["content"]
            result = parse_json_block(content)
            result["file"] = os.path.basename(path)
            return result, resp.get("usage", {})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code == 400 and "enable_thinking" in detail:
                payload.pop("enable_thinking", None)
                last_err = RuntimeError(detail)
                continue
            last_err = RuntimeError(f"HTTP {e.code}: {detail}")
        except Exception as e:  # 网络/解析错误：重试，最后一个失败向上抛
            last_err = e
        if attempt < args.retries:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{os.path.basename(path)}: {last_err}")


def aggregate(results):
    weight = sum(r["confidence"] for r in results)
    if weight <= 0:
        metal = sum(r["metal_prob"] for r in results) / len(results)
        conf = 0.0
    else:
        metal = sum(r["metal_prob"] * r["confidence"] for r in results) / weight
        conf = weight / len(results)
    names = [r["object"] for r in results if r["object"]]
    return {
        "object": max(set(names), key=names.count) if names else "",
        "metallic": round(metal, 4),
        "confidence": round(conf, 4),
    }


def write_meta(path, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser(
        description="Query Qwen-VL on Alibaba Bailian for object-level metalness")
    parser.add_argument("-s", "--scene", required=True, help="场景数据目录（含 transforms_train.json）")
    parser.add_argument("--vis_dir", type=str, default="",
                        help="可见光照片目录；默认按 visible/rgb/color/vis 顺序自动查找")
    parser.add_argument("--out", type=str, default="", help="输出路径，默认 <scene>/scene_meta.json")
    parser.add_argument("--model", type=str, default=None,
                        help="百炼模型名；不填则用文件顶部 MODEL")
    parser.add_argument("--api_key", type=str, default="",
                        help="留空则用文件顶部 API_KEY，再留空读环境变量 DASHSCOPE_API_KEY")
    parser.add_argument("--base_url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--max_images", type=int, default=6, help="最多询问多少张图（均匀抽样）")
    parser.add_argument("--timeout", type=float, default=60.0, help="单次请求超时（秒）")
    parser.add_argument("--retries", type=int, default=2, help="单张图失败后的重试次数")
    parser.add_argument("--max_tokens", type=int, default=400)
    parser.add_argument("--missing", choices=["nonmetal", "error"], default="nonmetal",
                        help="无可见光照片时的处理：nonmetal=写出 metallic=0，error=报错退出")
    parser.add_argument("--dry_run", action="store_true", help="只打印将要发送的请求，不联网")
    args = parser.parse_args()

    args.model = args.model or MODEL
    scene = os.path.abspath(args.scene)
    out_path = args.out or os.path.join(scene, "scene_meta.json")
    vis_dir = find_visible_dir(scene, args.vis_dir)
    images = pick_evenly(list_images(vis_dir), args.max_images)

    if not images:
        if args.missing == "error":
            sys.exit(f"[ERROR] 未找到可见光照片目录或目录为空: {vis_dir}（--missing error）")
        meta = {
            "object": "",
            "metallic": 0.0,
            "source": "no_visible_photos",
            "confidence": 0.0,
            "note": "未找到可见光照片，默认按非金属处理（等价于不启用金属先验）",
        }
        write_meta(out_path, meta)
        print(f"[WARN] 未找到可见光照片: {vis_dir}")
        print(f"[INFO] 已写出 {out_path}（metallic=0）")
        return

    api_key = args.api_key or API_KEY or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key and not args.dry_run:
        sys.exit("[ERROR] 缺少 API key：用 --api_key 或环境变量 DASHSCOPE_API_KEY")

    print(f"[INFO] scene={scene}")
    print(f"[INFO] vis_dir={vis_dir} ({len(images)} 张)  model={args.model}")
    if args.dry_run:
        call_vlm(images[0], args, api_key)
        print("[DRY_RUN] 请求构造正常，未联网、未写文件")
        return

    results, failed, usage_total = [], [], {"prompt_tokens": 0, "completion_tokens": 0}
    for path in images:
        try:
            result, usage = call_vlm(path, args, api_key)
            results.append(result)
            for key in usage_total:
                usage_total[key] += int(usage.get(key, 0))
            print(f"  {result['file']}: metal_prob={result['metal_prob']:.2f} "
                  f"conf={result['confidence']:.2f} {result['object']}")
        except Exception as e:
            failed.append(os.path.basename(path))
            print(f"  [WARN] {e}")

    if not results:
        sys.exit("[ERROR] 所有图片都调用失败，未写 scene_meta.json（失败不等于非金属）")

    summary = aggregate(results)
    meta = {
        "object": summary["object"],
        "metallic": summary["metallic"],
        "source": f"rgb_vlm:{args.model}",
        "confidence": summary["confidence"],
        "num_images_ok": len(results),
        "num_images_failed": len(failed),
        "per_image": results,
        "usage": usage_total,
    }
    write_meta(out_path, meta)

    if meta["confidence"] < 0.5:
        print(f"[WARN] 置信度 {meta['confidence']:.2f} < 0.5，请人工复核后再使用")
    print(f"[DONE] 写出 {out_path}：metallic={meta['metallic']:.3f} "
          f"confidence={meta['confidence']:.2f} 失败 {len(failed)}/{len(images)} 张")


if __name__ == "__main__":
    main()
