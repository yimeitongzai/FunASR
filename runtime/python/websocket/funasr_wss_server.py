import asyncio
import json
import websockets
import time
import logging
import tracemalloc
import numpy as np
import argparse
import ssl
import traceback
import torch


# ==============================================================================
# 参数解析部分：定义服务器启动时的各项配置
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", required=False, help="服务器监听地址，0.0.0.0 表示允许所有 IP 访问"
)
parser.add_argument("--port", type=int, default=10095, required=False, help="WebSocket 服务端口")
parser.add_argument(
    "--asr_model",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    help="ASR 模型名称或本地路径",
)
parser.add_argument("--asr_model_revision", type=str, default="v2.0.4", help="模型版本号")
parser.add_argument(
    "--asr_model_online",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    help="流式 ASR 模型名称或本地路径",
)
parser.add_argument("--asr_model_online_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--vad_model",
    type=str,
    default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    help="VAD 模型名称（语音端点检测，用于自动断句）",
)
parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--punc_model",
    type=str,
    default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    help="标点预测模型名称",
)
parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument("--ngpu", type=int, default=1, help="GPU 数量 (0 为使用 CPU)")
parser.add_argument("--device", type=str, default="cuda", help="推理设备选择：cuda (GPU) 或 cpu")
parser.add_argument("--ncpu", type=int, default=4, help="CPU 核心数")
parser.add_argument("--max_end_silence_time", type=int, default=2000, help="VAD 检测到多少毫秒静音则认为话结束了 (默认 3000ms)")
parser.add_argument(
    "--certfile",
    type=str,
    default="../../ssl_key/server.crt",
    required=False,
    help="SSL 证书文件路径 (开启加密连接时使用)",
)

parser.add_argument(
    "--keyfile",
    type=str,
    default="../../ssl_key/server.key",
    required=False,
    help="SSL 密钥文件路径",
)
args = parser.parse_args()


websocket_users = set() # 用于管理所有当前连接的客户端集合

# ==============================================================================
# 模型加载部分：初始化 ASR、VAD 和标点模型
# ==============================================================================
print("model loading")
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess # 用于 SenseVoice 的结果后处理，过滤标签

# 1. 加载 ASR 离线识别模型 (如 SenseVoice 或 Paraformer)
# 这里加载的模型负责处理整句或长段音频的最终识别
model_asr = AutoModel(
    model=args.asr_model,
    # model_revision=args.asr_model_revision,
    # ngpu=args.ngpu,
    # ncpu=args.ncpu,
    device=args.device,
    disable_update=True, # 禁用联网更新检查，防止因内网或代理问题导致启动报错
    disable_pbar=True,   # 禁用进度条，保持后台运行日志整洁
    disable_log=True,    # 禁用详细日志输出
    trust_remote_code=True,
)

# 2. 加载流式 ASR 模型 (用于 online 或 2pass 模式的第一路识别)
if args.asr_model_online != "":
    print(f"loading online model: {args.asr_model_online}")
    model_asr_streaming = AutoModel(
        model=args.asr_model_online,
        # model_revision=args.asr_model_online_revision,
        device=args.device,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
        trust_remote_code=True,
    )
else:
    model_asr_streaming = None

# 3. 加载 VAD 模型 (语音端点检测)
# VAD 非常重要，它负责实时监听音频流，判断用户什么时候开始说话，什么时候结束
model_vad = AutoModel(
    model=args.vad_model,
    model_revision=args.vad_model_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_update=True,
    disable_pbar=True,
    disable_log=True,
    trust_remote_code=True,
    max_end_silence_time=args.max_end_silence_time, # 设置静音断句时间
)

# 4. 加载标点预测模型 (可选)
if args.punc_model != "":
    model_punc = AutoModel(
        model=args.punc_model,
        model_revision=args.punc_model_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_update=True,
        disable_pbar=True,
        disable_log=True,
        trust_remote_code=True,
    )
else:
    model_punc = None


print("model loaded! only support one client at the same time now!!!!")


async def ws_reset(websocket):
    """
    当连接需要重置（如报错或结束）时，清空当前 WebSocket 连接的所有缓存状态并关闭连接。
    """
    print("ws reset now, total num is ", len(websocket_users))

    websocket.status_dict_asr_online["cache"] = {}
    websocket.status_dict_asr_online["is_final"] = True
    websocket.status_dict_vad["cache"] = {}
    websocket.status_dict_vad["is_final"] = True
    websocket.status_dict_punc["cache"] = {}

    await websocket.close()


async def clear_websocket():
    """
    清空所有活跃连接。
    """
    for websocket in websocket_users:
        await ws_reset(websocket)
    websocket_users.clear()


async def ws_serve(websocket, path=None):
    """
    WebSocket 服务入口协程，每一个新连接都会触发该函数的运行。
    """
    frames = []           # 存储最原始接收到的二进制音频帧切片
    frames_asr = []       # 存储经过 VAD 判断为有效人声的音频（待识别）
    frames_asr_online = []# 存储用于流式识别的音频缓存
    global websocket_users
    
    websocket_users.add(websocket)
    
    # status_dict 用于存储每个客户端连接对应的模型推理中间缓存（状态管理）
    websocket.status_dict_asr = {}
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {"cache": {}, "is_final": False}
    websocket.status_dict_punc = {"cache": {}}
    
    websocket.chunk_interval = 10 # 处理频率配置
    websocket.vad_pre_idx = 0     # 记录音频流的时间偏移
    speech_start = False          # 标记当前是否正处于“说话中”的状态
    speech_end_i = -1             # 标记当前句子的结束点
    websocket.wav_name = "microphone"
    websocket.mode = "2pass"
    
    print("new user connected", flush=True)

    try:
        # 持续异步监听客户端发来的消息
        async for message in websocket:
            # A. 处理字符串消息（JSON）：通常包含配置参数，如 mode, chunk_size, hotwords 等
            if isinstance(message, str):
                messagejson = json.loads(message)

                if "is_speaking" in messagejson:
                    websocket.is_speaking = messagejson["is_speaking"]
                    # 如果客户端手动停止说话，标记 online 模式的 is_final 为 True
                    websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
                if "chunk_interval" in messagejson:
                    websocket.chunk_interval = messagejson["chunk_interval"]
                if "wav_name" in messagejson:
                    websocket.wav_name = messagejson.get("wav_name")
                if "chunk_size" in messagejson:
                    chunk_size = messagejson["chunk_size"]
                    if isinstance(chunk_size, str):
                        chunk_size = chunk_size.split(",")
                    websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]
                if "encoder_chunk_look_back" in messagejson:
                    websocket.status_dict_asr_online["encoder_chunk_look_back"] = messagejson[
                        "encoder_chunk_look_back"
                    ]
                if "decoder_chunk_look_back" in messagejson:
                    websocket.status_dict_asr_online["decoder_chunk_look_back"] = messagejson[
                        "decoder_chunk_look_back"
                    ]
                if "hotwords" in messagejson:
                    websocket.status_dict_asr["hotword"] = messagejson["hotwords"]
                if "mode" in messagejson:
                    websocket.mode = messagejson["mode"]

            # B. 处理二进制消息：麦克风采集到的 PCM 原始音频切片数据
            websocket.status_dict_vad["chunk_size"] = int(
                websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval
            )
            if len(frames_asr_online) > 0 or len(frames_asr) >= 0 or not isinstance(message, str):
                if not isinstance(message, str):
                    frames.append(message)
                    duration_ms = len(message) // 32 # 计算这一段音频在 16k 采样率下的时长
                    websocket.vad_pre_idx += duration_ms

                    # 1. 执行实时流式识别 (可选，由 mode 决定)
                    frames_asr_online.append(message)
                    websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
                    if (
                        len(frames_asr_online) % websocket.chunk_interval == 0
                        or websocket.status_dict_asr_online["is_final"]
                    ):
                        if websocket.mode == "2pass" or websocket.mode == "online":
                            audio_in = b"".join(frames_asr_online)
                            try:
                                await async_asr_online(websocket, audio_in)
                            except Exception as e:
                                print(f"error in asr streaming, {websocket.status_dict_asr_online}, {e}")
                                traceback.print_exc()
                        frames_asr_online = []
                    
                    if speech_start:
                        frames_asr.append(message)
                    
                    # 2. 执行实时 VAD (端点检测)：判断人声开始和结束
                    try:
                        speech_start_i, speech_end_i = await async_vad(websocket, message)
                    except Exception as e:
                        print(f"error in vad: {e}")
                        traceback.print_exc()
                    
                    # 如果检测到“开始说话”，标记 speech_start 为 True 并提取回溯缓存
                    if speech_start_i != -1:
                        speech_start = True
                        beg_bias = (websocket.vad_pre_idx - speech_start_i) // duration_ms
                        frames_pre = frames[-beg_bias:]
                        frames_asr = []
                        frames_asr.extend(frames_pre)
                
                # C. 如果检测到“说话结束”（VAD 尾点）或者客户端手动停止说话
                if speech_end_i != -1 or not websocket.is_speaking:
                    # 在 2pass 或 offline 模式下，对整句音频进行最终识别，以确保准确率
                    if websocket.mode == "2pass" or websocket.mode == "offline":
                        audio_in = b"".join(frames_asr)
                        try:
                            await async_asr(websocket, audio_in) # 调用主 ASR 模型
                        except Exception as e:
                            print(f"error in asr offline: {e}")
                            traceback.print_exc()
                    
                    # 一句识别结束，清空当前句的各种缓存状态，准备接收下一句
                    frames_asr = []
                    speech_start = False
                    frames_asr_online = []
                    websocket.status_dict_asr_online["cache"] = {}
                    if not websocket.is_speaking:
                        websocket.vad_pre_idx = 0
                        frames = []
                        websocket.status_dict_vad["cache"] = {}
                    else:
                        # 虽然一句话结束了，但为了平滑切换，保留最后一点音频缓存
                        frames = frames[-20:]

    except websockets.ConnectionClosed:
        print("ConnectionClosed...", websocket_users, flush=True)
        await ws_reset(websocket)
        websocket_users.remove(websocket)
    except websockets.InvalidState:
        print("InvalidState...")
    except Exception as e:
        print("Exception:", e)


async def async_vad(websocket, audio_in):
    """
    调用 VAD 模型，返回当前音频片中的人声开始(speech_start)和结束(speech_end)偏移量。
    """
    if isinstance(audio_in, bytes):
        audio_in = np.frombuffer(audio_in, dtype=np.int16).astype(np.float32) / 32768.0
    
    segments_result = model_vad.generate(input=[audio_in], **websocket.status_dict_vad)[0]["value"]

    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


async def async_asr(websocket, audio_in):
    """
    对整句音频调用离线 ASR 模型进行最终的高精度识别。
    """
    if len(audio_in) > 0:
        # 1. 自动识别输入格式并转换为 float32 的 torch.Tensor (提高兼容性)
        if isinstance(audio_in, bytes):
            audio_in_np = np.frombuffer(audio_in, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_in_np = audio_in
        
        # 将 numpy 转换为 tensor，并确保在正确的设备上
        audio_in_tensor = torch.from_numpy(audio_in_np).to(args.device)
        
        # 2. 推理获得识别文字
        params = websocket.status_dict_asr.copy()
        if "cache" not in params:
            params["cache"] = {}
            
        # 针对 Fun-ASR-Nano 等模型，包装成它期望的格式
        if "Fun-ASR-Nano" in args.asr_model:
             # Fun-ASR-Nano 内部对 input=[tensor] 有更稳健的处理路径
             res = model_asr.generate(input=[audio_in_tensor], **params)
        else:
             # 其他模型尝试标准调用
             res = model_asr.generate(input=[audio_in_np], **params)

        if len(res) == 0:
            return
        
        rec_result = res[0]
        
        # 针对一些模型返回嵌套列表的情况进行兼容处理 (例如 [[{...}]])
        if isinstance(rec_result, list) and len(rec_result) > 0:
            rec_result = rec_result[0]
        
        # 兼容旧版模型可能直接返回字符串或非字典格式的情况
        if not isinstance(rec_result, dict):
            rec_result = {"text": str(rec_result)}
        
        # 3. 如果存在标点模型，则对文本进行标点预测处理
        if model_punc is not None and len(rec_result["text"]) > 0:
            rec_result = model_punc.generate(
                input=rec_result["text"], **websocket.status_dict_punc
            )[0]
        
        # 3. 针对 SenseVoice/Fun-ASR-Nano 模型进行后处理，过滤情绪、语种标签（如 <|zh|>）
        if "iic/SenseVoiceSmall" in args.asr_model or "SenseVoice" in args.asr_model or "Fun-ASR-Nano" in args.asr_model:
            rec_result["text"] = rich_transcription_postprocess(rec_result["text"])

        # 4. 将识别结果打包成 JSON，通过 WebSocket 发送回客户端
        if len(rec_result["text"]) > 0:
            mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps(
                {
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": websocket.wav_name,
                    "is_final": websocket.is_speaking,
                }
            )
            await websocket.send(message)

    else:
        # 如果音频为空，发送空文字结果
        mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
        message = json.dumps(
            {
                "mode": mode,
                "text": "",
                "wav_name": websocket.wav_name,
                "is_final": websocket.is_speaking,
            }
        )
        await websocket.send(message)    

async def async_asr_online(websocket, audio_in):
    """
    执行实时在线（流式）识别。
    """
    if model_asr_streaming is None:
        return
        
    if len(audio_in) > 0:
        # 将 bytes 转换为 float32 numpy 数组
        if isinstance(audio_in, bytes):
            audio_in_np = np.frombuffer(audio_in, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            audio_in_np = audio_in

        rec_result = model_asr_streaming.generate(
            input=[audio_in_np], **websocket.status_dict_asr_online
        )[0]
        if websocket.mode == "2pass" and websocket.status_dict_asr_online.get("is_final", False):
            return
        if len(rec_result["text"]):
            mode = "2pass-online" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps(
                {
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": websocket.wav_name,
                    "is_final": websocket.is_speaking,
                }
            )
            await websocket.send(message)


async def main():
    """
    程序主入口：启动 WebSocket 服务端。
    """
    if len(args.certfile) > 0 and os.path.exists(args.certfile):
        # 开启 SSL 加密连接
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_cert = args.certfile
        ssl_key = args.keyfile
        ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)
        
        async with websockets.serve(
            ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
        ):
            await asyncio.Future()  # 持续运行
    else:
        # 开启普通不加密连接 (ws://)
        async with websockets.serve(
            ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None
        ):
            await asyncio.Future()  # 持续运行

if __name__ == "__main__":
    import os
    try:
        # 适配新版 Python 的 asyncio 运行方式
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
