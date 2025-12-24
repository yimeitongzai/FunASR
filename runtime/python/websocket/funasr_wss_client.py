# -*- encoding: utf-8 -*-
import os
import time
import websockets, ssl
import asyncio

# import threading
import argparse
import json
import traceback
from multiprocessing import Process

# from funasr.fileio.datadir_writer import DatadirWriter

import logging

logging.basicConfig(level=logging.ERROR)

# ==============================================================================
# 客户端参数解析：控制连接地址、输入源和识别模式
# ==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="localhost", required=False, help="远程服务器的 IP 地址，如果是本地则填 localhost"
)
parser.add_argument("--port", type=int, default=10095, required=False, help="远程服务器 WebSocket 监听端口")
parser.add_argument("--chunk_size", type=str, default="5, 10, 5", help="流式识别的分片 Latency 配置")
parser.add_argument("--encoder_chunk_look_back", type=int, default=4, help="流式模型编码器回看的分片数")
parser.add_argument("--decoder_chunk_look_back", type=int, default=0, help="流式模型解码器回看的分片数")
parser.add_argument("--chunk_interval", type=int, default=10, help="音频切片的时间间隔频率")
parser.add_argument(
    "--hotword",
    type=str,
    default="",
    help="热词文件路径：每行一个词及其权重，例如：阿里巴巴 20",
)
parser.add_argument("--audio_in", type=str, default=None, help="音频输入源：None 表示开启麦克风实时录音，填文件路径表示处理已有音频文件")
parser.add_argument("--audio_fs", type=int, default=16000, help="音频采样率 (默认 16000Hz)")
parser.add_argument(
    "--send_without_sleep",
    action="store_true",
    default=True,
    help="如果是处理文件，是否不带间隔地快速发送",
)
parser.add_argument("--thread_num", type=int, default=1, help="多线程测试时的并发线程数")
parser.add_argument("--words_max_print", type=int, default=10000, help="屏幕打印的最大文字数量")
parser.add_argument("--output_dir", type=str, default=None, help="识别结果的保存目录")
parser.add_argument("--ssl", type=int, default=1, help="是否启用 SSL 加密连接 (1=启用, 0=禁用)")
parser.add_argument("--use_itn", type=int, default=1, help="是否启用数字转换 (ITN，如 将'一千'转为'1000')")
parser.add_argument("--mode", type=str, default="2pass", help="识别模式：offline(离线/整句识别), online(实时流式识别), 2pass(双路实时识别)")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
print(args)

from queue import Queue

voices = Queue()
offline_msg_done = False # 离线模式下的结束标志

if args.output_dir is not None:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


async def record_microphone():
    """
    麦克风实时采集并发送模块。
    通过调用 pyaudio 硬件接口，不断读取 PCM 音频流并通过网络发送。
    """
    is_finished = False
    import pyaudio

    global voices
    FORMAT = pyaudio.paInt16 # 采样深度：16位
    CHANNELS = 1             # 单声道
    RATE = 16000             # 采样率：16k
    # 计算每次从声卡驱动读取的音频块大小（CHUNK）
    chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    CHUNK = int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()

    # A. 开启输入流
    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )
    
    # B. 解析热词
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            f_scp = open(args.hotword, encoding="utf-8")
            hot_lines = f_scp.readlines()
            for line in hot_lines:
                words = line.strip().split(" ")
                if len(words) < 2:
                    continue
                try:
                    fst_dict[" ".join(words[:-1])] = int(words[-1])
                except ValueError:
                    continue
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = args.hotword

    use_itn = True if args.use_itn != 0 else False

    # C. 首次连接握手：先发送识别配置 JSON
    message = json.dumps(
        {
            "mode": args.mode,
            "chunk_size": args.chunk_size,
            "chunk_interval": args.chunk_interval,
            "encoder_chunk_look_back": args.encoder_chunk_look_back,
            "decoder_chunk_look_back": args.decoder_chunk_look_back,
            "wav_name": "microphone",
            "is_speaking": True, # 标记开始说话
            "hotwords": hotword_msg,
            "itn": use_itn,
        }
    )
    await websocket.send(message)

    # D. 进入死循环：不断读取硬件采集的声音，像流水一样发送给服务器
    while True:
        data = stream.read(CHUNK) # 从硬件读取一段声音
        await websocket.send(data) # 发送给服务器
        await asyncio.sleep(0.005) # 短暂休眠，给 CPU 释放执行权


async def record_from_scp(chunk_begin, chunk_size):
    """
    文件读取与发送模块：从音频文件（wav/pcm）读取并模拟实时流发送给服务器。
    """
    global voices
    is_finished = False
    if args.audio_in.endswith(".scp"):
        f_scp = open(args.audio_in)
        wavs = f_scp.readlines()
    else:
        wavs = [args.audio_in]

    # ... (热词处理与 record_microphone 类似)
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            f_scp = open(args.hotword)
            hot_lines = f_scp.readlines()
            for line in hot_lines:
                words = line.strip().split(" ")
                if len(words) < 2: continue
                try: fst_dict[" ".join(words[:-1])] = int(words[-1])
                except ValueError: continue
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = args.hotword

    sample_rate = args.audio_fs
    wav_format = "pcm"
    use_itn = True if args.use_itn != 0 else False

    if chunk_size > 0:
        wavs = wavs[chunk_begin : chunk_begin + chunk_size]

    for wav in wavs:
        wav_splits = wav.strip().split()
        wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
        wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
        if not len(wav_path.strip()) > 0: continue
        
        # 1. 读取音频文件数据
        if wav_path.endswith(".pcm"):
            with open(wav_path, "rb") as f: audio_bytes = f.read()
        elif wav_path.endswith(".wav"):
            import wave
            with wave.open(wav_path, "rb") as wav_file:
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                audio_bytes = bytes(frames)
        else:
            wav_format = "others"
            with open(wav_path, "rb") as f: audio_bytes = f.read()

        # 2. 计算分块大小，模拟实时发送频率
        stride = int(60 * args.chunk_size[1] / args.chunk_interval / 1000 * sample_rate * 2)
        chunk_num = (len(audio_bytes) - 1) // stride + 1

        # 3. 发送握手头信息
        message = json.dumps({
            "mode": args.mode, "chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval,
            "encoder_chunk_look_back": args.encoder_chunk_look_back, "decoder_chunk_look_back": args.decoder_chunk_look_back,
            "audio_fs": sample_rate, "wav_name": wav_name, "wav_format": wav_format,
            "is_speaking": True, "hotwords": hotword_msg, "itn": use_itn,
        })
        await websocket.send(message)

        # 4. 循环发送音频切片
        for i in range(chunk_num):
            beg = i * stride
            data = audio_bytes[beg : beg + stride]
            await websocket.send(data)
            
            # 最后一个分块发送结束标志
            if i == chunk_num - 1:
                await websocket.send(json.dumps({"is_speaking": False}))

            # 发送间隔控制：模拟真实语速
            sleep_duration = (0.001 if args.mode == "offline" else 60 * args.chunk_size[1] / args.chunk_interval / 1000)
            await asyncio.sleep(sleep_duration)

    if not args.mode == "offline": await asyncio.sleep(2)
    if args.mode == "offline":
        global offline_msg_done
        while not offline_msg_done: await asyncio.sleep(1)

    await websocket.close()


async def message(id):
    """
    结果接收与显示模块：负责监听从服务器传回的 JSON 文字结果并渲染到控制台。
    """
    global websocket, voices, offline_msg_done
    text_print = "" # 记录本地已识别的所有文字，用于拼接打印
    text_print_2pass_online = ""
    text_print_2pass_offline = ""
    
    if args.output_dir is not None:
        ibest_writer = open(os.path.join(args.output_dir, "text.{}".format(id)), "a", encoding="utf-8")
    else:
        ibest_writer = None
        
    try:
        while True:
            # 1. 接收服务器消息
            meg = await websocket.recv()
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]
            timestamp = ""
            offline_msg_done = meg.get("is_final", False)
            if "timestamp" in meg: timestamp = meg["timestamp"]

            # 2. 如果开启了文件输出，则写入文件
            if ibest_writer is not None:
                text_write_line = "{}\t{}\t{}\n".format(wav_name, text, timestamp) if timestamp != "" else "{}\t{}\n".format(wav_name, text)
                ibest_writer.write(text_write_line)

            if "mode" not in meg: continue
            
            # 3. 屏幕刷新与实时显示逻辑
            if meg["mode"] == "online":
                # 实时模式：文字随说随跳
                text_print += "{}".format(text)
                text_print = text_print[-args.words_max_print :]
                os.system("clear")
                print("\r识别结果: " + text_print)
            elif meg["mode"] == "offline":
                # 离线整句模式：说完一整句后显示结果
                text_print += "{}".format(text)
                print("\r识别结果: " + wav_name + ": " + text_print)
                offline_msg_done = True
            else:
                # 2pass 模式：结合了 online 的快和 offline 的准
                if meg["mode"] == "2pass-online":
                    text_print_2pass_online += "{}".format(text)
                    text_print = text_print_2pass_offline + text_print_2pass_online
                else:
                    text_print_2pass_online = ""
                    text_print = text_print_2pass_offline + "{}".format(text)
                    text_print_2pass_offline += "{}".format(text)
                text_print = text_print[-args.words_max_print :]
                os.system("clear")
                print("\r识别结果: " + text_print)

    except Exception as e:
        print("连接已断开:", e)


async def ws_client(id, chunk_begin, chunk_size):
    """
    客户端管理协程：负责连接服务器、分配任务。
    """
    if args.audio_in is None:
        chunk_begin = 0
        chunk_size = 1
    global websocket, voices, offline_msg_done

    for i in range(chunk_begin, chunk_begin + chunk_size):
        offline_msg_done = False
        voices = Queue()
        # 处理 SSL 安全连接
        if args.ssl == 1:
            ssl_context = ssl.SSLContext()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            uri = "wss://{}:{}".format(args.host, args.port)
        else:
            uri = "ws://{}:{}".format(args.host, args.port)
            ssl_context = None
        
        print("connect to", uri)
        async with websockets.connect(
            uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
        ) as websocket:
            # 同时开启“录制发送”和“接收显示”两个异步任务
            if args.audio_in is not None:
                task = asyncio.create_task(record_from_scp(i, 1))
            else:
                task = asyncio.create_task(record_microphone())
            task3 = asyncio.create_task(message(str(id) + "_" + str(i)))
            await asyncio.gather(task, task3)
    exit(0)


def one_thread(id, chunk_begin, chunk_size):
    asyncio.get_event_loop().run_until_complete(ws_client(id, chunk_begin, chunk_size))
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    if args.audio_in is None:
        # 针对麦克风输入的实时处理模式
        p = Process(target=one_thread, args=(0, 0, 0))
        p.start()
        p.join()
        print("end")
    else:
        # 针对文件的批量识别模式（多进程并发）
        if args.audio_in.endswith(".scp"):
            f_scp = open(args.audio_in)
            wavs = f_scp.readlines()
        else:
            wavs = [args.audio_in]
        
        total_len = len(wavs)
        chunk_size = int(total_len / args.thread_num) if total_len >= args.thread_num else 1
        remain_wavs = total_len - chunk_size * args.thread_num if total_len >= args.thread_num else 0

        process_list = []
        chunk_begin = 0
        for i in range(args.thread_num):
            now_chunk_size = chunk_size + (1 if remain_wavs > 0 else 0)
            if remain_wavs > 0: remain_wavs -= 1
            p = Process(target=one_thread, args=(i, chunk_begin, now_chunk_size))
            chunk_begin += now_chunk_size
            p.start()
            process_list.append(p)

        for p in process_list: p.join()
        print("end")
