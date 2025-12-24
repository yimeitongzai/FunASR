from funasr import AutoModel

model_dir = "FunAudioLLM/Fun-ASR-Nano-2512"

model = AutoModel(
    model=model_dir,
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
    disable_update=True,
)
wav_path = "/home/chuanyue/.cache/modelscope/hub/models/FunAudioLLM/Fun-ASR-Nano-2512/example/zh.mp3"
res = model.generate(input=[wav_path], cache={}, batch_size_s=0)
text = res[0]["text"]
print(text)