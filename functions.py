import streamlit as st
import os
import time
from pathlib import Path
import wave
import pyaudio
from pydub import AudioSegment
from audiorecorder import audiorecorder
import numpy as np
from scipy.io.wavfile import write
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain.schema import SystemMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
import constants as ct
import sounddevice as sd
import numpy as np
import wave

#def record_audio(audio_input_file_path):
#    """
#    音声入力を受け取って音声ファイルを作成
#    """

#    audio = audiorecorder(
#        start_prompt="発話開始",
#        pause_prompt="やり直す",
#        stop_prompt="発話終了",
#        start_style={"color":"white", "background-color":"black"},
#        pause_style={"color":"gray", "background-color":"white"},
#        stop_style={"color":"white", "background-color":"black"}
#    )

#    if len(audio) > 0:
#        audio.export(audio_input_file_path, format="wav")
#    else:
#        st.stop()


def record_audio(audio_input_file_path, samplerate=16000, channels=1, silence_threshold=0.02, silence_duration=5):
    """
    無音が silence_duration 秒続いたら自動停止する録音関数（Streamlit対応・実時間カウント版）
    - 最後に音があった時刻を基準に無音時間を算出するため、秒数カウントが正確になります。
    Args:
        audio_input_file_path: 出力WAVファイルパス
        samplerate: サンプリングレート
        channels: チャンネル数
        silence_threshold: 無音判定の閾値（float 0.0~1.0）
        silence_duration: 無音が続く秒数で録音停止
    """

    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(audio_input_file_path), exist_ok=True)

    st.info("🎤 録音開始。発話後、5秒間沈黙すると自動で停止します。")

    recorded_data = []
    amplitude_list = []  # コールバック→ループ間で最新の振幅を共有するためのリスト
    start_time = time.time()
    # 最後に「音あり」と判定した時刻（初期値は開始時刻）
    last_sound_time = [start_time]  # mutable container を使ってコールバックから書き換え可能にする

    # grace period: 録音開始直後の安定化時間（この間は無音判定を行わない）
    grace_period = 0.8  # 0.8秒程度が安定しやすいです。必要に応じて調整してください。

    # UI 要素
    progress_bar = st.progress(0)
    status_text = st.empty()

    chunk_duration = 0.1  # ループのsleep間隔（短めにするとUIレスポンスが良くなる）
    # コールバック（音声データの受け取りはここで行う）
    def callback(indata, frames, time_info, status):
        # indata は float32 の -1.0 .. 1.0 スケールが期待される
        recorded_data.append(indata.copy())
        amplitude = float(np.max(np.abs(indata)))
        amplitude_list.append(amplitude)
        # コールバック内では時刻だけ更新（UI操作はしない）
        # 実際の last_sound_time 更新は、後段の if amplitude >= threshold で行うか、ここでも行って良い
        if amplitude >= silence_threshold:
            last_sound_time[0] = time.time()

    # 録音ストリームを開いてメインループで UI 更新と無音秒数判定を行う
    with sd.InputStream(samplerate=samplerate, channels=channels, callback=callback):
        while True:
            time.sleep(chunk_duration)  # コールバックは別スレッドで動く
            now = time.time()

            # まだ grace_period 内なら無音時間は 0 にして経過を表示する
            if now - start_time < grace_period:
                silent_time = 0.0
            else:
                silent_time = now - last_sound_time[0]

            # UI をメインスレッド側で安全に更新
            last_amp = amplitude_list[-1] if amplitude_list else 0.0
            status_text.text(f"音量: {last_amp:.3f} / 無音 {silent_time:.2f} 秒")
            progress_bar.progress(min(int((silent_time / silence_duration) * 100), 100))

            # 無音が指定秒数続いたら終了
            if silent_time >= silence_duration:
                break

    st.success(f"✅ 録音終了（{silence_duration}秒間無音を検知）")

    # WAV保存（float32 -> int16）
    recorded_data_np = np.concatenate(recorded_data)
    # もしステレオで shape (N,2) の場合、wave.writeframes は interleaved int16 を期待するので変換する
    # recorded_data_np は float32 の範囲 [-1,1]
    with wave.open(audio_input_file_path, 'w') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes((recorded_data_np * 32767).astype(np.int16).tobytes())

    # UI の片付け
    progress_bar.empty()
    status_text.empty()

    return audio_input_file_path

def transcribe_audio(audio_input_file_path):
    """
    音声入力ファイルから文字起こしテキストを取得
    Args:
        audio_input_file_path: 音声入力ファイルのパス
    """

    with open(audio_input_file_path, 'rb') as audio_input_file:
        transcript = st.session_state.openai_obj.audio.transcriptions.create(
            model="whisper-1",
            file=audio_input_file,
            language="en"
        )
    
    # 音声入力ファイルを削除
    os.remove(audio_input_file_path)

    return transcript

def save_to_wav(llm_response_audio, audio_output_file_path):
    """
    一旦mp3形式で音声ファイル作成後、wav形式に変換
    Args:
        llm_response_audio: LLMからの回答の音声データ
        audio_output_file_path: 出力先のファイルパス
    """

    temp_audio_output_filename = f"{ct.AUDIO_OUTPUT_DIR}/temp_audio_output_{int(time.time())}.mp3"
    with open(temp_audio_output_filename, "wb") as temp_audio_output_file:
        temp_audio_output_file.write(llm_response_audio)
    
    audio_mp3 = AudioSegment.from_file(temp_audio_output_filename, format="mp3")
    audio_mp3.export(audio_output_file_path, format="wav")

    # 音声出力用に一時的に作ったmp3ファイルを削除
    os.remove(temp_audio_output_filename)

def play_wav(audio_output_file_path, speed=1.0):
    """
    音声ファイルの読み上げ
    Args:
        audio_output_file_path: 音声ファイルのパス
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
    """

    # 音声ファイルの読み込み
    audio = AudioSegment.from_wav(audio_output_file_path)
    
    # 速度を変更
    if speed != 1.0:
        # frame_rateを変更することで速度を調整
        modified_audio = audio._spawn(
            audio.raw_data, 
            overrides={"frame_rate": int(audio.frame_rate * speed)}
        )
        # 元のframe_rateに戻すことで正常再生させる（ピッチを保持したまま速度だけ変更）
        modified_audio = modified_audio.set_frame_rate(audio.frame_rate)

        modified_audio.export(audio_output_file_path, format="wav")

    # PyAudioで再生
    with wave.open(audio_output_file_path, 'rb') as play_target_file:
        p = pyaudio.PyAudio()
        stream = p.open(
            format=p.get_format_from_width(play_target_file.getsampwidth()),
            channels=play_target_file.getnchannels(),
            rate=play_target_file.getframerate(),
            output=True
        )

        data = play_target_file.readframes(1024)
        while data:
            stream.write(data)
            data = play_target_file.readframes(1024)

        stream.stop_stream()
        stream.close()
        p.terminate()
    
    # LLMからの回答の音声ファイルを削除
    os.remove(audio_output_file_path)

from langchain.chains import ConversationChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate
)
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory

#ChatOpenAIインスタンスを受け取るように修正
def create_chain(template, llm=None):
    if llm is None:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    memory = ConversationSummaryBufferMemory(llm=llm, return_messages=True)
    chain = ConversationChain(llm=llm, prompt=prompt, memory=memory)
    return chain

def create_problem_and_play_audio():
    """
    問題生成と音声ファイルの再生
    Args:
        chain: 問題文生成用のChain
        speed: 再生速度（1.0が通常速度、0.5で半分の速さ、2.0で倍速など）
        openai_obj: OpenAIのオブジェクト
    """

    # 問題文を生成するChainを実行し、問題文を取得
    problem = st.session_state.chain_create_problem.predict(input="")

    # LLMからの回答を音声データに変換
    llm_response_audio = st.session_state.openai_obj.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=problem
    )

    # 音声ファイルの作成
    audio_output_file_path = f"{ct.AUDIO_OUTPUT_DIR}/audio_output_{int(time.time())}.wav"
    save_to_wav(llm_response_audio.content, audio_output_file_path)

    # 音声ファイルの読み上げ
    play_wav(audio_output_file_path, st.session_state.speed)

    return problem, llm_response_audio

def create_evaluation():
    """
    ユーザー入力値の評価生成
    """

    llm_response_evaluation = st.session_state.chain_evaluation.predict(input="")

    return llm_response_evaluation