import os
import base64
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import requests
from flask_socketio import SocketIO, emit
from deepgram import DeepgramClient, ClientOptionsFromEnv,LiveTranscriptionEvents,LiveOptions
from deepgram import (
    SpeakWebSocketEvents,
    SpeakOptions,
)
import json
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.messages import trim_messages
from langchain_groq import ChatGroq
import re
import sounddevice as sd
import numpy as np
import time
import queue

app = Flask(__name__)
CORS(app)
load_dotenv()
socketio = SocketIO(app, cors_allowed_origins="*")

gpt4o = ChatOpenAI(
    model_name="gpt-4.1-nano",
    streaming=True,
    temperature=0.2,
)

llm = ChatGroq(api_key="api-key", model="llama3-8b-8192")
SYSTEM_INSTRUCTIONS = 'You are a helpful assistant. Please answer concisely, dont give long responses, only write summarised short responses.'


# Initialize Deepgram with your API key
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_API_KEY = '268ba1624e9ddf2a5f045a0c10f85eeae72b2d30'
print(DEEPGRAM_API_KEY)
client_config = ClientOptionsFromEnv(api_key=DEEPGRAM_API_KEY)
deepgram = DeepgramClient(config=client_config)

stt_connections = {}
tts_connections = {}
tts_flush = {}
transcriptions = {}
llm_memory_sessions = {}
audio_queue = {}
abort_status = {}


counter = 0

@app.route('/', methods= ['GET'])
def hello():
    return "Flask Socket.IO Server"


@socketio.on("stop_audio")
def stop_audio(data):
    socket_id = data["socket_id"]
    print("Closing Connection")
    try:
        tts_connections[socket_id].clear()
        abort_status[socket_id] = True
        audio_queue[socket_id] = queue.Queue()
    except Exception as e:
        print(e)

@socketio.on("initialise_client")
def initialise_client(data):
    socket_id = data['socket_id']
    transcriptions[socket_id] = ''
    history = data.get('history', [])
    instructions = data.get('instructions','')
    llm_memory_sessions[socket_id] = ConversationBufferWindowMemory(return_messages=True, k=10)
    llm_memory_sessions[socket_id].chat_memory.add_message(SystemMessage(content=instructions))
    tts_flush[socket_id] = False
    abort_status[socket_id] = False
    audio_queue[socket_id] = queue.Queue()

    starting_message = data["message"]

    for msg in history:
        if msg["source"] == "user":
            llm_memory_sessions[socket_id].chat_memory.add_user_message(msg["message"])
        else:
            llm_memory_sessions[socket_id].chat_memory.add_ai_message(msg["message"])


    try:
        with app.app_context():
            stt_connections[socket_id] = deepgram.listen.websocket.v("1")
            tts_connections[socket_id] = deepgram.speak.websocket.v("1")

            def on_open(self, open, **kwargs):
                print(f"Connection opened: {open}")

            def on_binary_data(self, data, **kwargs):
                audio_queue[socket_id].put(data)

            def on_close(self, close, **kwargs):
                print(f"Connection closed: {close}")

            # Set up event handlers
            tts_connections[socket_id].on(SpeakWebSocketEvents.Open, on_open)
            tts_connections[socket_id].on(SpeakWebSocketEvents.AudioData, on_binary_data)
            tts_connections[socket_id].on(SpeakWebSocketEvents.Close, on_close)


        
            def on_message(_, result, **kwargs):
                transcript = result.channel.alternatives[0].transcript
                is_final = result.is_final
                if is_final:
                    transcriptions[socket_id] = transcriptions[socket_id] + transcript
                if len(transcript) == 0:
                    return
                
            options = {
                "model":"aura-2-thalia-en",
                "encoding":"linear16",
                "sample_rate":48000,
                "container": "none",
                "control": True
            }

            # Start the connection
            if tts_connections[socket_id].start(options) is False:
                print("Failed to start TTS connection")
            
            tts_connections[socket_id].send_text(starting_message)
            tts_flush[socket_id] = True
            tts_connections[socket_id].flush()
                

            stt_connections[socket_id].on(LiveTranscriptionEvents.Transcript, on_message)
            options = LiveOptions(
                    model="nova-3",
                    punctuate=True,
                    interim_results=True,
                    sample_rate=16000,
                    encoding="linear16",
                    channels=1,

                )
            if stt_connections[socket_id].start(options) is False:
                print("Failed to start STT connection")
                return
    except Exception as e:
        with app.app_context():
            print("An Error Occured in Establishing STT Connection with Deepgram: ",e)
            if socket_id in stt_connections:
                stt_connections[socket_id].finish()




from queue import Empty

def get_combined_audio(audio_queue, num_chunks, timeout=None, return_bytes=False):
    """
    Pulls up to `num_chunks` items from `audio_queue`, concatenates them, and returns
    a single audio buffer.

    Args:
        audio_queue (queue.Queue): queue containing incoming audio chunks (bytes or np.ndarray).
        num_chunks (int): number of chunks to combine.
        timeout (float or None): how long to wait for each chunk (in seconds). None = block indefinitely.
        return_bytes (bool): if True, returns raw bytes; otherwise returns np.int16 array.

    Returns:
        np.ndarray or bytes: concatenated audio. If fewer than num_chunks were available,
                             returns whatever it could pull.
    """
    buffers = []
    for _ in range(num_chunks):
        try:
            chunk = audio_queue.get(timeout=timeout)
        except Empty:
            break

        # If chunk is raw bytes, convert to int16 array
        if isinstance(chunk, (bytes, bytearray)):
            arr = np.frombuffer(chunk, dtype=np.int16)
        elif isinstance(chunk, np.ndarray):
            arr = chunk
        else:
            raise TypeError(f"Unexpected chunk type: {type(chunk)}")

        buffers.append(arr)

    if not buffers:
        return b'' if return_bytes else np.array([], dtype=np.int16)

    combined = np.concatenate(buffers)
    return combined.tobytes() if return_bytes else combined


@socketio.on("request_tts")
def request_tts(data):
    socket_id = data['socket_id']
    if audio_queue[socket_id].qsize() > 0:
        if (audio_queue[socket_id].qsize() >=50):
            combined_bytes = get_combined_audio(audio_queue[socket_id], num_chunks=50, return_bytes=True)
            socketio.emit("tts_response", {"audio": combined_bytes})
        if tts_flush[socket_id] == True:
            combined_bytes = get_combined_audio(audio_queue[socket_id], num_chunks=audio_queue[socket_id].qsize(), return_bytes=True)
            socketio.emit("tts_response", {"audio": combined_bytes})

        


# Utility to sanitize and truncate text
def clean_and_truncate(text, limit=300):
    # Remove non-ASCII printable characters (emojis, newlines, etc.)
    cleaned = re.sub(r'[^\x20-\x7E]+', '', text).strip()
    return cleaned[:limit]  # Truncate if too long



@socketio.on("llm_streaming")
def llm_streaming(data):
    socket_id = data['socket_id']
    current_transcription = transcriptions[socket_id]
    transcriptions[socket_id] = ''
    tts_flush[socket_id] = False
    abort_status[socket_id] = False
    text_prompt = data.get('prompt', '')
    if text_prompt:
        current_transcription = text_prompt
    print(current_transcription)
    socketio.emit("stt_response", {"stt_response": current_transcription})
    if current_transcription == '':
        return
    llm_memory_sessions[socket_id].chat_memory.add_user_message(current_transcription)

    all_messages = llm_memory_sessions[socket_id].chat_memory.messages
    full_response = ''
    sentence_buffer = ''

    trimmed_state = trim_messages(
    all_messages,
    token_counter=len,      
    max_tokens=20,            
    strategy="last",
    start_on="human",         
    include_system=True,
    allow_partial=False,)
    global counter

    for chunk in llm.stream(trimmed_state):
        if hasattr(chunk, 'content'):
            chunk_content = chunk.content
            if chunk_content:
                if abort_status[socket_id] == True:
                    tts_connections[socket_id].flush()
                    return
                full_response += chunk_content
                sentence_buffer += chunk_content
                counter += 1

                if counter >= 50:
                    text_to_send = clean_and_truncate(sentence_buffer)
                    if text_to_send:
                        try:
                            if tts_connections[socket_id]:
                                tts_connections[socket_id].send_text(text_to_send)
                        except Exception as e:
                            print(f"❌ Error sending to TTS: {e}")
                    sentence_buffer = ''
                    counter = 0

    # Flush any remaining buffer
    final_text = clean_and_truncate(sentence_buffer)
    if final_text:
        try:
            if tts_connections[socket_id]:
                tts_connections[socket_id].send_text(final_text)
        except Exception as e:
            print(f"❌ Error sending final TTS buffer: {e}")

    # Flush and mark as flushed
    try:
        if tts_connections[socket_id]:
            tts_connections[socket_id].flush()
            tts_flush[socket_id] = True
    except Exception as e:
        print(f"❌ Error during TTS flush: {e}")

    print(full_response)
    socketio.emit("llm_response", {"llm_response": full_response})


@socketio.on("starting_message")
def speak_starting_message(data):
    try:
        socket_id = data['socket_id']
        message = data['message']
        print(socket_id)
        print(message)
        if tts_connections[socket_id]:
            try:
                tts_connections[socket_id].send_text(message)
                tts_connections[socket_id].flush()
            except Exception as e:
                print(e)
    except Exception as e:
        print("Error:",e)





    



@socketio.on("user_audio_chunk")
def transcribe_user_audio(data):
    socket_id = data['socket_id']
    audio = data['audio']
    stt_connections[socket_id].send(audio)








if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5001 )
