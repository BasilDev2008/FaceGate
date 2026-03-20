import anthropic
import os
from database import Database
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import openai # for speech to text
from elevenlabs import ElevenLabs, VoiceSettings # for the AI therapists voice
import tempfile
class TherapyEngine:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key = os.getenv("ANTHROPIC_API_KEY")
        )
        self.db = Database()
        self.model = "claude-opus-4-6"
        self.max_tokens = 1000
        self.openai_client = openai.OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
        self.eleven_client = ElevenLabs(api_key = os.getenc("ELEVENLABS_API_KEY"))
        self.voice_id = os.getenv("ELEVENLABS_VOICE_ID")
        self.sample_rate = 16000
        self.max_duration = 300
        self.silence_threshold = 2 # stop recording user's voice after 2 seconds of silence
    def build_system_prompt(self, user, session_history):
        # builds a set of personalized instructions for claude
        # it tells claude who it is, who it's talking to, and what it knows about them
        return f"""You are a calm, professional male AI therapist. You have a warm but composed presence, like a trusted doctor who genuinely cares but never loses composure.
    
You are speaking with {user.name}
Here is everything you know about {user.name}
{session_history}
Your personality:
- You speak in a calm, measured, confident tone  # never robotic or clinical
- You use natural language — short sentences, occasional "..." when thinking  # feels like a real person
- You say things like "I hear you", "That makes sense", "Tell me more about that"  # empathetic phrases
- You remember small details from past sessions and bring them up naturally  # builds trust over time
- You never lecture — you guide through questions  # therapy technique
- You have a subtle dry wit but only when appropriate  # human touch
- You make the user feel heard above everything else  # most important quality
Your guidelines:
- You use CBT techniques naturally without naming them  # professional but not clinical
- You never give medical diagnoses  # legal protection
- You recommend professional help if you detect serious distress  # user safety first
- You track emotional patterns over time  # long term care
- Reference past sessions naturally when relevant  # memory makes it personal
Current session start. Greet {user.name} personally based on what you know about them."""
    def get_session_history(self, user_id):
        sessions = self.db.get_sessions(user_id)
        if not sessions:
            return "This is their first session. You don't know anything about them"
        history = ""
        for session in sessions:
            history += f"\nSession {session.session_number} ({session.date}):\n"
            history += f"Summary: {session.summary}\n"  # what was discussed
            history += f"Emotional state: {session.emotional_state}\n"  # how they were feeling
            history += f"Key topics: {session.key_topics}\n"  # main themes of that session
        return history
def chat(self, user_id, conversation_history):
    # main conversation loop — listens, thinks, speaks, repeats
    user = self.db.get_user(user_id)  # get user details from database

    session_history = self.get_session_history(user_id)  # load their full therapy history

    system_prompt = self.build_system_prompt(user, session_history)  # build personalized instructions

    print(f"Session started for {user.name}")
    print("Speak when ready. Session will end when you say 'goodbye' or 'end session'")

    # start the session — therapist greets the user first
    greeting_response = self.client.messages.create(  # ask claude to greet the user
        model=self.model,
        max_tokens=self.max_tokens,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": "Please greet me and start our session."  # trigger the greeting
        }]
    )

    greeting = greeting_response.content[0].text  # get greeting text
    print(f"Therapist: {greeting}")  # print to terminal
    self.speak(greeting)  # speak the greeting out loud

    conversation_history.append({  # add greeting to history
        "role": "assistant",
        "content": greeting
    })

    while True:  # keep conversation going until user ends session
        user_message = self.listen()  # listen for user's voice

        if user_message is None:  # if nothing was heard
            self.speak("I didn't catch that. Could you say that again?")  # ask to repeat
            continue

        # check if user wants to end the session
        end_phrases = ["goodbye", "end session", "bye", "stop", "exit", "finish"]
        if any(phrase in user_message.lower() for phrase in end_phrases):
            farewell = "It was really good talking with you today. Take care of yourself and I'll see you next time."
            self.speak(farewell)  # say goodbye
            print(f"Therapist: {farewell}")
            break  # end the session loop

        conversation_history.append({  # add user message to history
            "role": "user",
            "content": user_message
        })

        response = self.client.messages.create(  # send to claude
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,  # personalized instructions
            messages=conversation_history  # full conversation so far
        )

        assistant_message = response.content[0].text  # get claude's response

        conversation_history.append({  # add response to history
            "role": "assistant",
            "content": assistant_message
        })

        print(f"Therapist: {assistant_message}")  # print to terminal
        self.speak(assistant_message)  # speak the response out loud

    # session ended — summarize and save
    summary = self.summarize_session(user_id, conversation_history)  # summarize the session
    print(f"Session summary saved.")

    return conversation_history  # return full conversation
    def summarize_session(self, user_id, conversation_history):
        # called at the end of every session to create a summary for long term memory
        # this summary gets stored in the database and loaded in future sessions
        session_count = self.db.get_session_count(user_id)  # how many sessions this user has had

        summary_prompt = """Analyze this therapy session and provide a structured summary with:
1. Main topics discussed  # what was talked about
2. Emotional state of the user on a scale of 1-10  # 10 being most positive
3. Key insights or breakthroughs  # important moments
4. Patterns noticed  # recurring themes or behaviors
5. Recommended focus for next session  # what to follow up on

Keep it concise but thorough. This summary will be loaded into future sessions as long term memory."""

        response = self.client.messages.create(  # ask Claude to summarize the session
            model=self.model,
            max_tokens=500,  # summaries should be shorter than responses
            messages=[
                {
                    "role": "user",
                    "content": f"Please summarize this therapy session:\n\n{str(conversation_history)}\n\n{summary_prompt}"
                }
            ]
        )

        summary = response.content[0].text  # extract the summary text

        self.db.save_session(  # save the session to the database for future memory
            user_id=user_id,
            session_number=session_count + 1,  # increment session number
            summary=summary,  # the summary Claude wrote
            emotional_state="extracted from summary",  # will be parsed properly later
            key_topics="extracted from summary",  # will be parsed properly later
            full_conversation=str(conversation_history)  # save the full conversation too
        )

        return summary  # return the summary
def listen(self):
    # records the user's voice and converts it to text
    import webrtcvad # for voice activitiy detection

    vad = webrtcvad.Vad(2)
    sample_rate = 16000
    chunk_duration = 0.03
    chunk_size = int(sample_rate * chunk_duration)

    print("Listening...")

    audio_chunks = []
    silent_chunks = 0
    max_silent_chunks = int(self.silence_threshold / chunk_duration)
    max_chunks = int(self.max_duration / chunk_duration)
    total_chunks = 0
    speaking_started = False
    with sd.RawInputStream(  # open microphone stream
        samplerate=sample_rate,  # sample rate
        channels=1,  # mono audio
        dtype='int16',  # 16 bit audio — required by webrtcvad
        blocksize=chunk_size  # size of each chunk
    ) as stream:
        while total_chunks < max_chunks:  # keep recording until max duration
            chunk, _ = stream.read(chunk_size)  # read one chunk from microphone
            audio_chunks.append(chunk)  # save the chunk
            total_chunks += 1  # increment counter
            
            is_speech = vad.is_speech(bytes(chunk), sample_rate)  # check if chunk has voice
            
            if is_speech:  # if voice detected
                speaking_started = True  # user has started speaking
                silent_chunks = 0  # reset silence counter
            else:  # if silence detected
                if speaking_started:  # only count silence after user has started speaking
                    silent_chunks += 1  # increment silence counter
                    
            if speaking_started and silent_chunks >= max_silent_chunks:  # if user stopped speaking
                break  # stop recording
    
    if not audio_chunks:  # if nothing was recorded
        return None
    
    # combine all chunks into one audio array
    audio_data = np.frombuffer(b''.join([bytes(c) for c in audio_chunks]), dtype=np.int16)
    
    # save as temporary wav file so whisper can read it
    temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)  # create temp file
    wav.write(temp_file.name, sample_rate, audio_data)  # write audio to file
    
    # send to whisper for transcription
    with open(temp_file.name, 'rb') as audio_file:  # open the temp file
        transcript = self.openai_client.audio.transcriptions.create(  # send to whisper
            model="whisper-1",  # whisper model
            file=audio_file  # the audio file
        )
    
    os.remove(temp_file.name)  # delete temp file after transcription
    
    print(f"You said: {transcript.text}")  # show what was heard
    return transcript.text  # return the transcribed text
def speak(self, text):
    # converts claude's text response to a realistic human voice using elevenlabs
    
    audio = self.eleven_client.text_to_speech.convert(  # send text to elevenlabs
        voice_id=self.voice_id,  # which voice to use — the therapist's voice
        text=text,  # claude's response text
        model_id="eleven_multilingual_v2",  # most realistic elevenlabs model
        voice_settings=VoiceSettings(
            stability=0.75,  # how consistent the voice sounds — higher is more stable
            similarity_boost=0.85,  # how closely it matches the original voice
            style=0.3,  # how much expression — lower is calmer, fits a therapist
            use_speaker_boost=True  # enhances voice clarity
        )
    )
    
    # save audio to temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)  # create temp file
    for chunk in audio:  # elevenlabs returns audio in chunks
        temp_file.write(chunk)  # write each chunk to file
    temp_file.close()  # close the file
    
    # play the audio
    import pygame  # for playing audio
    pygame.mixer.init()  # initialize audio player
    pygame.mixer.music.load(temp_file.name)  # load the audio file
    pygame.mixer.music.play()  # play the audio
    
    while pygame.mixer.music.get_busy():  # wait until audio finishes playing
        pygame.time.Clock().tick(10)  # check every 10ms
    
    pygame.mixer.quit()  # cleanup audio player
    os.remove(temp_file.name)  # delete temp file after playing