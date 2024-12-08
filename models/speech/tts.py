import pyttsx3

class TTSModel:
    def __init__(self, rate=150, volume=1.0, voice=None):
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)
        if voice:
            self.engine.setProperty('voice', voice)

    def synthesize(self, text: str, output_file: str = None) -> bytes:
        try:
            if output_file:
                self.engine.save_to_file(text, output_file)
                self.engine.runAndWait()
                with open(output_file, "rb") as f:
                    return f.read()
            else:
                self.engine.say(text)
                self.engine.runAndWait()
                return b"Audio generated in real-time."
        except Exception as e:
            return f"Error in synthesis: {str(e)}".encode()
