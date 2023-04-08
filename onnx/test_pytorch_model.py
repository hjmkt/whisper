import whisper
from whisper.model import AudioEncoderWrapper


def run():
    model = whisper.load_model("base")
    model.encoder = AudioEncoderWrapper(model.encoder, model.decoder)

    audio = whisper.load_audio("/home/hjmkt/Documents/en.mp3")
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    options = whisper.DecodingOptions()
    result = whisper.decode(model, mel, options)

    print(result.text)

if __name__ == "__main__":
    run()
