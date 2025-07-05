#--------------------------Inference Pipeline-----------------------------------------#
import torch
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
#model = WhisperForConditionalGeneration.from_pretrained("./whisper-finetuned/checkpoint-25")
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")


def transcribe(audio_path):
    speech, sr = librosa.load(audio_path, sr=16000)
    inputs = processor.feature_extractor(speech, sampling_rate=16000, return_tensors="pt")
    attention_mask = torch.ones_like(inputs["input_features"]).long()
    
    with torch.no_grad():
        predicted_ids = model.generate(inputs["input_features"], attention_mask=attention_mask, language="en")

    transcription = processor.tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

# Test
"""if __name__ == "__main__":
    audio_file = "/mnt/sda1/FYP_2024/Helitha/ExentAI Task/good-boy-praiseful-male-smartsound-fx-1-00-01.mp3"
    transcription = transcribe(audio_file)
    print("Transcription:", transcription)
"""