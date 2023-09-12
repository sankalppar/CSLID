from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2Model
import torch
from datasets import load_dataset

dataset = load_dataset("audiofolder", data_dir="PartB_Gujarati/Dev/Audio")['train']
sampling_rate = dataset[0]["audio"]["sampling_rate"]
chunk_size = int(0.2*sampling_rate)
print(dataset[0]["audio"]["array"][:chunk_size])

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to("cuda")

with torch.no_grad():
    inputs = feature_extractor(dataset[0]["audio"]["array"][:chunk_size], sampling_rate = sampling_rate, return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    print(last_hidden_states.shape)

'''
with torch.no_grad():
    for i in range(len(dataset)):
        inputs = feature_extractor(dataset[i]["audio"]["array"], sampling_rate = sampling_rate, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        torch.save(last_hidden_states, "dev/" + dataset[i]["audio"]["path"].split("/")[-1].split(".")[0] + ".pt")
'''
