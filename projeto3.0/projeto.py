import os
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
import torchaudio
import librosa
from datasets import Dataset, load_metric


# Check if a GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo pré-treinado Wav2Vec
model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base').to(device)

# Carregar o tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained('facebook/wav2vec2-base')

# Carregar os dados de treinamento

pathToDataset = r'C:\Users\017597631\www\Projects\notebooks\projeto3.0\datasets'
dataset = []
labels = []
datasetTest = []
counter = 0
for folderForLabel in os.listdir(pathToDataset): 
    pathForFile = os.path.join(pathToDataset, folderForLabel) 
    for fileName in os.listdir(pathForFile): 
        # pathFilename = os.path.join(pathForFile, fileName)
        # aud, sample_rate = librosa.load(pathFilename, sr=16000)
        dataset.append(aud)
        labels.append(counter)
        datasetTest.append({"audio": aud, "label": counter, "path": pathFilename})
    counter += counter

# Separar o conjunto de dados em treinamento, validação e teste
train_data, val_data,  = train_test_split(dataset, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

print(train_data[0])

dataset

#Definir os argumentos de treinamento
training_args = TrainingArguments(
    output_dir='./output',  # Diretório de saída
    num_train_epochs=1,    # Número de épocas de treinamento
    per_device_train_batch_size=8,
    save_total_limit=2,
)

# Definir a função de pré-processamento dos dados
def preprocess_function(examples):
    print(examples)
    inputs = tokenizer(
        examples["audio"], 
        padding="longest", 
        truncation=True, 
        return_tensors="pt",
        save_strategy='epoch'
    )
    inputs["input_values"] = inputs.input_values.squeeze()
    inputs["labels"] = examples["label"]
    return inputs

# Criar o objeto Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=datasetTest,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=preprocess_function,
)


# Iniciar o treinamento
trainer.train()

checkpoint_dir = './checkpoint'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

model.save_pretrained(checkpoint_dir)
tokenizer.save_pretrained(checkpoint_dir)