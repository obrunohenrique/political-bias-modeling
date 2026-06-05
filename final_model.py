import os
import getpass
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset
from dotenv import load_dotenv
import mlflow
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EvalPrediction, 
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

# Carrega chaves do DagsHub (MLflow) configuradas no .env local
load_dotenv()

# =====================================================================
# 🛠️ CONFIGURAÇÃO GLOBAL DO PIPELINE (MODELO DEFINITIVO)
# =====================================================================
CONFIG = {
    "url_train": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/train.csv",
    "url_val": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/val.csv",
    "url_test": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/test.csv",

    # 👤 CONFIGURAÇÃO CONFIRMADA DO MEMBRO 1
    "max_training_samples": None,  
    "max_sequence_length": 256,    # 👈 Fixado em 256 tokens conforme melhor resultado

    # 👤 MODELO BASE
    "model_name": "neuralmind/bert-base-portuguese-cased", 

    # 🎯 HIPERPARAMETROS FIXADOS OTIMIZADOS
    "fixed_learning_rate": 3.1504750575460564e-05,
    "fixed_weight_decay": 0.08049241505310056,
    "fixed_train_batch_size": 16,
    "fixed_eval_batch_size": 8,

    # 👤 ARQUITETURA CUSTOMIZADA: DESATIVADA (Padrão puro)
    "use_custom_architecture": False,  
    "arch_pooling_strategy": "mean", 
    "arch_freeze_layers": 6,         
    "arch_dropout_rate": 0.3,        
    "arch_hidden_dimension": 256,    

    # 👤 LOSS CUSTOMIZADA: DESATIVADA (CrossEntropy Padrão)
    "use_custom_loss": False,         
}

LABEL_MAP = {"Esquerda": 0, "Direita": 1, "Neutro": 2}

# =====================================================================
# 📐 INICIALIZAÇÃO DO MODELO PADRÃO
# =====================================================================
def model_init():
    # Inicializa a arquitetura base padrão e oficial do BERTimbau
    return AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=3)

# =====================================================================
# 🏛️ CLASSES ESTRUTURAIS DO PIPELINE
# =====================================================================
class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if CONFIG["use_custom_loss"]:
            pesos_classes = torch.tensor([1.0, 1.0, 1.5], device=model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=pesos_classes)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return (loss, outputs) if return_outputs else loss

class PoliticalBiasDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe['texto'].astype(str).values
        self.labels = dataframe['vies_politico'].map(LABEL_MAP).values
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, add_special_tokens=True, max_length=self.max_len, padding='max_length', truncation=True, return_attention_mask=True, return_tensors='pt')
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'labels': torch.tensor(label, dtype=torch.long)}

def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    return {"macro_f1": f1_score(labels, preds, average='macro'), "accuracy": accuracy_score(labels, preds)}

# =====================================================================
# 🏃‍♂️ TREINAMENTO, SALVAMENTO E AVALIAÇÃO NO TESTE
# =====================================================================
def main():
    # 🖥️ VERIFICAÇÃO DE HARDWARE LOCAL
    print("\n🖥️  Verificando Hardware de Execução...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 SUCESSO: GPU NVIDIA Detectada! Executando na placa: {gpu_name}")
        use_gpu_optimization = True
    else:
        print("⚠️  AVISO: GPU não detectada. O pipeline rodará na CPU (Lento).")
        use_gpu_optimization = False

    print("🌐 Baixando os conjuntos de dados (Treino, Validação e TESTE)...")
    try:
        df_train = pd.read_csv(CONFIG["url_train"])
        df_val = pd.read_csv(CONFIG["url_val"])
        df_test = pd.read_csv(CONFIG["url_test"]) # 👈 Carregando o conjunto de teste real
    except Exception as e:
        print(f"❌ Erro ao baixar dados do GitHub: {e}")
        return

    if CONFIG["max_training_samples"] is not None:
        df_train = df_train.sample(n=CONFIG["max_training_samples"], random_state=42).reset_index(drop=True)

    print(f"🔤 Inicializando Tokenizer do BERTimbau...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    print(f"📦 Tokenizando e construindo Datasets com tamanho de {CONFIG['max_sequence_length']}...")
    final_train_dataset = PoliticalBiasDataset(df_train, tokenizer, CONFIG["max_sequence_length"])
    final_val_dataset = PoliticalBiasDataset(df_val, tokenizer, CONFIG["max_sequence_length"])
    final_test_dataset = PoliticalBiasDataset(df_test, tokenizer, CONFIG["max_sequence_length"]) # 👈 Criando o dataset de teste

    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("bertimbau-political-bias")
    user_name = getpass.getuser()

    # =====================================================================
    # 1. SCRIPT PARA TREINAR O MODELO FINAL
    # =====================================================================
    print("\n🏋️ [PASSO 1] Iniciando o treinamento final definitivo (6 épocas)...")
    
    final_training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=6, 
        logging_dir="./logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True, # Garante que recuperamos o melhor peso gerado na validação
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="mlflow",
        learning_rate=CONFIG["fixed_learning_rate"],
        weight_decay=CONFIG["fixed_weight_decay"],
        per_device_train_batch_size=CONFIG["fixed_train_batch_size"],
        per_device_eval_batch_size=CONFIG["fixed_eval_batch_size"],
        fp16=use_gpu_optimization,                  
        dataloader_pin_memory=use_gpu_optimization   
    )

    final_trainer = CustomLossTrainer(
        model_init=model_init,  
        args=final_training_args,
        train_dataset=final_train_dataset,
        eval_dataset=final_val_dataset, # Validação usada para o Early Stopping achar o ponto ótimo
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    # Iniciando a sessão do MLflow para registrar tudo
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.start_run(run_name=f"{user_name}-PRODUCAO-MODELO-FINAL")
        mlflow.log_params({
            "max_sequence_length": CONFIG["max_sequence_length"],
            "status": "Modelo Final de Produção",
            "architecture": "BERTimbau Base Puro"
        })

    # Treina o modelo
    final_trainer.train()

    # =====================================================================
    # 2. GUARDAR O MODELO FINAL (MELHOR PRÁTICA)
    # =====================================================================
    print("\n💾 [PASSO 2] Salvando o modelo e o tokenizador estruturados...")
    
    # Caminho da pasta onde o modelo será guardado localmente
    caminho_salvamento_local = "./modelo_final_producao"
    
    # Salva os pesos matemáticos ótimos do modelo e as configurações
    final_trainer.save_model(caminho_salvamento_local)
    # Salva o arquivo de vocabulário do tokenizador junto na mesma pasta (essencial para carregar depois)
    tokenizer.save_pretrained(caminho_salvamento_local)
    print(f"📁 Modelo salvo localmente com sucesso na pasta: '{caminho_salvamento_local}'")

    # Registra o modelo diretamente no repositório de artefatos do MLflow (Substitui o joblib)
    if os.getenv("MLFLOW_TRACKING_URI"):
        print("☁️ Fazendo upload do modelo final como um artefato nativo do MLflow...")
        components = {
            "model": final_trainer.model,
            "tokenizer": tokenizer,
        }
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path="modelo_final_bertimbau"
        )

    # =====================================================================
    # 3. TESTAR NO CONJUNTO DE TESTE (AVALIAÇÃO REAL CEGA)
    # =====================================================================
    print("\n🎯 [PASSO 3] Executando predições cegas no CONJUNTO DE TESTE...")
    
    # Roda a predição no dataset de teste que nunca foi visto em nenhuma época
    test_results = final_trainer.evaluate(eval_dataset=final_test_dataset, metric_key_prefix="test")
    
    print(f"\n📊 --- RESULTADOS FINAIS NO CONJUNTO DE TESTE ---")
    print(f"🏆 Macro F1 no Teste: {test_results['test_macro_f1']:.4f}")
    print(f"📈 Acurácia no Teste: {test_results['test_accuracy']:.4f}")

    # Geração da Matriz de Confusão do Teste
    predictions_output = final_trainer.predict(final_test_dataset)
    preds = np.argmax(predictions_output.predictions, axis=1)
    labels_reais = predictions_output.label_ids

    classes = ["Esquerda", "Direita", "Neutro"]
    cm = confusion_matrix(labels_reais, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('Classe Real (Gabarito do Teste)')
    plt.xlabel('Classe Predita pelo Modelo')
    plt.title('Matriz de Confusão Final - Conjunto de Teste')
    plt.tight_layout()

    # Registrando as métricas e imagem finais no MLflow
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.log_figure(fig, "matriz_confusao_teste_final.png")
        mlflow.log_metric("teste_final_macro_f1", test_results['test_macro_f1'])
        mlflow.log_metric("teste_final_accuracy", test_results['test_accuracy'])
        mlflow.end_run()
        print("\n🚀 FIM! O modelo foi treinado, salvo localmente, enviado ao MLflow e avaliado com sucesso no Teste.")
        
    plt.close()

if __name__ == "__main__":
    main()
    