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
# 🛠️ CONFIGURAÇÃO GLOBAL DO PIPELINE (PARAMETROS FIXADOS OTIMIZADOS)
# =====================================================================
CONFIG = {
    "url_train": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/train.csv",
    "url_val": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/val.csv",
    "url_test": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/test.csv",

    # 👤 MEMBRO 1 (Fixado com o melhor valor encontrado no Optuna)
    "max_training_samples": None,  
    "max_sequence_length": 256,    

    # 👤 MEMBRO 2
    "model_name": "neuralmind/bert-large-portuguese-cased", 

    # 🎯 HIPERPARAMETROS FIXADOS (Idênticos aos do seu histórico de sucesso)
    "fixed_learning_rate": 3.1504750575460564e-05,
    "fixed_weight_decay": 0.08049241505310056,
    "fixed_train_batch_size": 16,
    "fixed_eval_batch_size": 8,

    # 👤 EIXO DO MEMBRO 4: DESATIVADO (Modelo base puro performou melhor)
    "use_custom_architecture": False,  
    "arch_pooling_strategy": "mean", 
    "arch_freeze_layers": 6,         
    "arch_dropout_rate": 0.3,        
    "arch_hidden_dimension": 256,    

    # 👤 MEMBRO 5 (Ativado para o novo experimento)
    "use_custom_loss": False,         
}

LABEL_MAP = {"Esquerda": 0, "Direita": 1, "Neutro": 2}

# =====================================================================
# 📐 INICIALIZAÇÃO DO MODELO
# =====================================================================
def model_init():
    if CONFIG["use_custom_architecture"]:
        return CustomBERTimbauClassifier(CONFIG["model_name"], num_labels=3)
    else:
        # Carrega diretamente a arquitetura base padrão do BERTimbau
        return AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=3)

# =====================================================================
# 🏛️ CLASSES ESTRUTURAIS DO PIPELINE
# =====================================================================
class CustomBERTimbauClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.pooling_strategy = CONFIG["arch_pooling_strategy"]
        self.num_labels = num_labels

        num_freezes = CONFIG["arch_freeze_layers"]
        if num_freezes > 0:
            for param in self.bert.embeddings.parameters(): param.requires_grad = False
            for i in range(num_freezes):
                for param in self.bert.encoder.layer[i].parameters(): param.requires_grad = False

        input_dim = self.bert.config.hidden_size
        if self.pooling_strategy == "concat_4": input_dim = self.bert.config.hidden_size * 4

        hidden_dim = CONFIG["arch_hidden_dimension"]
        self.dropout = torch.nn.Dropout(CONFIG["arch_dropout_rate"])
        
        if hidden_dim > 0:
            self.intermediate_dense = torch.nn.Linear(input_dim, hidden_dim)
            self.activation = torch.nn.GELU()
            self.classifier = torch.nn.Linear(hidden_dim, num_labels)
        else:
            self.intermediate_dense = None
            self.classifier = torch.nn.Linear(input_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling_strategy == "cls":
            pooled_output = outputs.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == "mean":
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
        elif self.pooling_strategy == "concat_4":
            hidden_states = outputs.hidden_states
            pooled_output = torch.cat([hidden_states[-i][:, 0, :] for i in range(1, 5)], dim=-1)
        else:
            raise ValueError(f"Estratégia de pooling desconhecida: {self.pooling_strategy}")

        x = self.dropout(pooled_output)
        if self.intermediate_dense is not None:
            x = self.intermediate_dense(x)
            x = self.activation(x)
            x = self.dropout(x)
        logits = self.classifier(x)
        
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if CONFIG["use_custom_loss"]:
            # Aplica pesos: Esquerda (1.0), Direita (1.0), Neutro (1.5) na GPU automaticamente
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
# 🏃‍♂️ PIPELINE PRINCIPAL DEFINTIVO (TREINAMENTO DO MEMBRO 5)
# =====================================================================
def main():
    # 🖥️ VERIFICAÇÃO DE HARDWARE
    print("\n🖥️  Verificando Hardware de Execução...")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 SUCESSO: GPU NVIDIA Detectada! O script usará a placa: {gpu_name}")
        use_gpu_optimization = True
    else:
        print("⚠️  AVISO: Nenhuma GPU NVIDIA detectada. Rodando na CPU.")
        use_gpu_optimization = False

    print("🌐 Carregando dados públicos do GitHub...")
    try:
        df_train = pd.read_csv(CONFIG["url_train"])
        df_val = pd.read_csv(CONFIG["url_val"])
    except Exception as e:
        print(f"❌ Erro ao baixar dados: {e}")
        return

    if CONFIG["max_training_samples"] is not None:
        df_train = df_train.sample(n=CONFIG["max_training_samples"], random_state=42).reset_index(drop=True)

    print(f"🔤 Inicializando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # Como os valores são fixos, criamos o dataset otimizado uma única vez fora de loops
    print(f"📦 Construindo Datasets com tamanho fixo de {CONFIG['max_sequence_length']} tokens...")
    final_train_dataset = PoliticalBiasDataset(df_train, tokenizer, CONFIG["max_sequence_length"])
    final_val_dataset = PoliticalBiasDataset(df_val, tokenizer, CONFIG["max_sequence_length"])

    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("bertimbau-political-bias")
    user_name = getpass.getuser()

    # =====================================================================
    # 🏋️ TREINAMENTO CONSOLIDADO FINAL
    # =====================================================================
    print(f"\n🏋️ Iniciando treinamento final com Custom Loss ativo (Peso 1.5 para Neutros)...")
    
    final_training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=6, 
        logging_dir="./logs",
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
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
        eval_dataset=final_val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    # Inicializa trackamento no MLflow direcionado para o experimento final do Membro 5
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.start_run(run_name=f"{user_name}-large-bert")
        mlflow.log_params({
            "max_sequence_length": CONFIG["max_sequence_length"],
            "learning_rate": CONFIG["fixed_learning_rate"],
            "weight_decay": CONFIG["fixed_weight_decay"],
            "use_custom_architecture": CONFIG["use_custom_architecture"],
            "use_custom_loss": CONFIG["use_custom_loss"],
            "loss_weights": "[1.0, 1.0, 1.5]"
        })

    final_trainer.train()

    print("📉 Gerando Relatórios Finais e Matriz de Confusão...")
    eval_results = final_trainer.evaluate()
    
    predictions_output = final_trainer.predict(final_val_dataset)
    preds = np.argmax(predictions_output.predictions, axis=1)
    labels_reais = predictions_output.label_ids

    classes = ["Esquerda", "Direita", "Neutro"]
    cm = confusion_matrix(labels_reais, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.title('Matriz de Confusão - Validação Final (Custom Loss)')
    plt.tight_layout()

    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.log_figure(fig, "matriz_confusao_membro5.png")
        mlflow.log_metric("final_macro_f1", eval_results['eval_macro_f1'])
        mlflow.log_metric("final_accuracy", eval_results['eval_accuracy'])
        mlflow.end_run()
        print("\n✅ Sucesso total! Verifique o gráfico e as métricas atualizadas no DagsHub.")
        
    plt.close()

if __name__ == "__main__":
    main()
