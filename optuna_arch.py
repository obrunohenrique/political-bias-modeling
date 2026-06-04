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
import optuna  
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
# 🛠️ CONFIGURAÇÃO GLOBAL DO PIPELINE
# =====================================================================
CONFIG = {
    "url_train": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/train.csv",
    "url_val": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/val.csv",
    "url_test": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/test.csv",

    # 👤 MEMBRO 1
    "max_training_samples": None,  
    "max_sequence_length": 128,    

    # 👤 MEMBRO 2
    "model_name": "neuralmind/bert-base-portuguese-cased", 

    # 🎯 HIPERPARAMETROS FIXADOS (Vindos do seu melhor run anterior)
    "fixed_learning_rate": 3.1504750575460564e-05,
    "fixed_weight_decay": 0.08049241505310056,
    "fixed_train_batch_size": 16,
    "fixed_eval_batch_size": 8,

    # 👤 EIXO DO MEMBRO 4: CONFIGURAÇÕES ESTRUTURAIS AUTOMATIZADAS
    "use_custom_architecture": True,  # 👈 ALTERADO: Forçado True para usar a variação estrutural
    "arch_pooling_strategy": "mean", 
    "arch_freeze_layers": 6,         
    "arch_dropout_rate": 0.3,        
    "arch_hidden_dimension": 256,    

    # 👤 MEMBRO 5
    "use_custom_loss": False,         
}

LABEL_MAP = {"Esquerda": 0, "Direita": 1, "Neutro": 2}

# =====================================================================
# 🎛️ ESPAÇO DE BUSCA EXCLUSIVO DA ARQUITETURA (EIXO MEMBRO 4)
# =====================================================================
def my_hp_space(trial):
    # O Optuna agora varia apenas os parâmetros da arquitetura customizada
    return {
        "arch_pooling_strategy": trial.suggest_categorical("arch_pooling_strategy", ["cls", "mean", "concat_4"]),
        "arch_freeze_layers": trial.suggest_categorical("arch_freeze_layers", [0, 6, 12]),
        "arch_dropout_rate": trial.suggest_float("arch_dropout_rate", 0.1, 0.5),
        "arch_hidden_dimension": trial.suggest_categorical("arch_hidden_dimension", [0, 256]),
    }

# =====================================================================
# 📐 INICIALIZAÇÃO DINÂMICA DO MODELO (EIXO MEMBRO 4)
# =====================================================================
def model_init():
    # Os parâmetros já são injetados dinamicamente no CONFIG no início do trial
    if CONFIG["use_custom_architecture"]:
        return CustomBERTimbauClassifier(CONFIG["model_name"], num_labels=3)
    else:
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
# 🏃‍♂️ PIPELINE PRINCIPAL 
# =====================================================================
def main():
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

    train_dataset = PoliticalBiasDataset(df_train, tokenizer, CONFIG["max_sequence_length"])
    val_dataset = PoliticalBiasDataset(df_val, tokenizer, CONFIG["max_sequence_length"])

    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
        mlflow.set_experiment("bertimbau-political-bias")
    user_name = getpass.getuser()

    # 🛠️ FUNÇÃO OBJETIVO INTERNA 
    def objective(trial):
        # 1. Captura as decisões estruturais do Optuna e atualiza o CONFIG global
        arch_params = my_hp_space(trial)
        for param_name, param_value in arch_params.items():
            if param_name in CONFIG:
                CONFIG[param_name] = param_value  

        # 2. Instancia os argumentos usando os hiperparâmetros que foram FIXADOS
        local_training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=3, 
            logging_dir="./logs",
            logging_steps=20,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            report_to="mlflow",
            learning_rate=CONFIG["fixed_learning_rate"],           # 👈 FIXADO
            weight_decay=CONFIG["fixed_weight_decay"],             # 👈 FIXADO
            per_device_train_batch_size=CONFIG["fixed_train_batch_size"], # 👈 FIXADO
            per_device_eval_batch_size=CONFIG["fixed_eval_batch_size"]    # 👈 FIXADO
        )

        local_trainer = CustomLossTrainer(
            model_init=model_init,  
            args=local_training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )

        if os.getenv("MLFLOW_TRACKING_URI"):
            parent_run_id = os.getenv("PARENT_RUN_ID")
            with mlflow.start_run(
                run_name=f"trial_{trial.number}", 
                tags={"mlflow.parentRunId": parent_run_id},
                nested=True
            ):
                # Registra os parâmetros da arquitetura testados neste trial
                mlflow.log_params(arch_params)
                # Opcional: Documenta no trial quais foram os hiperparâmetros fixos usados nele
                mlflow.log_params({
                    "fixed_lr": CONFIG["fixed_learning_rate"],
                    "fixed_wd": CONFIG["fixed_weight_decay"]
                })
                
                local_trainer.train()
                eval_results = local_trainer.evaluate()
                return eval_results["eval_macro_f1"]
        else:
            local_trainer.train()
            eval_results = local_trainer.evaluate()
            return eval_results["eval_macro_f1"]

    # 🚀 DISPARO DO ESTUDO OPTUNA
    print("\n🔥 [Optuna] Iniciando variação de arquitetura com Hiperparâmetros Fixos...")
    study = optuna.create_study(direction="maximize")

    if os.getenv("MLFLOW_TRACKING_URI"):
        os.environ["MLFLOW_NESTED_RUN"] = "True"
        with mlflow.start_run(run_name=f"{user_name}-architecture-search") as parent_run:
            os.environ["PARENT_RUN_ID"] = parent_run.info.run_id
            study.optimize(objective, n_trials=15)
    else:
        study.optimize(objective, n_trials=15)

    best_trial = study.best_trial

    print("\n🏆 [Optuna] BUSCA DE ARQUITETURA CONCLUÍDA!")
    print(f"Melhor pontuação alcançada (Macro F1): {best_trial.value:.4f}")
    print(best_trial.params)

    # =====================================================================
    # 🏋️ TREINO CONSOLIDADO FINAL (COM A MELHOR ARQUITETURA ENCONTRADA)
    # =====================================================================
    print("\n🏋️ Iniciando o treinamento final definitivo...")
    
    # Seta a melhor arquitetura encontrada no CONFIG global antes de buildar o modelo final
    for param_name, param_value in best_trial.params.items():
        if param_name in CONFIG:
            CONFIG[param_name] = param_value  

    final_training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=6, # Convergência total
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
        per_device_eval_batch_size=CONFIG["fixed_eval_batch_size"]
    )

    final_trainer = CustomLossTrainer(
        model_init=model_init,  
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.start_run(run_name=f"{user_name}-final-architecture-model")
        mlflow.log_params(best_trial.params)

    final_trainer.train()

    print("📉 Gerando Relatórios Finais e Matriz de Confusão...")
    eval_results = final_trainer.evaluate()
    
    predictions_output = final_trainer.predict(val_dataset)
    preds = np.argmax(predictions_output.predictions, axis=1)
    labels_reais = predictions_output.label_ids

    classes = ["Esquerda", "Direita", "Neutro"]
    cm = confusion_matrix(labels_reais, preds)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.title('Matriz de Confusão - Validação Final')
    plt.tight_layout()

    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.log_figure(fig, "matriz_confusao_final.png")
        mlflow.log_metric("final_macro_f1", eval_results['eval_macro_f1'])
        mlflow.log_metric("final_accuracy", eval_results['eval_accuracy'])
        mlflow.end_run()
        print("✅ Tudo pronto! Vá ao DagsHub conferir qual variação de camada e pooling performou melhor.")
        
    plt.close()

if __name__ == "__main__":
    main()
    