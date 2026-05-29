import os
import getpass
import pandas as pd
import numpy as np
import torch
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
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

# Carrega chaves do DagsHub (MLflow) configuradas no .env local
load_dotenv()

# =====================================================================
# 🛠️ GANCHOS DE OTIMIZAÇÃO (ESPAÇO RESERVADO PARA OS 5 MEMBROS DA EQUIPE)
# =====================================================================
CONFIG = {
    # 📌 REPOSITÓRIO PÚBLICO (Mantenha atualizado com os links do seu grupo)
    "url_train": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/train.csv",
    "url_val": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/val.csv",
    "url_test": "https://raw.githubusercontent.com/obrunohenrique/political-bias-dataset-builder/main/data/labeled/final_dataset/test.csv",

    # 👤 MEMBRO 1: Otimização de Volumetria, Filtros e Amostragem de Entrada
    "max_training_samples": None,  # Permite truncar o dataset (ex: 500) para testes rápidos. None usa tudo.
    "max_sequence_length": 128,    # Limite de tokens por manchete (padding/truncation)

    # 👤 MEMBRO 2: Escolha do Motor de Linguagem Base (Model Selection)
    "model_name": "neuralmind/bert-base-portuguese-cased", # BERTimbau Base padrão

    # 👤 MEMBRO 3: Curva de Convergência (Taxa de Aprendizado e Hiperparâmetros)
    "learning_rate": 2e-5,
    "batch_size": 16,
    "weight_decay": 0.01,

    # 👤 EIXO DO MEMBRO 4: CONFIGURAÇÕES ESTRUTURAIS AUTOMATIZADAS
    "use_custom_architecture": False, # Ativa a arquitetura paramétrica abaixo
    
    "arch_freeze_layers": 6,         # Quantas camadas congelar (0 a 12). 0 = treina o BERT todo.
    "arch_pooling_strategy": "mean", # Como resumir o BERT. Opções: 'cls', 'mean', 'concat_4'
    "arch_hidden_dimension": 256,    # Tamanho da camada intermediária. 0 = direto para as 3 classes.
    "arch_dropout_rate": 0.3,        # Força da regularização contra overfitting

    # 👤 MEMBRO 5: Função de Custo Calibrada (Mitigação de Falsos Neutros)
    "use_custom_loss": False,         # Altere para True para injetar pesos na Loss Function (CrossEntropy)
}

# =====================================================================
# 🎛️ CONFIGURAÇÃO CENTRAL DO RASTREAMENTO (MLFLOW / DAGSHUB)
# =====================================================================
if os.getenv("MLFLOW_TRACKING_URI"):
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment("bertimbau-political-bias")
    
    # 💡 SOLUÇÃO 1: Nome dinâmico baseado nas escolhas do membro
    # Exemplo de saída: "bruno-mean-fr6-dr0.3" ou "mateus-cls-fr0-dr0.0"
    user_name = getpass.getuser()
    if CONFIG["use_custom_architecture"]:
        descritivo_arquitetura = f"{CONFIG['arch_pooling_strategy']}-fr{CONFIG['arch_freeze_layers']}-dr{CONFIG['arch_dropout_rate']}"
    else:
        descritivo_arquitetura = "baseline-padrao"
        
    nome_da_run = f"{user_name}-{descritivo_arquitetura}"
    
    # Inicia a run com o nome inteligente
    active_run = mlflow.start_run(run_name=nome_da_run)
    
    # 💡 SOLUÇÃO 2: Registrar o CONFIG como PARÂMETROS (Viram colunas na UI)
    mlflow.log_params(CONFIG)
    
    # Mantém as tags auxiliares
    mlflow.set_tag("autor_execucao", user_name)
    mlflow.log_dict(CONFIG, "hiperparametros_config.json")
else:
    print("⚠️ MLFLOW_TRACKING_URI não encontrada no .env. O treino rodará sem tracking online.")

# Mapeamento estrito de classes para garantir consistência metodológica
LABEL_MAP = {"Esquerda": 0, "Direita": 1, "Neutro": 2}

# =====================================================================
# 👤 CLASSES AUXILIARES PARA OS MEMBROS 4 E 5 (GANCHOS DE CÓDIGO)
# =====================================================================

# GANCHO DO MEMBRO 4: Se você precisar customizar a arquitetura do BERTimbau
class CustomBERTimbauClassifier(torch.nn.Module):
    def __init__(self, model_name, num_labels=3):
        super().__init__()
        from transformers import AutoModel
        
        # Carrega o modelo base solicitando o retorno de todas as camadas ocultas (necessário para o concat_4)
        self.bert = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.pooling_strategy = CONFIG["arch_pooling_strategy"]
        self.num_labels = num_labels

        # -----------------------------------------------------------------
        # Abordagem 1: Congelamento de Camadas (Layer Freezing)
        # -----------------------------------------------------------------
        num_freezes = CONFIG["arch_freeze_layers"]
        if num_freezes > 0:
            print(f"🔒 [Arquitetura] Congelando os embeddings e as primeiras {num_freezes} camadas do BERT...")
            # Congela os embeddings iniciais
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False
            # Congela as N primeiras camadas do encoder
            for i in range(num_freezes):
                for param in self.bert.encoder.layer[i].parameters():
                    param.requires_grad = False

        # Define o tamanho do vetor que sai do BERT dependendo da estratégia de pooling
        # Se concatenar as 4 últimas camadas, o vetor final terá 4x o tamanho padrão (768 * 4 = 3072)
        input_dim = self.bert.config.hidden_size
        if self.pooling_strategy == "concat_4":
            input_dim = self.bert.config.hidden_size * 4

        # -----------------------------------------------------------------
        # Abordagem 3: Cabeças Densas Intermediárias e Regularização (Dropout)
        # -----------------------------------------------------------------
        hidden_dim = CONFIG["arch_hidden_dimension"]
        self.dropout = torch.nn.Dropout(CONFIG["arch_dropout_rate"])
        
        if hidden_dim > 0:
            # Se configurado, cria uma rede mais profunda: BERT -> Dense -> Ativação -> Dropout -> Saída
            print(f"🧠 [Arquitetura] Criando cabeça profunda com camada oculta de tamanho {hidden_dim}...")
            self.intermediate_dense = torch.nn.Linear(input_dim, hidden_dim)
            self.activation = torch.nn.GELU()
            self.classifier = torch.nn.Linear(hidden_dim, num_labels)
        else:
            # Se configurado 0, vai direto do BERT para a classificação (como no baseline)
            print("📏 [Arquitetura] Criando cabeça linear direta (padrão baseline)...")
            self.intermediate_dense = None
            self.classifier = torch.nn.Linear(input_dim, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        # Passa os dados pelo BERTimbau
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # -----------------------------------------------------------------
        # Abordagem 2: Alteração do Mecanismo de Pooling
        # -----------------------------------------------------------------
        if self.pooling_strategy == "cls":
            # Pega apenas o primeiro token [CLS] da última camada oculta
            pooled_output = outputs.last_hidden_state[:, 0, :]
            
        elif self.pooling_strategy == "mean":
            # Faz a média aritmética de todos os tokens de texto, ignorando os paddings
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            pooled_output = sum_embeddings / sum_mask
            
        elif self.pooling_strategy == "concat_4":
            # Recupera os estados de todas as camadas e concatena o [CLS] das últimas 4
            hidden_states = outputs.hidden_states
            # hidden_states[-1] é a última camada, [-2] a penúltima, etc.
            pooled_output = torch.cat([hidden_states[-i][:, 0, :] for i in range(1, 5)], dim=-1)
        else:
            raise ValueError(f"Estratégia de pooling desconhecida: {self.pooling_strategy}")

        # Aplica regularização inicial
        x = self.dropout(pooled_output)
        
        # Passa pela cabeça de classificação definida no __init__
        if self.intermediate_dense is not None:
            x = self.intermediate_dense(x)
            x = self.activation(x)
            x = self.dropout(x)
            
        logits = self.classifier(x)
        
        # Cálculo da Loss Function para o Hugging Face Trainer coletar automaticamente
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}
    

# GANCHO DO MEMBRO 5: Se você precisar alterar a função de custo (Loss Function)
class CustomLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        if CONFIG["use_custom_loss"]:
            # Exemplo de gancho: Pesos para penalizar erros de inversão ou falsos neutros
            # Ordem do mapa: [Esquerda, Direita, Neutro]
            pesos_classes = torch.tensor([1.0, 1.0, 1.5], device=model.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=pesos_classes)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            
        return (loss, outputs) if return_outputs else loss

# =====================================================================
# 📦 CLASSE DATASET PADRÃO (PYTORCH)
# =====================================================================
class PoliticalBiasDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.texts = dataframe['texto'].astype(str).values
        self.labels = dataframe['vies_politico'].map(LABEL_MAP).values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =====================================================================
# 📉 METRICAS DE AVALIAÇÃO (EIXO CENTRAL DE SUCESSO DO RELATÓRIO)
# =====================================================================
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    
    # Macro F1-Score é a métrica principal de decisão descrita no relatório
    macro_f1 = f1_score(labels, preds, average='macro')
    accuracy = accuracy_score(labels, preds)
    precision, recall, _, _ = precision_recall_fscore_support(labels, preds, average='macro')
    
    return {
        "macro_f1": macro_f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

# =====================================================================
# 🏃‍♂️ PIPELINE DE EXECUÇÃO
# =====================================================================
def main():
    print("🌐 Carregando dados públicos do GitHub...")
    try:
        df_train = pd.read_csv(CONFIG["url_train"])
        df_val = pd.read_csv(CONFIG["url_val"])
    except Exception as e:
        print(f"❌ Erro ao baixar dados. Verifique as URLs no CONFIG. Detalhes: {e}")
        return

    # Gancho Membro 1: Ajuste de tamanho de amostragem para debugging rápida
    if CONFIG["max_training_samples"] is not None:
        df_train = df_train.sample(n=CONFIG["max_training_samples"], random_state=42).reset_index(drop=True)
        print(f"⚠️ Debug Mode ativo: Dataset de treino reduzido para {len(df_train)} instâncias.")

    print(f"📊 Dados carregados: Treino ({len(df_train)} linhas) | Validação ({len(df_val)} linhas)")

    print(f"🔤 Inicializando Tokenizer: {CONFIG['model_name']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])

    # Instanciação dos Datasets estruturados do PyTorch
    train_dataset = PoliticalBiasDataset(df_train, tokenizer, CONFIG["max_sequence_length"])
    val_dataset = PoliticalBiasDataset(df_val, tokenizer, CONFIG["max_sequence_length"])

    print("🤖 Inicializando modelo de linguagem base...")
    if CONFIG["use_custom_architecture"]:
        # Se Membro 4 ativar o gancho de arquitetura customizada
        print("💡 Utilizando Arquitetura Customizada (Gancho Membro 4)")
        model = CustomBERTimbauClassifier(CONFIG["model_name"], num_labels=3)
    else:
        # Padrão nativo do Hugging Face Sequence Classification
        model = AutoModelForSequenceClassification.from_pretrained(CONFIG["model_name"], num_labels=3)

    # Configurações de Treinamento do Hugging Face
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=6,
        per_device_train_batch_size=CONFIG["batch_size"],
        per_device_eval_batch_size=CONFIG["batch_size"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="mlflow" # Força o ecossistema a sincronizar com as variáveis do .env/DagsHub
    )

    # Orquestrador de Treino (Utilizando a classe customizada para o Gancho de Loss do Membro 5)
    trainer = CustomLossTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    print("🚀 Iniciando Fine-Tuning do BERTimbau...")
    trainer.train()

    print("📉 Executando Avaliação Final no conjunto de validação...")
    eval_results = trainer.evaluate()
    print(f"\n🏆 Resultado Final de Validação:")
    print(f"   🔹 Macro F1-Score: {eval_results['eval_macro_f1']:.4f}")
    print(f"   🔹 Acurácia:      {eval_results['eval_accuracy']:.4f}")

    # Encerra o ciclo do MLflow de forma limpa salvando as alterações
    if os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.end_run()
        print("✅ Run sincronizada e encerrada no painel do DagsHub!")

if __name__ == "__main__":
    main()
