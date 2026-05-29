import os
import mlflow
from dotenv import load_dotenv

# Carrega as variáveis do arquivo .env
load_dotenv()

# Inicializa as credenciais no ambiente do MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Define o nome do experimento
mlflow.set_experiment("teste-conexao-dagshub")

print("🔄 Tentando se conectar ao DagsHub...")

with mlflow.start_run():
    # Loga um parâmetro e uma métrica de teste
    mlflow.log_param("modelo_base", "bertimbau-base")
    mlflow.log_metric("acuracia_teste", 0.95)
    print("✅ Run enviada com sucesso! Verifique o painel do DagsHub.")
    