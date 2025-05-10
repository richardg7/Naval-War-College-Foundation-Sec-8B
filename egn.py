from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import gc
import psutil
from datetime import datetime
import logging
import warnings

# Configurações seguras para 6GB VRAM
MODEL_PATH = "./Foundation-Sec-8B"
OUTPUT_PATH = "./Escola-de-Guerra-Naval"
MAX_MEMORY = {0: "5GB"}  # Limite de VRAM
SHARD_SIZE = "500MB"     # Tamanho reduzido de shard

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("conversion_optimized.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()

logger = setup_logger()

def memory_safety_check():
    """Verifica se há memória suficiente antes de prosseguir"""
    if torch.cuda.is_available():
        free_vram = torch.cuda.mem_get_info()[0]/1024**3
        if free_vram < 4:  # Menos de 4GB livres
            logger.warning(f"VRAM crítica: {free_vram:.1f}GB livres")
            return False
    return True

def load_model_safely():
    """Carrega o modelo com proteção contra OOM"""
    try:
        # Configuração ultra-otimizada para 6GB VRAM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,  # Já carrega em float16
            device_map="auto",
            max_memory=MAX_MEMORY,
            low_cpu_mem_usage=True,
            offload_state_dict=True,
            offload_folder="./offload_temp"
        )
        # Força conversão para float16 (redundante, mas seguro)
        model = model.half()
        # Limpa cache imediatamente após o carregamento
        torch.cuda.empty_cache()
        logger.info("Modelo convertido para float16 e cache da GPU liberado.")
        return model
    except Exception as e:
        logger.error(f"Falha no carregamento: {str(e)}")
        return None

def convert_model():
    if not memory_safety_check():
        logger.error("Memória insuficiente para iniciar a conversão")
        return False

    logger.info("Iniciando conversão com otimização para 6GB VRAM")
    
    # Passo 1: Carregamento progressivo
    for attempt in range(3):  # 3 tentativas
        logger.info(f"Tentativa {attempt+1}/3")
        model = load_model_safely()
        if model: break
        torch.cuda.empty_cache()
        gc.collect()
    else:
        logger.error("Falha após 3 tentativas")
        return False

    # Passo 2: Carregar tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # Passo 3: Salvamento seguro
    try:
        model.save_pretrained(
            OUTPUT_PATH,
            max_shard_size=SHARD_SIZE,
            safe_serialization=True
        )
        tokenizer.save_pretrained(OUTPUT_PATH)
        logger.info("Conversão concluída com sucesso!")
        return True
    except Exception as e:
        logger.error(f"Falha ao salvar: {str(e)}")
        return False
    finally:
        # Limpeza final garantida
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    if convert_model():
        exit(0)
    else:
        logger.error("""
        SOLUÇÕES RECOMENDADAS:
        1. Reduza SHARD_SIZE para '250MB'
        2. Execute em uma GPU com mais VRAM
        3. Use --max_shard_size 250MB no script original
        4. Experimente quantização 8-bit com: pip install bitsandbytes
        """)
        exit(1)