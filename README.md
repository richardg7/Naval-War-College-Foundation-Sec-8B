# Naval-War-College-Foundation-Sec-8B

# Conversão de Modelo LLaMA (Meta) da Hugging Face para Ollama

Este repositório contém um script Python para conversão otimizada de modelos LLaMA (Meta) hospedados na Hugging Face para uso local com suporte a ambientes de GPU limitados e compatibilidade com o Ollama.

## 📌 Visão Geral

O script `egn.py` permite:

- Carregamento seguro do modelo Hugging Face com suporte a `float16`
- Otimização para placas com 6GB de VRAM (ex: RTX 2060)
- Salvamento em shards (`max_shard_size`) compatíveis com modelos grandes
- Conversão com segurança de memória e offload automático
- Geração de estrutura final compatível para uso com o [Ollama](https://ollama.com)

## 🧰 Requisitos

- Python 3.10 ou superior
- GPU com suporte CUDA (mínimo recomendado: 6GB VRAM)
- Dependências:
  ```bash
  pip install torch transformers psutil
  ```
  ou
  ```bash
    pip install transformers
  ```

> 💡 Sugestão: `pip install bitsandbytes` para quantização alternativa em 8-bit.

## 🔧 Como Utilizar

1. Coloque seu modelo original (Hugging Face) no caminho especificado por `MODEL_PATH`.
2. Execute o script:

   ```bash
   python egn.py
   ```

3. O modelo convertido será salvo na pasta definida como `OUTPUT_PATH`.

4. Para uso com o Ollama, compacte a pasta de saída:

   ```bash
   tar -czvf Escola-de-Guerra-Naval.tar.gz Escola-de-Guerra-Naval
   ```

## ⚙️ Configurações Principais

- `MODEL_PATH`: Caminho para o modelo original Hugging Face (ex: `./Foundation-Sec-8B`)
- `OUTPUT_PATH`: Caminho de saída para o modelo convertido
- `SHARD_SIZE`: Tamanho máximo de cada shard (`500MB` por padrão)
- `MAX_MEMORY`: Alocação máxima por GPU (`5GB` por padrão)

## 🚨 Solução de Problemas

Caso ocorra erro de memória insuficiente:

- Reduza `SHARD_SIZE` para `'250MB'`
- Aumente a VRAM disponível ou utilize `bitsandbytes` com quantização 8-bit
- Utilize offload de parâmetros com `offload_folder` configurado
- Libere a GPU com `torch.cuda.empty_cache()` antes da execução

---

## 📜 Licença

```text
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
```

---

## ✍️ Autor

[Richard Guedes](https://www.linkedin.com/in/richard-guedes/)
Este script foi desenvolvido com foco em desempenho, compatibilidade e acessibilidade para pesquisadores e profissionais que desejam experimentar modelos LLaMA localmente com recursos limitados.
