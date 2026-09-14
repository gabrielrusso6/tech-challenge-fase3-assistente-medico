# Preparação, piloto e treinamento

## Decisão de implementação
Treinamento LoRA com Transformers + PEFT; modelo inicial Qwen/Qwen2.5-1.5B-Instruct.
Usa float32, batch 1, gradient checkpointing, LoRA em q_proj/v_proj, rank 8 e seed 42.
A escolha prioriza um caminho único de adaptador para Mac/MPS, Windows/CPU e Linux.
O tempo e a memória no M4 24 GB ainda precisam do piloto; não há garantia antecipada.
Não utiliza MLX nem conversão de adaptador para Ollama. O modo local integra o adaptador
PEFT diretamente ao LangChain. Os modelos ajustados não são baixados automaticamente.

## Windows agora: não exige pesos nem GPU
Na raiz do repositório:

~~~powershell
git pull --ff-only
.\.venv\Scripts\python.exe -m pytest -v
.\.venv\Scripts\python.exe -m tc3.corpus
.\.venv\Scripts\python.exe -m tc3.train --dry-run
~~~

Saída do corpus esperada: 166 exemplos / 83 grupos; 98 treino, 34 validação, 34 teste.
O dry-run valida integridade, separação e formato. Não tokeniza, não baixa o modelo,
não treina e não mede desempenho. A suíte leve pode mostrar um módulo de treino ignorado:
o job separado training-smoke instala os extras e executa o teste real de LoRA.

## Mac à noite: instalação
Use Python 3.12. Na raiz do checkout:

~~~bash
git pull --ff-only
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-training.txt -r requirements-dev.txt
python -m pip install -e . --no-deps
python -m pip check
python -m tc3.corpus
python -m tc3.train --dry-run
~~~

## Piloto curto obrigatório
Baixa o modelo base na primeira execução. Precisa de internet e espaço para cache/pesos.

~~~bash
python -m tc3.train --execute --device mps --steps 2 --output models/pilot
~~~

O piloto tokeniza todos os exemplos sem truncamento silencioso, faz duas atualizações e
mede loss de validação antes/depois. Ele salva adaptador + tokenizer + configuração e logs.
Duas etapas NÃO são o treinamento final. Se exceder memória, não execute novamente sem
avaliar a mensagem: podemos diminuir modelo ou comprimento, preservando o contrato de dados.
Não sobrescreva a saída: o script bloqueia diretórios não vazios.
Não há retomada de optimizer; um novo --output representa outro experimento.

## Treinamento inicial, após o piloto funcionar
~~~bash
python -m tc3.train --execute --device mps --steps 200 --output models/tc3-lora
~~~

- O conjunto de teste não participa da otimização. A validação inteira é usada para loss.
- O modelo salvo é o último passo, não seleção automática do melhor checkpoint.
- O loss ignora o prompt e considera apenas a resposta. Padding não é necessário em batch 1.
- Salva run.json (versões, revisão, hardware, contagens, hashes, duração), training.jsonl,
  adapter_config.json, adapter_model.safetensors e tokenizer.
- No MPS registra memória alocada ao final, não pico; não rotular esse valor como pico.
- Modelo base deve continuar disponível. Adaptador sozinho não executa.
- Preserve run.json com os pesos; o backend recusa hashes divergentes ou treino incompleto.
- O corpus sintético ainda é limitado. Melhorar loss não prova competência médica.

## Avaliação antes/depois em casos reservados
Depois de ajustar configurações usando validação, execute o teste final apenas para relatar.
Use a mesma revisão do modelo nos dois lados (registrada em models/tc3-lora/run.json).

~~~bash
python -m tc3.evaluate_llm --device mps --model Qwen/Qwen2.5-1.5B-Instruct --revision REVISAO_DO_RUN_JSON --output runtime/evaluation/base.jsonl
python -m tc3.evaluate_llm --device mps --model Qwen/Qwen2.5-1.5B-Instruct --adapter models/tc3-lora --output runtime/evaluation/tuned.jsonl
python -m tc3.compare runtime/evaluation/base.summary.json runtime/evaluation/tuned.summary.json
~~~

Substitua REVISAO_DO_RUN_JSON pelo hash model_revision real. As chamadas geram respostas,
métricas de schema/fontes, tempos e erros reais. Repetir o teste para escolher prompts
contamina a avaliação; use --split validation durante ajustes.
A comparação recusa condições incompatíveis e registra deltas negativos sem escondê-los.
Não conta exemplos bloqueados como respostas corretas.

## Assistente com o adaptador
~~~bash
tc3 --mode local --model Qwen/Qwen2.5-1.5B-Instruct --adapter models/tc3-lora --device mps --patient SYN-001
~~~

Para o avaliador em outro sistema, --device cpu usa o mesmo formato de adaptador.
Isso é caminho de código, ainda precisa ser testado com os pesos treinados. O teste pequeno
de recarregamento no CI não comprova latência nem qualidade do modelo de 1.5B.

## Distribuição para entrega
Pesos em models/ estão ignorados no Git para evitar commits grandes. Após o treino, publicar
o adaptador com run.json e licença/atribuição em um release ou repositório de modelos,
registrar URL e SHA256 no README e verificar download em ambiente limpo. Nada foi publicado
como modelo treinado nesta etapa.

## Fontes técnicas consultadas
- [Modelo base e licença](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- [PEFT: configuração, treinamento e salvamento de LoRA](https://huggingface.co/docs/peft/en/quicktour)
- [Transformers 4.56.2](https://pypi.org/project/transformers/4.56.2/)
- [PEFT 0.17.1](https://pypi.org/project/peft/0.17.1/)
- [PyTorch 2.8.0](https://pypi.org/project/torch/2.8.0/)
