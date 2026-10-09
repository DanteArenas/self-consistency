# Self-consistency: prueba de concepto en una RTX 2060

Prueba basada en *Optimal Self-Consistency for Efficient Reasoning with Large
Language Models*, Feng et al., arXiv:2511.12309v2.
[Paper original](2511.12309v2.pdf).

El objetivo es generar varias respuestas por pregunta y elegir la respuesta final
más frecuente mediante self-consistency clásica, sin Blend-ASC ni asignación adaptativa.

## Estado actual

[El ejemplo](src/self_consistency/basic_example.py) usa vLLM con
`Qwen/Qwen2.5-Math-1.5B-Instruct` en una RTX 2060 de 6 GB, en FP16 sin
cuantización. Se verificaron respuestas completas y correctas para tres preguntas
en inglés. Esto valida inferencia básica, no exactitud sobre un benchmark.

El modelo de 1.5B es una adaptación local: el paper usa Qwen2.5-MATH-7B.
PyTorch y vLLM son herramientas compartidas con el paper. La integración con
Language Model Evaluation Harness está pendiente.

Self-consistency aún no está implementada: `src/self_consistency/self_consistency.py`
es un archivo vacío reservado para el siguiente paso.

## Ejecución

Desde la raíz, activa el entorno que contiene vLLM. Si lo recreaste como `.venv`:

```bash
source .venv/bin/activate
python src/self_consistency/basic_example.py
```

Si conservas el entorno anterior, activa `.venv-vllm/bin/activate` en su lugar.
Consulta [el registro del entorno](docs/ENTORNO_LOCAL.md) para detalles de
instalación y hardware. Todavía no hay un archivo de dependencias fijadas.

## Parámetros y salidas

- Contexto máximo: 1024 tokens entre entrada y salida.
- Respuesta máxima: 256 tokens; temperatura 0.8 y top-p 0.95.
- Presupuesto de memoria de GPU: 0.75 de la memoria total.
- Una secuencia activa, lote de hasta 1024 tokens y ejecución eager para reducir memoria.

El script imprime las respuestas y guarda `basic_example_output.txt` y
`basic_example_output.md` junto al script, sobrescribiéndolos en cada ejecución.
Estas salidas locales se excluyen de Git. El Markdown conserva LaTeX y puede
previsualizarse con Markdown Preview Enhanced.

## Siguiente paso

Generar varias muestras por pregunta, extraer y normalizar la respuesta final y
aplicar una votación simple. El ejemplo actual pasa texto a `LLM.generate`;
adaptar la plantilla de chat del modelo Instruct es una mejora pendiente antes
de evaluar su comportamiento de forma sistemática.
