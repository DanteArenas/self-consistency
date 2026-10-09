# Registro de preparación local

Fecha: 8 de octubre de 2026. Registro basado en las salidas de terminal
compartidas por el usuario.

## Entorno de la prueba de concepto

- Entorno virtual: `.venv-vllm`, Python 3.12.15.
- Motor: vLLM 0.31.0, ejecutado en WSL.
- GPU: NVIDIA GeForce RTX 2060, 6144 MiB de VRAM.
- Driver reportado: 591.74; compatibilidad CUDA reportada por el driver: 13.1.
- Modelo descargado: `Qwen/Qwen2.5-Math-1.5B-Instruct`.

La descarga y carga de los pesos terminaron correctamente. El registro de vLLM
reportó aproximadamente 2.97 GiB de memoria de GPU para la carga del modelo.
Se usó FP16, sin cuantización. Este modelo de 1.5B es una adaptación para la
prueba local; el paper usa Qwen2.5-MATH-7B.

El ejemplo está en `src/self_consistency/basic_example.py`, con
`gpu_memory_utilization=0.75` y `max_model_len=1024`. El primer intento con
utilización 0.85 falló porque solicitaba 5.1 GiB y había 4.99 GiB libres.

## Herramientas de compilación

Tras cargar los pesos, el motor falló con `Failed to find C compiler` durante
la inicialización de Triton/PyTorch Inductor. Para instalar las herramientas de
compilación se indicó y se introdujo en la terminal:

```bash
sudo apt update && sudo apt install build-essential
```

`build-essential` instala herramientas de compilación del sistema, como GCC,
G++ y Make. Se instala en el sistema Linux de WSL, fuera del entorno virtual.
La salida inicial terminaba en ese comando. La ejecución posterior confirmó
que el compilador estaba disponible y la compilación pudo completarse.

Comprobación posterior prevista:

```bash
gcc --version
python basic_example.py
```

## Primera inferencia completada

El registro posterior del 8 de octubre de 2026 confirma que la compilación y la
inicialización terminaron y se generaron salidas para las dos preguntas. El fallo
por ausencia de compilador quedó resuelto. El motor usó FP16 y el backend
TRITON_ATTN; las advertencias sobre BF16 y FlashAttention no impidieron ejecutar.

Las salidas quedaron incompletas con el límite predeterminado de 16 tokens de
`SamplingParams`. Se modificó el ejemplo para permitir `max_tokens=256`.
La ejecución posterior produjo respuestas completas con ese límite.

## Estado al 9 de octubre de 2026

Una ejecución falló al iniciar con `No available memory for the cache blocks`:
el presupuesto calculado para la caché KV era negativo. Se mantuvieron el
presupuesto 0.75 y contexto 1024, y se agregaron `max_num_seqs=1`,
`max_num_batched_tokens=1024` y `enforce_eager=True` para reducir memoria.

Se revisaron las salidas de las tres preguntas en inglés: 9 manzanas, x = 5 y
8 dólares de cambio, todas completas y correctas. El ejemplo guarda texto y
Markdown junto al script, sobrescribiéndolos en cada ejecución. Estas salidas
locales se excluyen de Git.

Se proporcionaron instrucciones para recrear el entorno como `.venv` con
Python 3.12. No se ha confirmado aquí qué nombre se conserva actualmente ni
se ha creado un archivo de dependencias fijadas.

Se ha validado inferencia básica en tres preguntas, no self-consistency ni
exactitud sobre un dataset. El ejemplo usa texto con `LLM.generate`; todavía
no aplica explícitamente la plantilla de chat del modelo Instruct.
