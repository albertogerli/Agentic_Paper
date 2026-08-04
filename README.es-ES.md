

<div align="center">

# 🧑‍🔬 Agentic_Paper

**Un orquestador LLM multiagente para la revisión por pares académica.**

[![PyPI version](https://img.shields.io/pypi/v/agentic-paper.svg)](https://pypi.org/project/agentic-paper/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/albertogerli/Agentic_Paper/actions/workflows/ci.yml/badge.svg)](https://github.com/albertogerli/Agentic_Paper/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-46aef7.svg)](https://github.com/astral-sh/ruff)

*Construido para estudiantes, doctorandos e investigadores que buscan una segunda opinión transparente y reproducible sobre un manuscrito: no otro chatbot opaco.*

</div>

---

## ¿Por qué Agentic_Paper?

**No es otro envoltorio (wrapper) de ChatGPT.**

Un solo LLM, al recibir un artículo y el prompt *"por favor revisa esto"*, te da el promedio de internet. `Agentic_Paper` hace algo genuinamente diferente:

- 🧠 **12 agentes revisores especializados** se ejecutan **en paralelo**, cada uno con su propio rol, prompt y complejidad base — Metodología, Resultados, Literatura, Estructura, Impacto, Contradicción, Ética, Origen con IA, Alucinación, Validador de Citas, Validador Statcheck, Evaluador de Revisiones.
- 🧑‍⚖️ Un **Coordinador** sintetiza sus veredictos estructurados, identifica desacuerdos y prioriza las revisiones.
- ✉️ Un **Editor** + un agente de **Resumen para Autor/Editor** generan una carta de decisión al estilo de revista académica y la nota confidencial para el editor, por separado.
- 📜 **Cada llamada al LLM es auditada** — conteo de tokens, latencia, estimación de costo, hash del prompt, bandera de modo de pensamiento, semilla — todo se escribe en `audit.jsonl` para que puedas demostrar qué se preguntó y qué se respondió. Ninguna alucinación se esconde en la oscuridad.
- 🔎 **Las citas se validan contra [OpenAlex](https://openalex.org)** (~250M de registros académicos abiertos, sin necesidad de clave API). Las referencias falsas se marcan automáticamente.
- 🧮 **Los valores p reportados se recomputan** mediante el paquete R `statcheck` — si un artículo dice `t(28) = 2.3, p = .01` y las matemáticas dicen `p ≈ 0.029`, lo verás.
- 🔌 **MultiProveedor, enchufable**: OpenAI, Anthropic Claude, Google Gemini, **y cualquier punto final local compatible con OpenAI** — ver [§ Modelos Locales y Gratuitos](#-modelos-locales-y-gratuitos-con-ollama).
- 🎛️ **Todo tipado**: los revisores no devuelven prosa libre, devuelven modelos `pydantic` validados. Los agentes posteriores consumen estructura, no subcadenas.

Salidas: un informe en Markdown, un panel HTML independiente, un JSON estructurado y una carpeta con alcance de `run_id` que puedes entregar cuando una revista pregunte *"¿cómo se produjo esta evaluación?"*.

---

## Instalación

```bash
pip install agentic-paper
```

Eso es todo. Python puro; funciona en macOS, Linux y Windows con **Python 3.10+**.

Para la interfaz web opcional (demo en vivo FastAPI + HTMX):

```bash
pip install "agentic-paper[web]"
```

Para verificaciones estadísticas de sensatez (recomendado para artículos empíricos), también instala [R](https://www.r-project.org/) y los paquetes `statcheck` + `jsonlite`:

```r
install.packages(c("statcheck", "jsonlite"))
```

Si R no está disponible, el resto de la tubería (pipeline) sigue ejecutándose: el Validador Statcheck simplemente reporta *"no disponible"* en el informe final.

---

## Inicio Rápido

### 1. Configurar una clave de proveedor

```bash
export OPENAI_API_KEY="sk-..."
# Opcional, para enrutamiento multi-proveedor:
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_API_KEY="..."
```

> 💡 **¿Sin presupuesto?** Omite este paso y salta a [Modelos Locales y Gratuitos](#-modelos-locales-y-gratuitos-con-ollama).

### 2. Revisar un artículo desde la terminal

```bash
agentic-paper paper.pdf --seed 42
```

Las salidas se guardan en `output_paper_review/<run_id>/` — abre `dashboard_*.html` para un informe estilizado, o lee `review_report_*.md` directamente.

### 3. O usar la interfaz web

```bash
agentic-paper-web --port 8000
# → http://127.0.0.1:8000/
```

Una página limpia de zona de descarga: arrastra un PDF, observa a los 12 agentes pensar en vivo (flujo `thinking_delta` real cuando el proveedor lo soporte), luego lee el informe en línea. Formulario opcional de **Trae-Tu-Propia-Clave (BYOK)** para compartir la demo con colegas sin exponer tu cuenta: las claves se mantienen en el marco de pila del worker, nunca se registran en logs, nunca se escriben en disco.

```
┌─────────────────────────────────────────────┐
│  drop a PDF here  →  watch the agents work  │
│  ⠋ methodology   reading…                   │
│  ✓ results       done (4.2 s, $0.018)       │
│  ⠴ literature    thinking…                  │
│  …                                          │
└─────────────────────────────────────────────┘
```

### ⚡ Modo Automático: nunca fallar por falta de una clave

Los perfiles de enrutamiento de la interfaz web (`max` / `std` / `quick`) distribuyen deliberadamente los agentes entre **múltiples proveedores** para aprovechar las fortalezas de cada modelo: por ejemplo, `std` envía razonamiento de alto nivel a Claude, nivel estándar a GPT, nivel básico a Gemini. Si solo pegas **una** clave API en el formulario BYOK, un enrutamiento ingenuo daría 404 en los otros dos proveedores y arruinaría la ejecución.

**El Modo Automático soluciona esto de forma transparente.** Cuando se envía el formulario BYOK:

1. Cada nivel se verifica contra las claves que realmente proporcionaste.
2. Los niveles que apuntan a un proveedor no disponible se remapean a un modelo equivalente en un proveedor que *sí* tienes (p. ej. `tier_high: anthropic/claude-opus-4-7` → `google/gemini-3-pro`).
3. Se preserva `thinking_budget` y la intensidad del rol del nivel: el Modo Automático elige el modelo insignia de razonamiento del proveedor de respaldo para `tier_high`, el de gama media para `tier_standard`, y el más económico para `tier_basic`.
4. Un **banner amarillo** en la parte superior de la página de ejecución lista cada remapeo con el original vs. nuevo (proveedor, modelo) para que sepas exactamente qué cambió.

La ejecución continúa de extremo a extremo con una sola clave, sin ediciones manuales de configuración. El Modo Automático solo se activa cuando se proporciona al menos una clave BYOK: las ejecuciones que usan la configuración del servidor se dejan intactas.

---

## 🦙 Modelos Locales y Gratuitos (con Ollama)

**No necesitas una tarjeta de crédito para usar Agentic_Paper.** El `ProviderRegistry` acepta cualquier punto final compatible con OpenAI, lo que significa que puedes ejecutar toda la tubería contra [Ollama](https://ollama.com), [LM Studio](https://lmstudio.ai), [vLLM](https://github.com/vllm-project/vllm) o cualquier servidor local que controles. **Revisión por pares gratuita, totalmente privada, todo en tu portátil.**

### Paso a paso: Ollama + Llama 3

```bash
# 1. Install Ollama from https://ollama.com (one-line installer)
# 2. Pull a model — Llama 3.1 8B fits on a laptop with 16 GB RAM
ollama pull llama3.1

# 3. Start Ollama in the background (it auto-serves an OpenAI-compatible API on :11434)
ollama serve &

# 4. Point Agentic_Paper at it — two env vars is all it takes
export OPENAI_API_KEY="ollama"                          # any non-empty string
export OPENAI_API_BASE="http://localhost:11434/v1"      # Ollama's OpenAI-compat endpoint

# 5. Run the review using your local model
agentic-paper paper.pdf --config config.local.yaml
```

Configuración mínima `config.local.yaml` para conectar cada nivel al modelo local:

```yaml
output_dir: output_paper_review
routing:
  tier_high:     { provider: openai, model: llama3.1 }
  tier_standard: { provider: openai, model: llama3.1 }
  tier_basic:    { provider: openai, model: llama3.1 }
providers:
  openai:
    api_key_env: OPENAI_API_KEY
    base_url: http://localhost:11434/v1
```

### Niveles de modelos locales recomendados

| Hardware | Modelo sugerido | Notas |
|---|---|---|
| Portátil, 16 GB RAM | `llama3.1` (8B) | Línea base sólida. Las revisiones son más lentas pero coherentes. |
| Estación de trabajo, 32 GB+ | `llama3.1:70b` o `qwen2.5:32b` | Más cerca de la calidad de GPT-4o en razonamiento. |
| Caja con GPU, 24 GB+ VRAM | `deepseek-r1` vía vLLM | Excelente para los revisores de Metodología / Contradicción. |
| Mac Studio (M2 Ultra+) | `llama3.1:70b` MLX | Nativo para silicona Apple; más rápido que CUDA con memoria comparable. |

### Advertencias con modelos locales

- **Salidas estructuradas**: los modelos de pesos abiertos pequeños ocasionalmente violan el esquema JSON. Agentic_Paper reintenta con `tenacity` y recurre a `response_format: json_object`. Los modelos más grandes (≥ 30B) son notablemente más fiables.
- **Calidad**: un modelo local de 7-8B no igualará a Claude Opus 4.7, pero para una primera pasada en un borrador (detectar contradicciones, citas faltantes, problemas estructurales), es más que suficiente.
- **Privacidad**: nada sale de tu máquina. Perfecto para manuscritos no publicados bajo embargo o NDA.
- **Costo**: literalmente cero (salvo electricidad).

### Enrutamiento mixto: local gratuito + nivel superior de pago

También puedes mantener los agentes económicos de forma local y enrutar solo el razonamiento intensivo a un proveedor de pago:

```yaml
routing:
  tier_high:     { provider: anthropic, model: claude-opus-4-7, thinking_budget: auto }
  tier_standard: { provider: openai,    model: gpt-5.4-mini }
  tier_basic:    { provider: ollama_local, model: llama3.1 }
providers:
  ollama_local:
    api_key_env: OPENAI_API_KEY
    base_url: http://localhost:11434/v1
```

El marco de trabajo trata cualquier nombre de proveedor personalizado con una `base_url` como compatible con OpenAI.

---

## Arquitectura (en 30 segundos)

```
        PDF ──▶ PaperExtractor ──▶ paper.txt + complexity score
                                          │
                                          ▼
                              ┌────────────────────────┐
                              │ ConcurrentAgentRunner  │
                              │   (asyncio.gather)     │
                              └──────────┬─────────────┘
                                         │ 12 reviewers in parallel
                                         ▼
                              Coordinator ─▶ Author/Editor Summary
                                         │
                                         ▼
                                      Editor
                                         │
                                         ▼
                          Markdown · JSON · HTML · audit.jsonl
                          (all under output/<run_id>/)
```

La base de código es deliberadamente pequeña y modular:

- **`orchestrator.py`** — coordina la tubería; no sabe sobre concurrencia.
- **`agent_runner.py`** — `ConcurrentAgentRunner` posee el mecanismo `asyncio`. Intercambiable por Celery / Ray / Dask sin tocar el orquestador.
- **`storage.py`** — ABC `StorageProvider` + `LocalFileStorage`. Implementa `S3Storage` o `PostgresStorage` una vez; todo lo demás sigue funcionando.
- **`providers/`** — un módulo por proveedor (`OpenAI`, `Anthropic`, `Google`, compat-OpenAI). Cada uno implementa una interfaz `LLMProvider` uniforme.
- **`agents/`** — un archivo por rol. Cada uno define `KEY`, `NAME`, `INSTRUCTIONS`, `SCHEMA`, `base_complexity`. Añadir un 13º revisor es un archivo de 30 líneas.
- **`schemas.py`** — modelos `pydantic`. Cada llamada al LLM devuelve una instancia validada, no una cadena analizada.
- **`external/`** — OpenAlex (citas), statcheck (subproceso R).

Si solo lees un archivo para entender el proyecto, lee [`agentic_paper/orchestrator.py`](agentic_paper/orchestrator.py). Tiene ~570 líneas y se lee como el índice de este README.

---

## ¿Qué hay en el directorio de ejecución?

Después de que `agentic-paper paper.pdf` finalice, `output_paper_review/<run_id>/` contiene:

```
audit.jsonl              ← una fila JSON por llamada al LLM (12 campos)
paper.txt                ← texto extraído (conservado para agentes con reintento fallido)
paper_info.json          ← título / autores / resumen / secciones detectadas
review_<agent>.txt       ← veredicto validado y estructurado de cada revisor
review_report_*.md       ← el informe legible por humanos
review_results_*.json    ← paquete legible por máquina (incl. enrutamiento + resumen de auditoría)
executive_summary_*.md   ← resumen ejecutivo de una página (TL;DR)
dashboard_*.html         ← informe estilizado independiente (no requiere servidor)
prompts/<agent>.txt      ← prompt exacto enviado — prompt completo + volcado de contexto
responses/<agent>.json   ← carga útil de respuesta sin procesar del proveedor
paper_review_system.log  ← registro de depuración de toda la ejecución
```

Este es el paquete de reproducibilidad. Entregalo cuando una revista pregunte *"¿cómo se produjo esta evaluación?"* y la respuesta es *un solo archivo tar*.

---

## Reproducibilidad y determinismo

```bash
agentic-paper paper.pdf --seed 42
```

La semilla se reenvía a cada proveedor que lo soporte:
- **OpenAI** — `seed=N` en Responses + Chat Completions.
- **Google Gemini** — `GenerateContentConfig.seed=N`.
- **Anthropic** — se registra en la auditoría pero no se propaga (la API Messages aún no expone una semilla); combínala con `temperature: 0` para máxima estabilidad.

El costo, la latencia y el conteo de tokens de cada llamada son consultables desde `audit.jsonl` con un solo comando `jq`: no se requiere un stack de observabilidad separado.

---

## Limitaciones (sinceras)

Cosas que `Agentic_Paper` **no** hace:

- **Sustituir la revisión por pares humana.** Detecta problemas mecánicos — inconsistencias internas, vacíos en citas, informes estadísticos incorrectos — más rápido que un revisor humano cansado. No tiene gusto, profundidad de dominio en *tu* nicho, ni conocimiento de las normas específicas de cada revista.
- **Inspeccionar figuras, tablas o ecuaciones renderizadas como imágenes.** Solo se analiza el texto (pdfplumber + heurísticas).
- **Verificar hechos más allá de las citas.** Sin anclaje a PubMed / arXiv / Semantic Scholar: solo resolución en OpenAlex de referencias explícitas.
- **Síntesis multi-artículo.** Un artículo por ejecución; usa un bucle de shell para lotes.
- **Traducir.** Los artículos en idiomas no ingleses funcionan técnicamente, pero los prompts de los revisores asumen un registro de revisión por pares en inglés.

---

## Desarrollo

```bash
git clone https://github.com/albertogerli/Agentic_Paper.git
cd Agentic_Paper
pip install -e ".[dev,web]"
pytest -q --cov=agentic_paper --cov-fail-under=60
```

224 pruebas, ~74 % de cobertura de línea, CI en Python 3.10 / 3.11 / 3.12.

PRs bienvenidos — especialmente: nuevas recetas para modelos locales, nuevos roles de revisor, implementaciones de `StorageProvider` para S3/Postgres, paquetes de prompts en idiomas no ingleses.

---

## Cómo citar

Si `Agentic_Paper` contribuye a la producción de investigación, por favor cita:

```bibtex
@software{gerli_agentic_paper_2026,
  author    = {Gerli, Alberto G.},
  title     = {Agentic\_Paper: A multi-agent, multi-provider, structured-output
               peer-review pipeline for scientific manuscripts},
  year      = {2026},
  url       = {https://github.com/albertogerli/Agentic_Paper},
  version   = {2.0.0}
}
```

---

## Licencia

[MIT](LICENSE). Úsalo, bifúrcalo, distribúyelo.

## Contacto

- **Issues / PRs**: <https://github.com/albertogerli/Agentic_Paper/issues>
- **Email**: <alberto@albertogerli.it>
- **Taller**: Physalia 2026 — *Flujos de trabajo agentivos para la revisión científica*
