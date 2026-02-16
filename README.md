## Estructura de Módulos

```
clinical_safeguard/
├── core/
│   ├── __init__.py
│   ├── pipeline.py          # SafeguardPipeline — orquestador
│   ├── base.py              # GuardrailStage ABC
│   └── exceptions.py        # Jerarquía de excepciones internas
├── stages/
│   ├── __init__.py
│   ├── deterministic.py     # DeterministicStage
│   └── semantic.py          # SemanticBERTStage
├── models/
│   ├── __init__.py
│   ├── request.py           # PromptInput
│   ├── response.py          # StageResult, FinalResponse
│   └── enums.py             # Label, ResponseCode
├── config/
│   ├── __init__.py
│   └── settings.py          # Settings (Pydantic BaseSettings)
├── resources/
│   ├── keywords_crisis.yaml
│   ├── keywords_malign.yaml
│   └── bypass_patterns.yaml
└── api/
    ├── __init__.py
    ├── app.py               # FastAPI app factory
    ├── router.py            # /v1/evaluate, /health
    └── middleware.py        # Logging, correlation-id
```

## Diagrama de Clases

```
                        ┌──────────────────────────┐
                        │    GuardrailStage (ABC)  │
                        │──────────────────────────│
                        │ + name: str              │
                        │ + enabled: bool          │
                        │──────────────────────────│
                        │ + process(PromptInput)   │
                        │   -> StageResult  (abs)  │
                        └────────────┬─────────────┘
                                     │ inherits
                    ┌────────────────┴───────────────┐
                    │                                │
       ┌────────────┴──────────┐       ┌─────────────┴───────────┐
       │  DeterministicStage   │       │   SemanticBERTStage     │
       │───────────────────────│       │─────────────────────────│
       │ - _processor:         │       │ - _classifier: Pipeline │
       │     KeywordProcessor  │       │ - _model_id: str        │
       │ - _bypass_patterns:   │       │ - _threshold: float     │
       │     list[re.Pattern]  │       │ - _timeout: int         │
       │───────────────────────│       │─────────────────────────│
       │ + process() -> Result │       │ + process() -> Result   │
       │ - _load_keywords()    │       │ - _load_model()         │
       │ - _check_bypass()     │       │ - _run_inference()      │
       └───────────────────────┘       └─────────────────────────┘

                        ┌──────────────────────────┐
                        │    SafeguardPipeline     │
                        │──────────────────────────│
                        │ - _stages: list[         │
                        │     GuardrailStage]      │
                        │──────────────────────────│
                        │ + evaluate(PromptInput)  │
                        │   -> FinalResponse       │
                        │ - _run_stages()          │
                        │ - _merge_results()       │
                        │ - _build_response()      │
                        └──────────────────────────┘
```

### Regla de Precedencia de Etiquetas (Short-Circuit)

Este es el contrato de comportamiento del pipeline, no solo de datos:

* REGLA 1 — Short-circuit inmediato:
  Si cualquier stage devuelve label=Crisis o label=Maligna
  con short_circuit=True → pipeline se detiene, no ejecuta stages restantes.

* REGLA 2 — Precedencia en merge (si ambos stages completan):
  Crisis > Maligna > Server Error > Válida
* REGLA 3 — Fail-closed absoluto:
  Cualquier excepción no manejada en evaluate() →
  FinalResponse(code=500, etiqueta="Server Error", ...)
  El texto original NUNCA se incluye en el response de error.

## Resumen de la Fase 1

`36 tests, 100% coverage, 0 fallos.`

### Entregables de esta fase:

- Infraestructura core:
    - `GuardrailStage ABC`,
    - Jerarquía de Excepciones,
    - `SafeguardPipeline` con short-circuit,
    - merge por precedencia, y fail-closed handler que jamás propaga excepciones al `caller`.
- Contratos inmutables:
    - PromptInput frozen,
    - StageResult frozen,
    - FinalResponse completamente tipado.
    - Ninguna etapa puede mutar el input.
- Configuración por entorno:
    - Settings con `get_settings()` cacheado como singleton,
    - overrideable en tests con `monkeypatch + cache_clear()`.

Una decisión de diseño que vale la pena mencionar: el `_fail_closed_response()` devuelve `texto_procesado=""`
deliberadamente.
El prompt original solo viaja en la respuesta cuando el sistema procesó exitosamente — un fallo de integridad no debe
filtrar el input hacia el caller.

## Resumen de la Fase 2

`68/68 tests, 100% coverage.`

### Entregables de esta fase:

- DeterministicStage — FlashText (Aho-Corasick O(n)) para keywords, regex compilado en startup para bypass patterns.
- Jerarquía de evaluación: bypass → crisis → malign → válido.
- 3 archivos YAML referenciados con fuentes primarias en cada sección:
    1. keywords_crisis.yaml — 7 categorías mapeadas a los 10 niveles del C-SSRS (Posner et al., 2011; FDA gold standard
       2012)
    2. keywords_malign.yaml — daño a terceros + misuse clínico + facilitación ilegal
    3. bypass_patterns.yaml — 12 patrones regex con IDs estables, basados en OWASP LLM01, Perez (2022), Greshake (2023),
       Shen (2023)

## Resumen de la Fase 3

`103/103 passed. Coverage 100% en todos los módulos.`

### Entregables de esta fase:

- SemanticBERTStage — wrapper HuggingFace con cuatro propiedades de diseño que vale la pena mencionar explícitamente:
    - **Lazy loading + double-checked locking**: El modelo no se carga en `__init__` — el servicio arranca y responde
      health checks antes de que el modelo esté en memoria. El lock garantiza una sola carga bajo concurrencia sin pagar
      el costo de sincronización en cada request caliente.
    - **`pipeline_factory` inyectable**: La dependencia de transformers.pipeline es un parámetro de construcción, no un
      import hardcodeado. Los 35 tests del módulo corren en `~3` segundos sin GPU, sin descargar nada, sin tocar el
      filesystem de modelos.
    - **Timeout via `thread daemon`**: La inferencia corre en un thread daemon con join (`timeout=N`). Si supera el
      límite, `StageExecutionError(TimeoutError) → pipeline fail-closed`. El thread puede seguir corriendo en background
      pero al ser daemon no bloquea el proceso.
      `_DEFAULT_LABEL_MAP` normalizado a MAYÚSCULAS.

## Resumen de la Fase 4

`156/156 tests. 100% coverage. 0 fallos.`

### Desglose final por capa:

| Módulo                    | Test    | Cobertura |
|---------------------------|---------|-----------|
| `core/pipeline.py`        | 18 unit | 100%      |
| `core/exeptions.py`       | 4 units | 100%      |
| `models`                  | 12 unit | 100%      |
| `config/settings.py`      | 4 unit  | 100%      |
| `stages/deterministic.py` | 32 unit | 100%      |
| `stages/semantic.py`      | 35 unit | 100%      |
| `api/app.py` + `main.py`  | 11 unit | 100%      |

### Tres decisiones de esta fase que valen la pena anotar:

- `HTTP 200` siempre: Todos los business codes (`100/400/406/500`) viajan en el `JSON`, nunca como `HTTP-Status`.
  Un retry automático del cliente ante un `HTTP 500` podría reenviar un prompt que el sistema ya decidió bloquear — esto
  lo elimina por diseño.
- Lifespan fail-fast: Si los recursos no cargan al startup, el servicio no arranca. No hay modo degradado silencioso. Un
  servicio de seguridad que no puede garantizar su integridad no debe aceptar tráfico.
- `TestClient` con `app.state` inyectado. Los tests HTTP no tocan el lifespan en la mayoría de casos — inyectan el
  pipeline directamente en `app.state`.
  Esto desacopla los tests de HTTP contract de los tests de startup, sin sacrificar cobertura.

## Hydra: Configuración Dinámica

Hydra resuelve dos cosas que pydantic-settings no puede:

1. **Composición declarativa**: `pipeline` se describe como una lista ordenada de objetos en YAML, no como flags
   booleanos. Pasar de 2 a 3 stages, o reordenarlos, es editar el YAML sin tocar Python.
2. **Instanciación estructurada (_target_)**: Hydra puede instanciar clases Python directamente desde config via
   hydra.utils.instantiate(). Cada stage se define con _target_: src.stages.semantic.SemanticBERTStage y sus
   parámetros — esto es el mecanismo que garantiza que solo se pueden crear stages implementados. Si _target_ apunta a
   una clase que no existe, Hydra falla en startup con un error claro.

### Estructura de archivos a crear/modificar

```
clinical_safeguard/
├── conf/                          # NUEVO — directorio raíz de Hydra
│   ├── config.yaml                # NUEVO — config principal (referencia a grupos)
│   ├── pipeline/                  # NUEVO — grupo de configuración del pipeline
│   │   ├── default.yaml           # solo DeterministicStage
│   │   ├── clinical.yaml          # Deterministic + SemanticBERT
│   │   └── full.yaml              # Deterministic + SemanticBERT + AttackDetection
│   └── app/                       # NUEVO — config de la app (host, puerto, etc.)
│       └── default.yaml
```

## Fase Semántica

### Modelo Crisis Clínica:

- Modelo elegido: `mental/mental-bert-base-uncased`
- Razones frente a las alternativas:
    - fine-tuneado en datasets de crisis [(paper)](https://arxiv.org/abs/2110.15621).

### Modelo Ataques:

- Modelo elegido: `ProtectAI/deberta-v3-base-prompt-injection-v2`
    - Número de parámetros: **0.2B** (Base)
    - Arquitectura: DeBERTa v3 Base


- Razones frente a las alternativas:
    - Apache 2.0, entrenado en 7+ datasets públicos de seguridad, diseñado específicamente para detectar prompt
      injection en aplicaciones LLM, labels documentados y estables.


- Alternativas adicionales descartadas:
    - [OWASP JAIL](https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html) — no
      es un modelo, sino un conjunto de patrones regex. Útil como referencia para los bypass patterns, pero no como
      modelo de clasificación semántica.
    - `meta-llama/Prompt-Guard-86M` requiere aceptar la licencia de Meta y tiene restricciones comerciales para +700M de
      usuarios.