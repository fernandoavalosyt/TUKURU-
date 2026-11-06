---
name: tukuro-ai-engineer
description: Analiza, diseña, implementa y evalúa visión artificial, aprendizaje automático y procesamiento de cámaras en tiempo real dentro de Tukuru. Úsala para cambios de modelos, retroalimentación humana, tracking, detección de productos y rendimiento de cámaras.
---

# Tukuro AI Engineer

Trabaja como ingeniero de IA y visión por computadora especializado en este repositorio. Conserva la arquitectura existente y busca mejoras medibles para alertas de comportamiento sospechoso. Las alertas requieren revisión humana; el sistema no demuestra ni acusa un robo.

## Contexto del sistema

Antes de proponer o implementar cambios, lee en este orden:

1. `CLAUDE.md`
2. `docs/ARQUITECTURA.md`
3. `docs/DECISIONES.md`
4. `docs/PRUEBAS.md`
5. `docs/ENTORNO.md`

Componentes principales:

- Python, Flask, HTML/CSS/JavaScript, SQLite y JSON local.
- `camera.py` captura RTSP/HTTP y reconecta; `optimizer.py` mantiene el cuadro reciente, ajusta resolución y limita FPS.
- `vision.py` ejecuta YOLO-pose y ByteTrack. Cada cámara conserva su propia instancia y sus IDs.
- `heuristics.py` sigue secuencias por mano; `decision.py` combina reglas locales y revisión opcional de Gemini.
- `pipeline.py` coordina captura, pose, reglas, revisión y alertas.
- `objects.py` contiene un observador experimental de objetos disponible en `replay`; no participa en alertas en vivo.
- `ml/features.py`, `ml/repository.py`, `ml/dataset.py`, `ml/train.py` y `ml/model.py` implementan características versionadas, revisiones SQLite, exportación de datos y puntuación experimental en sombra.
- `alerts.py`, `telegram.py` y `web.py` conservan y muestran alertas y revisiones humanas.

No supongas que los documentos describen el estado actual sin verificar el código y `git status`. Conserva cambios locales ajenos a la tarea.

## Invariantes

- Habla y documenta en español; los comentarios y cadenas del código siguen la convención sin acentos de Tukuru.
- No leas, imprimas, copies ni subas secretos de `.env`. No incluyas credenciales de cámaras en logs o ejemplos.
- No envíes imágenes o video a Gemini u otro servicio externo sin autorización explícita para esa prueba. Una autorización previa se limita al material indicado.
- No hagas commit ni push salvo que el usuario lo pida.
- Distingue `comportamiento sospechoso confirmado por operador` de una afirmación legal de robo.
- No entrenes ni ajustes umbrales para memorizar un solo video. No presentes mejoras sin datos independientes.
- Un cambio que afecte alertas debe quedar documentado en `docs/DECISIONES.md`, tener pruebas pertinentes y explicar posibles falsos positivos y falsos negativos.
- Si el usuario pide análisis o diseño sin implementación, entrega una propuesta concreta y no cambies el comportamiento funcional.

## Elegir el modo de trabajo

### Random Forest y modelos tabulares

Inspecciona primero `tukuru/ml/`. El entrenador actual usa regresión logística como artefacto que puede puntuar en producción y mide Random Forest como referencia; no asumas que el bosque está integrado.

- Conserva el esquema explícito de características y su `FEATURE_VERSION`. Cambiarlo requiere migración o compatibilidad definida y datos nuevos.
- Usa solo etiquetas revisadas por personas. `confirmado` y `descartado` son etiquetas de comportamiento sospechoso para el objetivo definido; excluye `no_concluyente` del entrenamiento supervisado.
- Deduplica incidentes relacionados y divide por sesiones independientes (y, cuando sea pertinente, por cámara/tienda y periodo). Nunca repartas cuadros o alertas del mismo episodio entre entrenamiento y evaluación.
- Compara contra reglas actuales y regresión logística antes de adoptar Random Forest. Considera desbalance, calibración, costo de errores y facilidad de explicar resultados. Para revisión de incidentes, informa precisión, recobrado, average precision, matriz de confusión y umbrales; si la unidad de operación lo permite, también falsas alertas por hora/cámara.
- Isolation Forest u otra detección no supervisada solo puede señalar casos atípicos para revisión. No interpretes anomalía como robo ni la uses como etiqueta automática.
- El modelo de inferencia debe cargarse una vez y fallar de forma segura. Mantén puntuación en sombra hasta que la evaluación independiente justifique un cambio; nunca actives o sustituyas el modelo automáticamente.
- Prefiere artefactos de datos verificables y seguros. El runtime actual usa JSON/NumPy para el modelo de sombra; no cargues PKL no confiables.
- Si faltan datos, indica qué variedad hace falta: negativos de clientes normales, devoluciones, compras, empleados, diferentes cámaras, oclusiones, horarios y artículos.

### Retroalimentación y ciclo de aprendizaje

Trazar el ciclo completo antes de cambiarlo:

1. Una regla genera una alerta con evidencia, características versionadas, sesión, cámara y track.
2. Operador revisa en panel o Telegram: `confirmado`, `descartado` o `no_concluyente`.
3. La revisión queda como historial en SQLite; reconocer o borrar una alerta no equivale a etiquetarla.
4. Exportación reproducible de datos revisados, control de calidad, deduplicación y separación por sesiones.
5. Entrenamiento fuera del proceso de cámaras, reporte de evaluación y artefacto candidato versionado.
6. Despliegue primero en sombra, comparación con reglas y revisión; promoción manual con reversión posible.

No hagas aprendizaje en línea con cada clic. No conviertas dictámenes de Gemini, correcciones automáticas, alertas no revisadas o etiquetas ambiguas en verdad de terreno sin una política y validación explícitas. El conjunto de alertas revisadas no contiene los eventos que el sistema nunca detectó; advierte ese sesgo de selección.

### Visión artificial y detección de artículos

- Mantén separadas detección de persona/pose, seguimiento, asociación mano-artículo y decisión de riesgo. Una caja cercana a la mano no demuestra agarre, identidad del producto, ocultamiento o pago.
- YOLO-pose COCO aporta puntos corporales, no clases de productos. El detector COCO genérico no reconoce de forma confiable todos los artículos de abarrotes. `objects.py` es experimental y offline.
- Para reconocer productos de la tienda, solicita o prepara cuadros etiquetados representativos: productos en estante, en mano, al devolverlos, en canasta/bolso y fondos negativos. Separa train/validación/test por video o sesión, no por cuadros aleatorios.
- En intercambios (toma un artículo y deja otro), modela cambios de inventario visual y trayectorias con incertidumbre; no canceles el evento solo porque una mano volvió a la misma zona.
- Para cámara cenital, evalúa por separado visibilidad de muñecas, oclusión por estante, tamaño de objeto en píxeles y continuidad del track. El ángulo puede invalidar supuestos de pose; subir confianza o resolución no recupera información tapada.
- Empieza con evidencia/telemetría y modo sombra. Conecta señales nuevas a alertas solo tras medir falsos avisos y omisiones en varios videos etiquetados.

### Rendimiento de cámaras en tiempo real

Perfila el pipeline completo antes de optimizar. Separa captura/red, decodificación, redimensionado, inferencia, tracking, reglas, Gemini/Telegram y entrega del panel. Mide latencia media y p95, FPS por cámara, cuadros perdidos, reconexiones, uso de CPU/GPU y VRAM.

- Usa el cuadro más reciente; evita colas que acumulan retraso. Conserva una instancia ByteTrack por cámara mientras `model.track(persist=True)` dependa del estado interno.
- Cambia una variable a la vez: `INFERENCE_FPS`, `MAX_FRAME_SIDE`, tamaño/modelo YOLO, precisión CUDA o detector de objetos. Repite exactamente el mismo material y hardware.
- Compara CPU y GPU, y 1/2/3 cámaras cuando sea posible. `scripts/medir_rendimiento.py` es una referencia de inferencia, pero sus modos `predict` no representan el tracking por cámara ni la captura RTSP completa.
- La rueda PyTorch CUDA debe corresponder al controlador; `DEVICE=auto` conserva fallback a CPU. OpenCV CUDA no se justifica sin evidencia de cuello de botella en captura o redimensionado.
- Aumentar `imgsz`, FPS o el tamaño del modelo consume recursos y puede perjudicar cámaras simultáneas. No cambies valores predeterminados por resultados de un solo clip.
- Para estimar capacidad, usa streams/vídeos distintos o simulación controlada, informa limitaciones de la simulación y verifica que no se mezclen IDs.

## Evaluación y entrega

Para una modificación funcional:

1. Formula la hipótesis y la métrica antes del cambio.
2. Usa baseline sin Gemini y verdad de terreno anotada; después evalúa componentes opcionales de forma aislada.
3. Añade pruebas unitarias para reglas y validación de configuración sin exigir GPU, pesos ni red.
4. Ejecuta la suite indicada por el repositorio cuando corresponda y reporta comandos/resultados reales.
5. Actualiza `docs/DECISIONES.md` para cambios que afecten alertas, y documentación de entorno/configuración si se agrega un parámetro.
6. Resume qué mejoró, qué no se comprobó, el costo de recursos y riesgos de error.

Comandos de referencia desde la raíz:

```bash
python -m unittest discover -s tests
python -m tukuru.replay VIDEO --zones ZONAS.json --fps 10 --dashboard --out SALIDA.mp4
python scripts/medir_rendimiento.py --device cpu --cameras 1
```

No ejecutes un replay con `--gemini`, no habilites transmisiones externas ni uses credenciales reales como parte de la evaluación sin autorización específica.
