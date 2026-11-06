# Detector local de cabezas para camara cenital

## Estado

Tukuru todavia no integra un detector entrenado para cabezas cenitales. YOLO-pose
continua siendo el detector de personas y articulaciones en vivo. Los ajustes por
camara de confianza y altura solo modifican filtros; no crean detecciones nuevas.

Se agrego un flujo local para recoger cuadros sin overlay y etiquetar cajas de cabeza:

1. Conecta la camara en Tukuru y confirma en el panel que figure **conectada**.
2. Desde la raiz del repositorio, muestrea una sesion. Usa un ID distinto para cada
   grabacion independiente:

   ```
   python scripts/muestrear_cabezas.py --camera CAM-01 --session CAM01-turno-2026-11-05 --duration 120 --interval 1
   ```

   Guarda como maximo 120 cuadros originales en `data/head_dataset/<sesion>/images/`.
   Consulta unicamente la instancia local de Tukuru en `127.0.0.1`; no usa Gemini,
   Telegram ni otros servicios. El endpoint local comprueba que la camara este conectada.
   Tambien acepta un video local con `--video RUTA`, en lugar de `--camera`.

3. Abre `scripts/etiquetar_cabezas.html` en un navegador local. Escribe el mismo ID de
   sesion, selecciona las imagenes de **una sola sesion**, y dibuja una caja ajustada a
   cada cabeza visible. En cuadros sin cabeza, pulsa **Marcar sin cabeza** para guardar
   un ejemplo negativo. Exporta el JSON; el navegador no carga librerias ni envia datos
   por red. Si el navegador se cierra, exporta antes para conservar el trabajo.
4. Copia el JSON descargado a una ruta accesible desde WSL y conviertelo:

   ```
   python scripts/importar_etiquetas_cabezas.py --session CAM01-turno-2026-11-05 --annotations RUTA/etiquetas.json
   ```

   Escribe etiquetas YOLO clase `0` (cabeza) en `labels/`, valida nombres y coordenadas
   contra el manifiesto y conserva las imagenes y etiquetas dentro de `data/` (ignorado
   por Git).

5. Entrena un candidato local desde los pesos ya disponibles, sin descargar modelos:

   ```
   python scripts/entrenar_detector_cabezas.py --session CAM01-turno-2026-11-05 --epochs 30 --imgsz 640 --batch 4
   ```

   El script separa cuadros positivos y negativos para una validacion experimental,
   inicializa YOLO de deteccion con los pesos locales `yolov8n-pose.pt` y conserva el
   mejor `best.pt` bajo `data/head_dataset/training/`. No instala el candidato en Tukuru.

## Datos necesarios antes de entrenar

Recoge situaciones variadas de la camara real: personas solas y agrupadas, distintos
horarios e iluminacion, distancias, cabezas parcialmente ocultas y cuadros negativos sin
personas. Anota todas las cabezas visibles, no solo las de episodios sospechosos. Guarda
grabaciones independientes con IDs distintos; no dividas cuadros de una misma grabacion
entre entrenamiento y validacion. Los cuadros consecutivos son muy parecidos y producirian
una evaluacion engañosa.

## Primer candidato

Se entreno en CPU el 2026-11-05 con una sola sesion (`CAM01-turno-01`): 76 cuadros
etiquetados, 62 cajas y 15 cuadros de validacion con 12 cabezas. Se transfirieron 349 de
355 capas desde `models/yolov8n-pose.pt`. Las metricas de esa particion aleatoria fueron
precision 0.519, recall 0.583, mAP50 0.493 y mAP50-95 0.205. El tiempo observado fue
65.2 ms de inferencia por cuadro en CPU.

El candidato esta en
`data/head_dataset/training/head_CAM01-turno-01_20261105_230845/run/weights/best.pt`.
Estas metricas son solo una referencia: entrenamiento y validacion provienen del mismo
video y cuadros cercanos, por lo que puede haber fuga temporal y sobreajuste. No se
conecto a las camaras ni a las alertas. Para una evaluacion fiable hacen falta grabaciones
independientes con sus anotaciones. Aunque detecte una cabeza, por si sola no da pose de
muñecas ni evidencia de alcance, ocultamiento, toma de producto o robo.
