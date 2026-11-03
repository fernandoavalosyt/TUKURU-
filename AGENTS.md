# Tukuru - instrucciones para agentes

Este proyecto detecta **robo hormiga** en tiendas de abarrotes con vision artificial (YOLO-pose + heuristica por mano + Gemini como juez).

Lee, en este orden:

1. `CLAUDE.md` - contexto, mapa del codigo, convenciones, reglas de trabajo y trampas. **Obligatorio.**
2. `docs/ARQUITECTURA.md` - flujo, estructuras y reglas de decision.
3. `docs/DECISIONES.md` - por que es asi, alternativas descartadas y pendientes.
4. `docs/PRUEBAS.md` - videos de prueba, verdad de terreno, como reproducir.
5. `docs/ENTORNO.md` - preparar el entorno en otra PC (dependencias, modelo, `.env`) y problemas frecuentes.

Reglas minimas (detalle en `CLAUDE.md`): conversar y comentar en espanol; no enviar imagenes a Gemini sin autorizacion explicita; no mostrar ni subir secretos de `.env`; no hacer commit sin que se pida; ser honesto con los resultados.

Comandos: `python -m unittest discover -s tests` y `python -m tukuru.replay VIDEO --zones ZONAS.json --dashboard --out SALIDA.mp4`.
