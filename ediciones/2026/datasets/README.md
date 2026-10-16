# Datasets — Módulo 2 (Fundamentos de ML), edición 2026

Datos usados en los notebooks del módulo. Se publican al repo público
`sebastiancontz/ust-fundamentos-machine-learning-colab` (job `colab` del workflow) y se consumen desde Colab por
URL raw (ver `_plantilla-notebook.ipynb`). **Todos los datos son sintéticos (generados para el curso)
o de uso público.** No se incluyen datos personales, reservados ni de marca.

## Convención de procedencia

Toda fila **Sintético** debe enlazar su script generador `.py`, declarar que no representa a una
organización real y registrar su licencia como **No declarada** cuando el titular no haya publicado
una. Esa procedencia se replica visiblemente en los materiales que usen el dataset.

| Archivo | Origen | Descripción | Usado en |
|---|---|---|---|
| `clientes.csv` | **Sintético** (generado para el curso; [`generar_clientes.py`](../../../scripts/generar_clientes.py)) | ~2000 clientes de telecom con `fuga` (churn) sí/no. **Desbalanceado a propósito (~18% fuga)** para enseñar por qué el accuracy engaña y para qué sirve la validación estratificada. Columnas: `edad`, `antiguedad_meses`, `plan`, `cargo_mensual`, `llamadas_soporte`, `fuga`. La antigüedad baja y las llamadas a soporte son los predictores fuertes. No representa a una organización ni a personas reales. | Clase 4 (clasificación); Clase 5 (segmentación, sin usar `fuga`; + anomalías) |
| `banco_marketing.csv` | **Uso público** — [Bank Marketing (UCI #222)](https://archive.ics.uci.edu/dataset/222/bank+marketing), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); Moro, Cortez & Rita (2014) | 8.000 contactos telefónicos **reales** de la campaña de un banco portugués (muestra estratificada del `bank-additional-full.csv` de 41.188; ~11% contrata). Target `contrata` 0/1 (depósito a plazo). Columnas traducidas y numéricas (ordinales como códigos; `unknown` → celda vacía = **faltantes reales**). **`duracion_llamada_seg` se conserva a propósito**: es el caso real documentado por UCI de variable que "sabe la respuesta" (solo se conoce tras la llamada) — la Clase 6 la usa para demostrar la fuga y luego la descarta. Generado por `scripts/gen_banco_marketing.py`. | Clase 6 (caso principal: tuning, interpretación, persistencia y valor esperado) |
| `credito_aleman.csv` | **Uso público** — [Statlog German Credit Data (UCI #144)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); Hans Hofmann (1994) | 1000 solicitudes de crédito con `incumplimiento` 0/1 (~30% malos). Versión preparada para el curso: 8 columnas traducidas al español y **todas numéricas** (ordinales como códigos: `cuenta_corriente` 0–3, `ahorros` 0–4, `historial_crediticio` 0–4), lista para SHAP/PDP. Generada por `scripts/gen_c06_tuning.py`/OpenML (`credit-g`). | Clase 6 (interpretabilidad: caso credit scoring con lupa regulatoria) |
| `vivienda.csv` | **Sintético** (generado para el curso) | Precios de viviendas (UF) con superficie, comuna, etc. | — (sin uso actual; era candidato para la Clase 3, que finalmente reutiliza `precios_casas_rm.csv`) |
| `precios_casas_rm.csv` | **Compilación propia** — Sebastián Contreras, mediante *web scraping* de avisos de [Enlace Inmobiliario](https://www.enlaceinmobiliario.cl/) y [GoPlaceIt](https://www.goplaceit.com/) | Casas de la RM: área, dormitorios, baños, comuna, cercanía a metro y `precio_uf`. Datos agregados y sin identificadores personales. | Clases 2 y 3 (datos/preparación y regresión) |
| `trafico_aereo_cl.csv` | **Uso público** — [Junta de Aeronáutica Civil](https://datos.gob.cl/dataset/trafico-aereo), [CC0](https://creativecommons.org/publicdomain/zero/1.0/); descargado el 2026-07-12 (recurso `trafico_historico.csv`) | Pasajeros aéreos de Chile, **serie mensual 2010–2025** (192 meses), en formato StatsForecast: `unique_id` en {`total`, `nacional`, `internacional`}, `ds` (mes, día 1), `y` (pasajeros). Agregado desde el detalle operador-ruta-mes usando **solo salidas** (`OPER_2`=SALEN) para no doble contar; `total` = nacional + internacional. Estacionalidad anual marcada (los eneros, por vacaciones de verano, son el máximo del año; junio el mínimo), tendencia creciente y el **quiebre estructural COVID** (mínimo en abril 2020: 55k) con recuperación posterior. 576 filas. | Clase 7 (forecasting): práctica principal del notebook |
| `demanda_sen_2025.csv` | **Uso público** — [Coordinador Eléctrico Nacional](https://sipub.api.coordinador.cl/api/v2/recursos/demanda_sistema_real/) (API pública SIP; licencia abierta no declarada); descargado el 2026-07-12 con `scripts/cen_downloader.py` (requiere `CEN_API_KEY` en `.env`, gitignoreado) | Demanda eléctrica **real horaria** del Sistema Eléctrico Nacional (SEN) de Chile, **año 2025 completo**: 8.760 observaciones. Formato StatsForecast: `unique_id` (constante `SEN`), `ds` (marca horaria), `y` (demanda en MW). Estacionalidad **diaria** fuerte (~30%) y **semanal** (~6%, laboral > fin de semana); **anual casi nula** (verano ≈ invierno, ~9.900 MW). **Extracto crudo: requiere limpieza antes del notebook** (ver notas). | Clase 7 (forecasting): caso operativo real de alta frecuencia. Rol exacto (demo en notebook vs. ejemplo en clase) se define al construir la clase (Fase 5D) |

## Notas

- **Clase 5 — PCA**: usa el dataset de **cáncer de mama de Wisconsin**, que viene incluido en
  scikit-learn (`from sklearn.datasets import load_breast_cancer`). No se guarda CSV: el notebook lo
  carga directo desde sklearn (30 mediciones del tumor + diagnóstico benigno/maligno).


- **`precios_casas_rm.csv`**: re-guardado en **UTF-8** (la fuente venía en latin-1; afectaba `baños`).
  Los valores de `precio_uf` corresponden a **precio de venta** (no arriendo). Para la práctica de la
  Clase 2 se introducen algunos **valores faltantes de forma deliberada** (documentado en el notebook)
  con el fin de enseñar imputación; el dataset original viene completo.
- Procedencia/licencia de los datos públicos: usar solo con fines **educativos**. Si se conoce la
  fuente original y su licencia, anotarla aquí.
- **`demanda_sen_2025.csv`**: datos operativos públicos del Coordinador Eléctrico Nacional, vía su
  API SIP (registro gratuito → `user_key`; la key vive en `.env`, gitignoreado, NUNCA en el repo).
  Se cita la fuente (Coordinador Eléctrico Nacional) en notebook y slides. **Licencia:** la API SIP
  no declara licencia abierta (`termsOfService`/`license` nulos en el spec); por **decisión del
  usuario (2026-07-12)** se publica **con atribución** al Coordinador. Es **extracto crudo** y tiene tres
  particularidades reales a resolver al preparar la clase (Fase 5D):
  1. **Apagón nacional del 25-feb-2025**: la tarde de ese día la demanda cae a 68–260 MW (colapso real
     del sistema, no error de datos) y se recupera en horas. Sirve como ejemplo de **shock estructural**,
     pero distorsiona el entrenamiento → decidir si se conserva como caso didáctico o se excluyen esas
     horas (documentado). Hay 16 horas < 5.000 MW, casi todas de ese episodio.
  2. **Cambio de hora**: el 6-abr (25 h) genera una **marca de tiempo duplicada** y el 7-sep (23 h) deja
     **un hueco** — el índice horario no queda regular. Limpiar (deduplicar/rellenar) antes de modelar.
  3. **Sin estacionalidad anual**: verano ≈ invierno (~9.900 MW). La estacionalidad útil es **diaria**
     (`season_length=24`) y **semanal**, no anual. El año completo aporta historia y ventanas de
     validación, no contraste estacional (punto enseñable: no toda serie tiene ciclo anual).

## Datasets del Proyecto Final (Clase 8) — espejo de datos públicos

Los 11 datasets del **menú del proyecto final** se **espejan aquí** (formato **Parquet**) como respaldo,
por si se caen los enlaces de origen (los portales de datos abiertos cambian URLs). Descargados el
**2026-07-14** desde su fuente oficial. Se aplicó una **preparación mínima** para que los alumnos no
partan de cero, sin resolverles el trabajo evaluado: **columnas traducidas al español**, tipos
corregidos (decimales con coma → `float`), los de **forecasting agregados a una grilla razonable con
`fecha` lista** (NO en formato `unique_id`/`ds`/`y`), las categóricas de colisiones **decodificadas**, y
recorte a columnas útiles donde había ruido. **Sigue siendo trabajo del alumno**: EDA, *outliers*,
faltantes, selección de variables, armar la serie y modelar. El notebook `notebooks/datasets.ipynb`
carga desde este espejo con `pd.read_parquet(...)`. La ficha oficial (fuente canónica) y la atribución
que exige cada licencia están abajo y en la pauta (`proyecto-final.qmd`). Todo el pipeline es
reproducible con `scripts/prep_datasets_proyecto.py`.

| Archivo | Fuente (entidad) | Licencia | Objetivo / valor | Preparación aplicada |
|---|---|---|---|---|
| `netbilling.parquet` | CNE (Comisión Nacional de Energía) · datos.gob.cl | CC BY 4.0 | `potencia_kw` (regresión) | 39.636 × 9. Sin cambios de estructura. |
| `capacidad_instalada.parquet` | CNE · datos.gob.cl | CC BY 4.0 | `potencia_neta_mw` (regresión) | 1.304 × 13. Recortado (fuera IDs/RUT/coordenadas/combustibles); `potencia_*_mw` a `float`. |
| `indicadores_rem20.parquet` | MINSAL–DEIS · datos.gob.cl | CC0 | `NUMERO_EGRESOS` (regresión) | 165.235 × 15. **Quitados los indicadores derivados** (ocupación, letalidad, promedios, rotación) = fuga. |
| `trafico_vial_gb.parquet` | Department for Transport (GB) · data.gov.uk | OGL v3.0 | `total_vehiculos` (regresión) | 1.678 × 9. Columnas traducidas del inglés. |
| `colisiones_gb.parquet` | Department for Transport (GB) · GOV.UK | OGL v3.0 | `gravedad_colision` (clasificación: `fatal`/`grave`/`leve`) | 100.927 × 15 (de 44). Categóricas **decodificadas** (clima, luz, tipo de vía, superficie, urbano/rural, día); columnas-fuga de severidad ajustada **excluidas**; faltantes (`-1`) → vacío. |
| `licencias_conducir.parquet` | datos.gob.cl (Chile) | CC0 | `tipo_tramite` (clasificación, 6 categorías) | 71.224 × 14. `tipo_tramite` recodificado desde `Glosa Solicitud` (`;`+`latin-1` ya resueltos). |
| `establecimientos_salud.parquet` | MINSAL–DEIS · datos.gob.cl | CC0 | `TipoSistemaSaludGlosa` (clasificación) | 5.707 × 14. Recortado a columnas útiles; agregada `sistema_publico` (1/0) para el caso binario. |
| `generacion_sen.parquet` | CNE · datos.gob.cl | CC BY 4.0 | `generacion_mwh` (forecasting) | 292 × 5. **Agregado a subsistema × mes** + `fecha`; serie = `subsistema`. |
| `defunciones_semana.parquet` | MINSAL–DEIS · datos.gob.cl | CC0 | `muertes` (forecasting) | 14.192 × 5. **Agregado a región × semana** + `fecha` (semana ISO); serie = `region`. |
| `transporte_victoria.parquet` | Dept. of Transport and Planning, Victoria (AU) | CC BY 4.0 | `pasajeros` (forecasting) | 594 × 5. **Formato largo** (`modo`, `anio`, `mes`, `fecha`, `pasajeros`); serie = `modo`. |
| `urgencias_respiratorias.parquet` | MINSAL–DEIS · datos.gob.cl | **CC BY-NC** (No Comercial) | `casos_total` (forecasting) | 10.839 × 10 (de 3,5M). **Agregado a región × semana** + `fecha` + desglose por edad; serie = `region`. **Uso educativo/no comercial**. |

> **Cumplimiento de licencias:** CC0 = dominio público (sin condiciones); CC BY 4.0 y OGL v3.0 =
> redistribución permitida **con atribución** (arriba); **CC BY-NC** (urgencias) = solo **uso no
> comercial** — el repo del curso es educativo/sin fines de lucro. Toda reutilización debe conservar la
> fuente y la licencia. El microdato completo de urgencias (3,5M filas) queda en la ficha oficial.
