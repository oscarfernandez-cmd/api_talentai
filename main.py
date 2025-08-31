import os
import io
import json
import base64
import openai
import pdfplumber
import fitz
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import logging
from concurrent.futures import ThreadPoolExecutor

# === Configuración ===
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cv_processor")

# === Funciones ===

def extract_text_pdf(pdf_path: str) -> str:
    """Extrae texto de un PDF usando pdfplumber."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"[pdfplumber] Página {i} no legible: {e}")
    except Exception as e:
        logger.error(f"No se pudo abrir {pdf_path} con pdfplumber: {e}")
    return text.strip()


def extract_text_ocr(pdf_path: str) -> str:
    """Extrae texto de un PDF mediante OCR usando OpenAI."""
    text = ""
    try:
        doc = fitz.open(pdf_path)

        def process_page(page_num: int) -> str:
            try:
                page = doc[page_num]
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                img_str = base64.b64encode(img_bytes).decode("utf-8")

                response = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extrae TODO el texto de esta página del CV:"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ]
                    }]
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                logger.warning(f"[OCR] Página {page_num+1} falló: {e}")
                return ""

        # Procesamiento paralelo de páginas
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_page, range(len(doc))))
            text = "\n".join(results)

    except Exception as e:
        logger.error(f"No se pudo abrir {pdf_path} con PyMuPDF para OCR: {e}")

    return text.strip()


def pdf_to_text_or_ocr(pdf_path: str) -> str:
    """Extrae texto de un PDF o aplica OCR si está vacío."""
    text = extract_text_pdf(pdf_path)
    if not text:
        logger.info(f"No se detectó texto en {pdf_path}, aplicando OCR...")
        text = extract_text_ocr(pdf_path)
    return text


def get_json_from_openai(content: str) -> dict:
    """Envía el contenido a OpenAI para generar JSON estructurado."""
    prompt = f"""
Eres un asistente experto en procesamiento de CVs. Tu tarea es extraer información de un CV y clasificar al candidato.

Instrucciones:
1. Extrae información general:
- nombre
- correo
- teléfono
- educación
- experiencia
- habilidades
- certificaciones
- idiomas
- cursos

2. Evalúa si el candidato es adecuado para el puesto:

Requisitos mínimos para ser candidato:
- Tener experiencia laboral mínima de 2 años en áreas administrativas, gestión o relacionadas con IA.
- Contar con al menos 1 certificación o curso en IA, administración o áreas afines.
- Mostrar nivel intermedio o avanzado en al menos 1 idioma (inglés preferente).
- Incluir al menos 3 habilidades técnicas o administrativas relevantes al puesto.

Skills específicos requeridos:
- Carreras afines:
  - Ingeniería en Sistemas Computacionales
  - Ingeniería en Desarrollo de Software
  - Ingeniería en Mecatrónica
  - Ingeniería en Ciencias de Datos
- Perfil:
  - Graduado o último semestre de la carrera
  - Experiencia en gestión de proyectos
  - Experiencia en lenguajes de programación (Python, JavaScript, Visual Basic, HTML, PHP)
  - Conocimiento en bases de datos
  - Conocimiento en IA
  - Inglés intermedio o avanzado
- Modalidad: Híbrida
- Disponibilidad: Tiempo completo

- Si cumple con lo anterior, "candidato": true.
- Si no cumple, "candidato": false y en "motivo_no_candidato" explica la razón principal.
- De los cursos encontrados en el CV, solo devuelve los 10 más relevantes al perfil buscado.

3. Devuelve todo en un JSON válido con esta estructura:

{{
    "informacion_general": {{
        "nombre": "",
        "correo": "",
        "teléfono": [],
        "educación": [],
        "experiencia": [],
        "habilidades": [],
        "certificaciones": [],
        "idiomas": []
    }},
    "candidato": true,
    "motivo_no_candidato": ""
}}

4. Si algún campo no existe, usa "" o [] según corresponda.
5. No agregues texto adicional, solo JSON válido.

Documento (CV):
---
{content}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Falló la llamada a OpenAI: {e}")
        return {"error": "openai_failed", "raw_response": ""}


# === API FastAPI ===
app = FastAPI(title="API Procesamiento de CVs", version="1.0")


@app.post("/procesar-cv/")
async def procesar_cv(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    try:
        # Guardar archivo temporal
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Extraer texto
        text = pdf_to_text_or_ocr(temp_path)

        # Procesar con OpenAI
        result = get_json_from_openai(text)

        # Responder como JSON
        if isinstance(result, str):
            try:
                return JSONResponse(content=json.loads(result))
            except:
                return JSONResponse(content={"raw_response": result})
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Error procesando CV: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        # Eliminar archivo temporal
        if os.path.exists(temp_path):
            os.remove(temp_path)
