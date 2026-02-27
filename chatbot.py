import os
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class DocumentValidator:
    """Validador de documentos para el chatbot RAG"""
    
    def __init__(self, src_folder: str = "./src"):
        self.src_folder = Path(src_folder)
    
    def validate_src_folder(self) -> bool:
        """Valida que la carpeta src/ exista y tenga exactamente 1 PDF"""
        print("Validando carpeta src/...")
        
        # 1. Verificar que existe src/
        if not self.src_folder.exists():
            print("Error: La carpeta './src' no existe")
            print("Solucion: Crea la carpeta y coloca tu documento PDF ahi")
            return False
        
        # 2. Buscar archivos PDF
        pdf_files = list(self.src_folder.glob("*.pdf"))
        
        # 3. Validar cantidad
        if len(pdf_files) == 0:
            print("Error: No se encontraron archivos PDF en './src'")
            print("Solucion: Coloca exactamente UN archivo PDF en la carpeta src/")
            return False
        
        elif len(pdf_files) > 1:
            print("Error: Multiples documentos encontrados en './src'")
            print("Archivos encontrados:")
            for pdf in pdf_files:
                print(f"   - {pdf.name}")
            print("\nValide la cantidad de recursos (maximo 1) en './src'")
            return False
        
        # 4. Validación exitosa
        pdf_file = pdf_files[0]
        print(f"Documento encontrado: {pdf_file.name}")
        
        # 5. Validar que es un PDF válido
        if self._validate_pdf_content(pdf_file):
            print("PDF valido y legible")
            return True
        else:
            print("Error: El PDF no se puede leer correctamente")
            return False
    
    def get_document_path(self) -> Optional[Path]:
        """Retorna el path del documento si las validaciones pasan"""
        if self.validate_src_folder():
            pdf_files = list(self.src_folder.glob("*.pdf"))
            return pdf_files[0]
        return None
    
    def _validate_pdf_content(self, pdf_path: Path) -> bool:
        """Valida que el PDF se pueda abrir y tenga contenido"""
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Verificar que tiene páginas
                if len(pdf_reader.pages) == 0:
                    print("Advertencia: El PDF no tiene paginas")
                    return False
                
                # Intentar leer primera página
                first_page = pdf_reader.pages[0]
                text = first_page.extract_text()
                
                if len(text.strip()) < 10:
                    print("Advertencia: El PDF parece no tener texto extraible")
                    print("Nota: Verifica que no sea solo imagenes")
                    return False
                
                print(f"PDF info: {len(pdf_reader.pages)} paginas, ~{len(text)} chars en pagina 1")
                return True
                
        except ImportError:
            print("Error: PyPDF2 no esta instalado")
            print("Solucion: Ejecuta: pip install PyPDF2")
            return False
        except Exception as e:
            print(f"Error leyendo PDF: {str(e)}")
            return False
    
    def extract_pdf_text(self, pdf_path: Path) -> str:
        """Extrae todo el texto del PDF"""
        print(f"Extrayendo texto de: {pdf_path.name}")
        
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = []
                
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content.append(page_text)
                        print(f"   Pagina {i+1}: {len(page_text)} caracteres")
                
                full_text = "\n\n".join(text_content)
                print(f"Extraccion completa: {len(full_text)} caracteres totales")
                
                return full_text
                
        except Exception as e:
            print(f"Error extrayendo texto: {str(e)}")
            return ""

class DocumentProcessor:
    """Procesador que convierte PDF en chunks de texto"""
    
    def __init__(self):
        self.chunks = []
        
    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Divide texto en fragmentos con solapamiento"""
        print(f"Creando chunks de texto (tamaño: {chunk_size}, overlap: {overlap})...")
        
        # Dividir por párrafos primero
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Si el párrafo cabe en el chunk actual
            if len(current_chunk + paragraph) <= chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                # Guardar chunk actual si no está vacío
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Si el párrafo es muy largo, dividirlo
                if len(paragraph) > chunk_size:
                    words = paragraph.split()
                    temp_chunk = ""
                    for word in words:
                        if len(temp_chunk + word) <= chunk_size:
                            temp_chunk += word + " "
                        else:
                            if temp_chunk.strip():
                                chunks.append(temp_chunk.strip())
                            temp_chunk = word + " "
                    if temp_chunk.strip():
                        current_chunk = temp_chunk + "\n\n"
                else:
                    current_chunk = paragraph + "\n\n"
        
        # Agregar último chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        print(f"{len(chunks)} chunks creados")
        self.chunks = chunks
        return chunks

class EmbeddingEngine:
    """Motor de embeddings usando Hugging Face sentence-transformers"""
    
    def __init__(self):
        print("Cargando modelo de embeddings de Hugging Face...")
        try:
            from sentence_transformers import SentenceTransformer
            # Modelo pequeño y eficiente para embeddings
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.embeddings = None
            self.chunks = None
            print("Modelo de embeddings cargado")
        except Exception as e:
            print(f"Error cargando modelo de embeddings: {e}")
            raise
    
    def create_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Crea embeddings para los chunks"""
        print(f"Generando embeddings para {len(chunks)} chunks...")
        
        try:
            self.chunks = chunks
            # Generar embeddings de todos los chunks
            self.embeddings = self.model.encode(chunks, show_progress_bar=True)
            print(f"Embeddings generados: {self.embeddings.shape}")
            return self.embeddings
        except Exception as e:
            print(f"Error generando embeddings: {e}")
            raise
    
    def search_similar(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Busca chunks más similares a la pregunta"""
        if self.embeddings is None:
            raise ValueError("Primero debes crear embeddings")
        
        print(f"Buscando fragmentos relevantes para: '{query}'")
        
        # Generar embedding de la pregunta
        query_embedding = self.model.encode([query])
        
        # Calcular similitudes coseno
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Obtener índices de los más similares
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.chunks[idx], similarities[idx]))
        
        print(f"Encontrados {len(results)} fragmentos relevantes")
        return results

class HuggingFaceLLM:
    """Cliente LLM usando question-answering pipeline"""
    
    def __init__(self, model_name: str = "distilbert-base-cased-distilled-squad"):
        self.model_name = model_name
        print(f"Cargando modelo Question-Answering: {model_name}")
        
        try:
            from transformers import pipeline
            
            # Usar question-answering que está diseñado específicamente para esto
            self.qa_pipeline = pipeline(
                "question-answering",
                model=model_name
            )
            
            print("Modelo Question-Answering cargado correctamente")
            
        except Exception as e:
            print(f"Error cargando modelo Q&A: {e}")
            print("Intentando con modelo alternativo...")
            
            try:
                # Fallback a un modelo Q&A más pequeño
                self.qa_pipeline = pipeline(
                    "question-answering",
                    model="distilbert-base-uncased-distilled-squad"
                )
                print("Modelo Q&A alternativo cargado")
            except Exception as e2:
                print(f"Error con modelo alternativo: {e2}")
                raise
    
    def generate_response(self, context: str, question: str) -> str:
        """Genera respuesta usando question-answering"""
        
        try:
            print("Generando respuesta...")
            
            # Usar el pipeline Q&A directamente
            result = self.qa_pipeline(
                question=question,
                context=context[:5000]  # Limitar contexto para mejor rendimiento
            )
            
            answer = result['answer']
            confidence = result['score']
           
            # Si la confianza es baja, dar una respuesta más general
            if confidence < 0.00001:
                sentences = context.split('.')[:2]
                if len(sentences) >= 1:
                    return f"Basándome en el documento: {sentences[0].strip()}."
                else:
                    return "No encontré información específica sobre esa pregunta en el documento."
            
            # Expandir respuesta si es muy corta
            if len(answer) < 1:
                # Buscar la oración completa que contiene la respuesta
                sentences = context.split('.')
                for sentence in sentences:
                    if answer.lower() in sentence.lower():
                        return f"Según el documento: {sentence.strip()}."
                return f"El documento indica que: {answer}"
            
            return f"Según el documento: {answer}"
            
        except Exception as e:
            print(f"Error generando respuesta: {e}")
            return "Error: No pude procesar tu pregunta. Intenta reformularla."

class ChatbotRAG:
    """Chatbot RAG principal"""
    
    def __init__(self):
        print("Inicializando Chatbot RAG con Hugging Face")
        print("=" * 60)
        
        self.validator = DocumentValidator()
        self.processor = DocumentProcessor()
        self.embedding_engine = EmbeddingEngine()
        self.llm = HuggingFaceLLM()
        self.is_ready = False
        
    def setup(self) -> bool:
        """Configura el chatbot con el documento"""
        print("\nConfigurando chatbot...")
        
        # 1. Validar y obtener documento
        doc_path = self.validator.get_document_path()
        if not doc_path:
            return False
        
        # 2. Extraer texto del PDF
        text = self.validator.extract_pdf_text(doc_path)
        if not text:
            print("No se pudo extraer texto del PDF")
            return False
        
        # 3. Procesar en chunks
        chunks = self.processor.chunk_text(text)
        if not chunks:
            print("No se pudieron crear chunks del texto")
            return False
        
        # 4. Generar embeddings
        try:
            self.embedding_engine.create_embeddings(chunks)
            self.is_ready = True
            print("Chatbot RAG listo para usar!")
            return True
        except Exception as e:
            print(f"Error en configuracion: {e}")
            return False
    
    def chat(self, question: str) -> str:
        """Procesa pregunta y genera respuesta"""
        if not self.is_ready:
            return "El chatbot no esta configurado. Ejecuta setup() primero."
        
        try:
            # 1. Buscar fragmentos relevantes
            similar_chunks = self.embedding_engine.search_similar(question, top_k=2)
            
            # 2. Crear contexto con los mejores fragmentos
            context_parts = []
            for chunk, score in similar_chunks:
                context_parts.append(chunk)
            
            context = "\n\n".join(context_parts)
            
            # 3. Generar respuesta usando LLM
            response = self.llm.generate_response(context, question)
            
            return response
            
        except Exception as e:
            return f"Error procesando pregunta: {str(e)}"
    
    def interactive_chat(self):
        """Loop de chat interactivo"""
        print("\n" + "=" * 60)
        print("CHATBOT RAG INTERACTIVO")
        print("=" * 60)
        print("Haz preguntas sobre el documento. Escribe 'salir' para terminar.")
        print("-" * 60)
        
        while True:
            try:
                question = input("\nTu pregunta: ").strip()
                
                if question.lower() in ['salir', 'exit', 'quit', 'q']:
                    print("Hasta luego! Gracias por usar el chatbot RAG.")
                    break
                
                if not question:
                    print("Por favor escribe una pregunta.")
                    continue
                
                # Procesar pregunta con timeout implícito
                print("Procesando...")
                response = self.chat(question)
                print(f"\nRespuesta: {response}")
                print("-" * 60)
                
            except KeyboardInterrupt:
                print("\nHasta luego!")
                break
            except Exception as e:
                print(f"\nError inesperado: {e}")
                print("Intentando continuar...")
                continue

def main():
    """Función principal"""
    try:
        # Crear y configurar chatbot
        chatbot = ChatbotRAG()
        
        if chatbot.setup():
            # Iniciar chat interactivo
            chatbot.interactive_chat()
        else:
            print("\nSoluciones posibles:")
            print("   1. Verificar que existe la carpeta './src'")
            print("   2. Colocar exactamente UN archivo PDF en './src'")
            print("   3. Verificar que el PDF tiene texto extraible")
            
    except Exception as e:
        print(f"\nError critico: {e}")
        print("Verifica que tienes todas las dependencias instaladas")

if __name__ == "__main__":
    main()