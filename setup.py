#!/usr/bin/env python3
"""
Setup Simple para Chatbot RAG con Hugging Face
Instala solo las dependencias necesarias sin errores

Ejecutar: python setup_simple.py
"""

import subprocess
import sys
import os

def print_header():
    print("🚀 Setup Simple - Chatbot RAG con Hugging Face")
    print("=" * 60)
    print("Instalación mínima y funcional")
    print()

def upgrade_pip():
    """Actualiza pip para evitar problemas"""
    print("🔧 Actualizando pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ Pip actualizado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Advertencia actualizando pip: {e}")
        return True  # Continuar aunque falle

def install_basic_packages():
    """Instala paquetes básicos uno por uno"""
    print("\n📦 Instalando paquetes básicos...")
    
    packages = [
        "numpy>=1.26.0",  # Compatible con Python 3.12
        "PyPDF2>=3.0.0",  # Para leer PDFs
        "scikit-learn>=1.3.0",  # Para similaridad coseno
    ]
    
    for package in packages:
        print(f"🔄 Instalando {package}...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "--no-cache-dir", package
            ])
            print(f"   ✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"   ❌ {package}: {e}")
            return False
    
    print("✅ Paquetes básicos instalados")
    return True

def install_transformers():
    """Instala transformers y sentence-transformers con verificacion de versiones"""
    print("\n🤗 Instalando Hugging Face Transformers...")
    
    # Verificar version de transformers compatible
    transformers_packages = [
        "tokenizers>=0.13.0",
        ("transformers>=4.21.0", "transformers>=4.15.0"),  # Version compatible con text2text-generation, fallback
        "sentence-transformers>=2.2.0"
    ]
    
    success_count = 0
    for package_info in transformers_packages:
        if isinstance(package_info, tuple):
            # Intentar version principal, luego fallback
            primary, fallback = package_info
            print(f"🔄 Instalando {primary}...")
            
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--no-cache-dir", primary
                ])
                print(f"   ✅ {primary}")
                success_count += 1
            except subprocess.CalledProcessError:
                print(f"   ⚠️  {primary} fallo, intentando {fallback}...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install",
                        "--no-cache-dir", fallback
                    ])
                    print(f"   ✅ {fallback}")
                    success_count += 1
                except subprocess.CalledProcessError as e:
                    print(f"   ❌ Ambas versiones fallaron: {e}")
        else:
            # Paquete normal
            package = package_info
            print(f"🔄 Instalando {package}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "--no-cache-dir", package
                ])
                print(f"   ✅ {package}")
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"   ❌ {package}: {e}")
    
    print(f"✅ Transformers: {success_count}/3 paquetes instalados")
    return success_count >= 2

def test_installation():
    """Prueba que las importaciones funcionen y verifica pipelines"""
    print("\n🧪 Probando instalación...")
    
    tests = [
        ("numpy", "import numpy; print(f'✅ NumPy {numpy.__version__}')"),
        ("PyPDF2", "import PyPDF2; print('✅ PyPDF2 instalado')"),
        ("sklearn", "from sklearn.metrics.pairwise import cosine_similarity; print('✅ Scikit-learn OK')"),
    ]
    
    basic_success = 0
    for name, test_code in tests:
        try:
            exec(test_code)
            basic_success += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
    
    # Test transformers con verificacion de pipelines
    transformers_info = test_transformers_capabilities()
    
    # Resumen
    print(f"\n📊 Resultado: {basic_success}/3 paquetes básicos funcionando")
    
    if basic_success >= 2 and transformers_info['working']:
        print("🎉 ¡Instalación suficiente para funcionar!")
        if transformers_info['text2text']:
            print("🎯 Bonus: text2text-generation disponible para modelos T5/FLAN")
        return True
    elif basic_success >= 2:
        print("⚠️ Instalación parcial - funcionará con limitaciones")
        return True
    else:
        print("❌ Instalación insuficiente")
        return False

def test_transformers_capabilities():
    """Verifica qué capacidades de transformers están disponibles"""
    info = {
        'working': False,
        'text2text': False,
        'version': None
    }
    
    try:
        import transformers
        info['version'] = transformers.__version__
        print(f"🔄 Transformers version: {transformers.__version__}")
        
        from transformers import pipeline
        
        # Test pipelines básicos
        try:
            # Probar text-generation (siempre disponible)
            pipeline("text-generation", model="gpt2", device=-1)  # CPU
            print("✅ Pipeline text-generation OK")
            info['working'] = True
        except:
            print("❌ Pipeline text-generation fallo")
        
        # Test text2text-generation si está disponible
        try:
            if hasattr(transformers.pipelines, 'Text2TextGenerationPipeline'):
                print("✅ Pipeline text2text-generation disponible")
                info['text2text'] = True
            else:
                print("⚠️  Pipeline text2text-generation no disponible (version anterior)")
        except:
            pass
        
        # Test sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            print("✅ Sentence Transformers OK")
        except ImportError:
            print("⚠️ Sentence Transformers no disponible")
            
    except ImportError:
        print("❌ Transformers no disponible")
    
    return info

def create_src_folder():
    """Crea carpeta src/ si no existe"""
    src_path = "./src"
    if not os.path.exists(src_path):
        print(f"\n📁 Creando carpeta: {src_path}")
        os.makedirs(src_path)
        print("✅ Carpeta src/ creada")
        print("💡 Coloca tu archivo PDF aquí para usar el chatbot")
    else:
        print(f"✅ Carpeta {src_path} ya existe")

def print_usage_instructions():
    """Imprime instrucciones de uso"""
    print("\n" + "=" * 60)
    print("🎯 ¡Setup completado!")
    print()
    print("📋 Próximos pasos:")
    print("   1. Coloca tu archivo PDF en la carpeta './src/'")
    print("   2. Ejecuta: python chatbot.py")
    print()
    print("⚠️ Importante:")
    print("   - Solo coloca UN archivo PDF en src/")
    print("   - El PDF debe tener texto extraíble (no solo imágenes)")
    print()
    print("🆘 Si hay errores:")
    print("   - Verifica que tienes Python 3.8+")
    print("   - Intenta: pip install --upgrade pip setuptools")

def main():
    """Función principal"""
    print_header()
    
    # Ejecutar pasos
    steps = [
        ("Actualizando pip", upgrade_pip),
        ("Instalando paquetes básicos", install_basic_packages), 
        ("Instalando transformers", install_transformers),
        ("Probando instalación", test_installation),
    ]
    
    for step_name, step_func in steps:
        print(f"\n🔄 {step_name}...")
        success = step_func()
        if not success and step_name == "Instalando paquetes básicos":
            print("❌ Error crítico en paquetes básicos")
            print("💡 Intenta ejecutar manualmente:")
            print("   pip install --upgrade pip setuptools")
            print("   pip install numpy PyPDF2 scikit-learn")
            return
    
    # Crear estructura
    create_src_folder()
    
    # Instrucciones finales
    print_usage_instructions()

if __name__ == "__main__":
    main()