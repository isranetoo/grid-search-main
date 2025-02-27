import os
import base64
from io import BytesIO
import pandas as pd
import pytesseract
from PIL import Image
from scipy.ndimage import gaussian_filter, median_filter
import time
from datetime import datetime
import sys
from collections import Counter

# Import the advanced filters module
from advanced_filters import (
    advanced_preprocess,
    preprocess_pipeline,
    apply_multiple_ocr,
    scale_image,
    solve_captcha_with_advanced_filters
)

# Configura o caminho do Tesseract no Windows
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image, th1, th2, sigma1, sigma2):
    """Processa a imagem para otimização do OCR utilizando técnicas avançadas."""
    # Use the advanced preprocessing technique
    return advanced_preprocess(image, th1, th2, sigma1, sigma2)

def extract_text_from_image(image, ocr_configs):
    """Extrai texto da imagem usando múltiplas configurações do OCR com técnicas avançadas."""
    # Scale the image for better OCR
    scaled_image = scale_image(image)
    
    # Use the advanced OCR approach with multiple preprocessing strategies
    preprocessed_images = preprocess_pipeline(scaled_image)
    
    # Apply OCR using the advanced method
    result = apply_multiple_ocr(preprocessed_images)
    
    # If no result, fall back to traditional approach
    if not result:
        for config in ocr_configs:
            text = pytesseract.image_to_string(scaled_image, config=config)
            text = text.strip().replace(" ", "").replace("\n", "")
            text = ''.join(c for c in text if c.isalnum())
            if 3 <= len(text) <= 8:
                return text
    
    return result

def solve_captcha_local(th1, th2, sigma1, sigma2, image_data_base64):
    """Resolve o CAPTCHA aplicando preprocessamento avançado e OCR."""
    try:
        # Decodifica a imagem base64
        original_image = Image.open(BytesIO(base64.b64decode(image_data_base64)))
        
        # Primeiro tenta com a abordagem totalmente integrada
        result = solve_captcha_with_advanced_filters(original_image)
        
        # Se não obtiver resultado, tenta com parâmetros específicos
        if not result:
            processed_image = preprocess_image(original_image, th1, th2, sigma1, sigma2)
            
            # Configurações do OCR - testamos várias para maximizar acertos
            ocr_configs = [
                '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
                '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
                '--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
                '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
                '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
            ]
            
            result = extract_text_from_image(processed_image, ocr_configs)
        
        return result
    except Exception as e:
        print(f"Erro ao processar a imagem: {e}")
        return None

def grid_search_on_images(image_folder, param_ranges, results_file):
    """Executa uma busca em grade para encontrar os melhores parâmetros."""
    results = {}
    best_accuracy = 0
    best_params = None
    total_combinations = (
        len(param_ranges["th1"]) * 
        len(param_ranges["th2"]) * 
        len(param_ranges["sigma1"]) * 
        len(param_ranges["sigma2"])
    )
    
    print(f"Iniciando busca em grade com {total_combinations} combinações...")
    start_time = time.time()
    
    # Pré-carrega todas as imagens
    image_data = {}
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    total_images = len(image_files)
    
    print(f"Carregando {total_images} imagens...")
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        with open(img_path, "rb") as f:
            image_data[img_file] = base64.b64encode(f.read())
    
    # Teste inicial com o módulo advanced_filters - versão em estilo de terminal do tester_simple.py
    print("\n" + "="*80)
    print("TESTE INICIAL COM ADVANCED_FILTERS (SEM PARÂMETROS CUSTOMIZADOS)")
    print("="*80)
    advanced_scores = {}
    advanced_results = {}
    
    for img_file in image_files:
        expected = os.path.splitext(img_file)[0]  # Nome do arquivo sem extensão
        with open(os.path.join(image_folder, img_file), "rb") as f:
            img_data = f.read()
            img = Image.open(BytesIO(img_data))
            prediction = solve_captcha_with_advanced_filters(img)
            
        match = prediction.lower() == expected.lower()
        advanced_scores[expected] = 1 if match else 0
        advanced_results[expected] = prediction
        
        # Terminal output similar to tester_simple.py
        status = "\033[1;32mOK\033[0m" if match else f"\033[1;31mERROR\033[0m"
        print(f"Imagem: {img_file:<20} | Esperado: {expected:<10} | Detectado: {prediction:<10} | {status}")
    
    # Calcula e exibe a acurácia do teste inicial
    advanced_accuracy = sum(advanced_scores.values())
    print("\n" + "-"*80)
    print(f"Acurácia com advanced_filters: {advanced_accuracy}/{total_images} ({advanced_accuracy/total_images:.1%})")
    print("="*80 + "\n")
    
    # Salva os resultados do teste inicial
    pd.DataFrame.from_dict({
        "expected": list(advanced_scores.keys()),
        "detected": [advanced_results[k] for k in advanced_scores.keys()],
        "match": list(advanced_scores.values())
    }).to_csv(f"./results/advanced_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    # Contador de progresso
    current_combination = 0
    
    # Itera por todas as combinações de parâmetros
    for th1 in param_ranges["th1"]:
        for th2 in param_ranges["th2"]:
            for sigma1 in param_ranges["sigma1"]:
                for sigma2 in param_ranges["sigma2"]:
                    current_combination += 1
                    scores = {}
                    combination_results = {}
                    
                    print("\n" + "-"*80)
                    print(f"COMBINAÇÃO {current_combination}/{total_combinations}: th1={th1}, th2={th2}, sigma1={sigma1}, sigma2={sigma2}")
                    print("-"*80)
                    
                    # Para cada combinação de parâmetros, testa todas as imagens
                    for img_file in image_files:
                        expected = os.path.splitext(img_file)[0]  # Nome do arquivo sem extensão
                        prediction = solve_captcha_local(
                            th1, th2, sigma1, sigma2, image_data[img_file]
                        )
                        
                        match = prediction.lower() == expected.lower() if prediction else False
                        scores[expected] = 1 if match else 0
                        combination_results[expected] = prediction
                        
                        # Terminal output similar ao tester_simple.py
                        status = "\033[1;32mOK\033[0m" if match else f"\033[1;31mERROR\033[0m"
                        print(f"Imagem: {img_file:<20} | Esperado: {expected:<10} | Detectado: {prediction:<10} | {status}")
                    
                    # Calcula a acurácia
                    accuracy = sum(scores.values())
                    results[(th1, th2, sigma1, sigma2)] = scores
                    
                    # Salva os melhores parâmetros
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (th1, th2, sigma1, sigma2)
                        print("\n\033[1;33m" + "!"*80 + "\033[0m")
                        print(f"\033[1;33mNOVOS MELHORES PARÂMETROS ENCONTRADOS! Acurácia: {accuracy}/{total_images} ({accuracy/total_images:.1%})\033[0m")
                        print("\033[1;33m" + "!"*80 + "\033[0m")
                    
                    # Exibe progresso
                    elapsed_time = time.time() - start_time
                    progress = current_combination / total_combinations
                    estimated_total_time = elapsed_time / progress if progress > 0 else 0
                    remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
                    
                    print("\n" + "-"*80)
                    print(f"RESUMO DA COMBINAÇÃO {current_combination}/{total_combinations}")
                    print(f"Parâmetros: th1={th1}, th2={th2}, sigma1={sigma1}, sigma2={sigma2}")
                    print(f"Acertos: {accuracy}/{total_images} ({accuracy/total_images:.1%})")
                    print(f"Melhor até agora: {best_accuracy}/{total_images} ({best_accuracy/total_images:.1%})")
                    print(f"Progresso: {current_combination}/{total_combinations} ({progress:.1%})")
                    print(f"Tempo decorrido: {elapsed_time/60:.1f} minutos")
                    print(f"Tempo restante estimado: {remaining_time/60:.1f} minutos")
                    print(f"Tempo total estimado: {estimated_total_time/60:.1f} minutos")
                    print("-"*80 + "\n")
                    
                    # Exibe uma barra de progresso simplificada
                    progress_bar_length = 50
                    progress_filled = int(progress * progress_bar_length)
                    progress_bar = "[" + "=" * progress_filled + " " * (progress_bar_length - progress_filled) + "]"
                    sys.stdout.write(f"\r{progress_bar} {progress:.1%}")
                    sys.stdout.flush()
                    
                    # Salva resultados parciais periodicamente
                    if current_combination % 5 == 0:
                        print("\nSalvando resultados parciais...")
                        pd.DataFrame.from_dict(results, orient="index").to_csv(results_file)
    
    # Exibe os melhores resultados
    print("\n" + "="*80)
    print("RESULTADOS FINAIS")
    print("="*80)
    print(f"Melhores parâmetros encontrados: th1={best_params[0]}, th2={best_params[1]}, sigma1={best_params[2]}, sigma2={best_params[3]}")
    print(f"Melhor acurácia: {best_accuracy}/{total_images} ({best_accuracy/total_images:.1%})")
    print(f"Tempo total de execução: {(time.time() - start_time)/60:.1f} minutos")
    print("="*80)
    
    # Salva os resultados finais
    pd.DataFrame.from_dict(results, orient="index").to_csv(results_file)
    print(f"Resultados salvos em '{results_file}'")
    
    # Salva os melhores parâmetros em um arquivo separado
    with open(f"./results/best_params_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(f"Best parameters: th1={best_params[0]}, th2={best_params[1]}, " +
                f"sigma1={best_params[2]}, sigma2={best_params[3]}\n")
        f.write(f"Accuracy: {best_accuracy}/{total_images} ({best_accuracy/total_images:.1%})")
        
    return results, best_params, best_accuracy

# Configurações e execução
if __name__ == "__main__":
    image_folder = r'C:\Users\IsraelAntunes\Desktop\grid-search-main\image'
    
    # Parâmetros atualizados para melhor se adequarem às técnicas avançadas
    param_ranges = {
        "th1": [100, 120, 140, 160, 180, 200],
        "th2": [130, 140, 150, 160, 170, 180],
        "sigma1": [0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
        "sigma2": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    }
    
    # Cria um nome de arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"./results/results_{timestamp}.csv"
    
    # Cria o diretório de resultados se não existir
    os.makedirs("./results", exist_ok=True)
    
    print("\n" + "*"*80)
    print("* GRID SEARCH - DETECTOR AVANÇADO DE CAPTCHAS")
    print("*"*80)
    print(f"* Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"* Diretório de imagens: {image_folder}")
    print(f"* Arquivo de resultados: {results_file}")
    total_combinations = (
        len(param_ranges["th1"]) * 
        len(param_ranges["th2"]) * 
        len(param_ranges["sigma1"]) * 
        len(param_ranges["sigma2"])
    )
    print(f"* Total de combinações: {total_combinations}")
    print("*"*80 + "\n")
    
    # Executa a busca em grade
    results, best_params, best_accuracy = grid_search_on_images(image_folder, param_ranges, results_file)