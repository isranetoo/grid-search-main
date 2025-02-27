import os
import base64
from io import BytesIO
import pandas as pd
import numpy as np
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from scipy.ndimage import gaussian_filter, median_filter
import time
from datetime import datetime
from collections import Counter
import sys

# Configura o caminho do Tesseract no Windows
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image, th1, th2, sigma1, sigma2):
    """Processa a imagem para otimização do OCR com filtros avançados."""
    # Converte para preto e branco
    gray_image = image.convert("L")
    
    # Aumenta o contraste
    enhanced = ImageEnhance.Contrast(gray_image).enhance(2.0)
    
    # Aplica threshold adaptativo usando numpy
    img_array = np.array(enhanced)
    h, w = img_array.shape
    
    # Cria uma versão adaptativa do threshold
    window = 15
    threshold_map = np.zeros_like(img_array, dtype=np.float32)
    padded = np.pad(img_array, window//2, mode='reflect')
    
    for i in range(h):
        for j in range(w):
            window_vals = padded[i:i+window, j:j+window]
            threshold_map[i, j] = np.mean(window_vals) - th1
    
    # Aplica o threshold
    first_threshold = np.where(img_array > threshold_map, 255, 0).astype(np.uint8)
    
    # Primeiro filtro Gaussiano
    blurred = gaussian_filter(first_threshold, sigma=sigma1)
    # Remove ruído com filtro de mediana
    blurred = median_filter(blurred, size=3)
    
    # Segundo threshold
    second_threshold = np.where(blurred > th2, 255, 0).astype(np.uint8)
    
    # Operações morfológicas para limpar a imagem
    kernel = np.ones((2,2), np.uint8)
    morph = cv2.morphologyEx(second_threshold, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    
    # Segundo filtro Gaussiano
    final_blurred = gaussian_filter(morph, sigma=sigma2)
    final_threshold = np.where(final_blurred > th2, 255, 0).astype(np.uint8)
    
    # Aplica mais filtros para melhorar a nitidez
    processed_img = Image.fromarray(final_threshold)
    processed_img = processed_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    processed_img = processed_img.filter(ImageFilter.SHARPEN)
    processed_img = processed_img.filter(ImageFilter.UnsharpMask(radius=2, percent=150))
    
    # Remove linhas horizontais e verticais
    img_array = np.array(processed_img)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    detected_lines = cv2.morphologyEx(255 - img_array, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img_array, [c], -1, (255,255,255), 3)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    detected_lines = cv2.morphologyEx(255 - img_array, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(img_array, [c], -1, (255,255,255), 3)
        
    # Adiciona técnicas avançadas de remoção de ruído
    bilateral = cv2.bilateralFilter(img_array, 9, 75, 75)
    denoised = cv2.fastNlMeansDenoising(bilateral, None, 10, 7, 21)
    
    # Aplica detecção de bordas para realçar caracteres
    edges = cv2.Canny(denoised, 100, 200)
    dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    
    # Combina a imagem original com as bordas realçadas
    combined = cv2.bitwise_and(denoised, denoised, mask=255-dilated_edges)
    
    return Image.fromarray(combined)

def extract_text_from_image(image, ocr_configs):
    """Extrai texto da imagem usando múltiplas configurações do Tesseract OCR."""
    results = []
    
    # Adiciona escala da imagem para melhorar OCR
    w, h = image.size
    scaled_image = image.resize((w*2, h*2), Image.BICUBIC)
    
    for config in ocr_configs:
        text = pytesseract.image_to_string(scaled_image, config=config)
        # Remove espaços e caracteres de nova linha
        text = text.strip().replace(" ", "").replace("\n", "")
        # Filtra para manter apenas caracteres alfanuméricos
        text = ''.join(c for c in text if c.isalnum())
        if 3 <= len(text) <= 8:  # Comprimento típico de CAPTCHAs
            results.append(text)
            
        # Tenta também com a imagem original
        text = pytesseract.image_to_string(image, config=config)
        text = text.strip().replace(" ", "").replace("\n", "")
        text = ''.join(c for c in text if c.isalnum())
        if 3 <= len(text) <= 8:
            results.append(text)
    
    # Se temos resultados, pegamos o mais comum ou o primeiro
    if results:
        most_common = Counter(results).most_common(1)
        return most_common[0][0] if most_common else results[0]
    return ""

def solve_captcha_local(th1, th2, sigma1, sigma2, image_data_base64):
    """Resolve o CAPTCHA aplicando preprocessamento avançado e OCR."""
    try:
        # Decodifica a imagem base64
        original_image = Image.open(BytesIO(base64.b64decode(image_data_base64)))
        processed_image = preprocess_image(original_image, th1, th2, sigma1, sigma2)
        
        # Configurações do OCR - testamos várias para maximizar acertos
        ocr_configs = [
            '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
            '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
            '--psm 11 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
            '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
            # Novas configurações
            '--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
            '--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
            '--psm 9 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
            '--psm 12 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz',
            # Variações com diferentes whitelist characters
            '--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
        ]
        
        return extract_text_from_image(processed_image, ocr_configs)
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
                        
                        # Terminal output similar ao tester_advanced_filters.py
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
    
    return results, best_params, best_accuracy, total_images

# Configurações e execução
if __name__ == "__main__":
    image_folder = r'C:\Users\IsraelAntunes\Desktop\grid-search-main\image'
    
    # Parâmetros atualizados conforme solicitado
    param_ranges = {
        "th1": [99,100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "th2": [140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190],
        "sigma1": [0.75, 0.85, 0.95, 1.05, 1.15, 1.25, 1.35, 1.5, 1.65, 1.75, 1.85],
        "sigma2": [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8],
    }
    
    # Cria um nome de arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"./results/results_{timestamp}.csv"
    
    # Cria o diretório de resultados se não existir
    os.makedirs("./results", exist_ok=True)
    
    print("\n" + "*"*80)
    print("* GRID SEARCH - DETECTOR SIMPLES DE CAPTCHAS")
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
    results, best_params, best_accuracy, total_images = grid_search_on_images(image_folder, param_ranges, results_file)
    
    # Salva os melhores parâmetros em um arquivo separado
    with open(f"./results/best_params_{timestamp}.txt", "w") as f:
        f.write(f"Best parameters: th1={best_params[0]}, th2={best_params[1]}, " +
                f"sigma1={best_params[2]}, sigma2={best_params[3]}\n")
        f.write(f"Accuracy: {best_accuracy}/{total_images} ({best_accuracy/total_images:.1%})")
