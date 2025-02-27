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
    
    return Image.fromarray(img_array)

def extract_text_from_image(image, ocr_configs):
    """Extrai texto da imagem usando múltiplas configurações do Tesseract OCR."""
    results = []
    
    for config in ocr_configs:
        text = pytesseract.image_to_string(image, config=config)
        # Remove espaços e caracteres de nova linha
        text = text.strip().replace(" ", "").replace("\n", "")
        # Filtra para manter apenas caracteres alfanuméricos
        text = ''.join(c for c in text if c.isalnum())
        if 3 <= len(text) <= 8:  # Comprimento típico de CAPTCHAs
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
            '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz'
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
                    
                    # Para cada combinação de parâmetros, testa todas as imagens
                    for img_file in image_files:
                        expected = os.path.splitext(img_file)[0]  # Nome do arquivo sem extensão
                        prediction = solve_captcha_local(
                            th1, th2, sigma1, sigma2, image_data[img_file]
                        )
                        scores[expected] = 1 if prediction.lower() == expected.lower() else 0
                    
                    # Calcula a acurácia
                    accuracy = sum(scores.values())
                    results[(th1, th2, sigma1, sigma2)] = scores
                    
                    # Salva os melhores parâmetros
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = (th1, th2, sigma1, sigma2)
                    
                    # Exibe progresso
                    elapsed_time = time.time() - start_time
                    progress = current_combination / total_combinations
                    estimated_total_time = elapsed_time / progress if progress > 0 else 0
                    remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
                    
                    print(f"Progresso: {current_combination}/{total_combinations} " +
                          f"({progress:.1%}) - Tempo restante: {remaining_time/60:.1f} min")
                    print(f"Parâmetros: {th1}, {th2}, {sigma1}, {sigma2} - " +
                          f"Acertos: \033[1;{'32' if accuracy >= total_images/2 else '31'}m{accuracy}/{total_images}\033[0m " +
                          f"({accuracy/total_images:.1%})")
                    
                    # Salva resultados parciais periodicamente
                    if current_combination % 10 == 0:
                        print("Salvando resultados parciais...")
                        pd.DataFrame.from_dict(results, orient="index").to_csv(results_file)
    
    # Exibe os melhores resultados
    print(f"\nMelhores parâmetros encontrados: {best_params}")
    print(f"Melhor acurácia: {best_accuracy}/{total_images} ({best_accuracy/total_images:.1%})")
    
    # Salva os resultados finais
    pd.DataFrame.from_dict(results, orient="index").to_csv(results_file)
    print(f"Resultados salvos em '{results_file}'")
    
    return results, best_params, best_accuracy

# Configurações e execução
if __name__ == "__main__":
    image_folder = r'C:\Users\IsraelAntunes\Desktop\grid-search-main\image'
    
    # Parâmetros mais amplos e granulares para encontrar melhores combinações
    param_ranges = {
        "th1": [160, 170, 175, 180, 185, 190, 195, 200],
        "th2": [135, 140, 145, 150, 155, 160],
        "sigma1": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
        "sigma2": [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
    }
    
    # Cria um nome de arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"./results/results_{timestamp}.csv"
    
    # Cria o diretório de resultados se não existir
    os.makedirs("./results", exist_ok=True)
    
    # Executa a busca em grade
    results, best_params, best_accuracy = grid_search_on_images(image_folder, param_ranges, results_file)
    
    # Salva os melhores parâmetros em um arquivo separado
    with open(f"./results/best_params_{timestamp}.txt", "w") as f:
        f.write(f"Best parameters: th1={best_params[0]}, th2={best_params[1]}, " +
                f"sigma1={best_params[2]}, sigma2={best_params[3]}\n")
        f.write(f"Accuracy: {best_accuracy}")