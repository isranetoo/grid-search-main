import cv2
import numpy as np
import os
import glob
from pathlib import Path
import time
from datetime import datetime
import sys
from scipy.ndimage import gaussian_filter
from collections import Counter

# Check if OpenCV has GUI support
def has_gui_support():
    """Verifica se o OpenCV possui suporte a GUI."""
    try:
        cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
        cv2.destroyWindow('Test')
        return True
    except cv2.error:
        return False

# Global variable to track GUI support
GUI_SUPPORTED = has_gui_support()
if not GUI_SUPPORTED:
    print("\033[1;33mAVISO: O OpenCV foi compilado sem suporte a GUI. Alguns recursos visuais não estarão disponíveis.\033[0m")
    print("Para corrigir, reinstale o OpenCV com suporte a GUI ou use uma versão diferente.")


class ColorRangeTester:
    def __init__(self, image_dir='./image', results_dir='./results/color_filters'):
        """Inicializa o testador de faixas de cores.
        
        Args:
            image_dir: Diretório contendo as imagens a serem processadas
            results_dir: Diretório para salvar os resultados
        """
        self.image_dir = image_dir
        self.results_dir = results_dir
        self.ensure_dirs_exist()
        
    def ensure_dirs_exist(self):
        """Garante que os diretórios necessários existam."""
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        
    def filter_image(self, image_path, lower_hsv, upper_hsv, color_name='custom'):
        """Filtra uma imagem usando uma faixa de cores HSV.
        
        Args:
            image_path: Caminho para a imagem
            lower_hsv: Limite inferior da faixa HSV como array [H, S, V]
            upper_hsv: Limite superior da faixa HSV como array [H, S, V]
            color_name: Nome da cor ou faixa para organizar os resultados
            
        Returns:
            Tuple contendo a imagem original, máscara e resultado
        """
        # Carrega a imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"\033[1;31mErro ao carregar a imagem: {image_path}\033[0m")
            return None, None, None
        
        # Converte para HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Cria a máscara para a faixa de cores
        mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
        
        # Aplica a máscara na imagem original
        result = cv2.bitwise_and(img, img, mask=mask)
        
        # Cria um diretório específico para a cor
        color_dir = os.path.join(self.results_dir, color_name)
        Path(color_dir).mkdir(parents=True, exist_ok=True)
        
        # Extrai o nome base da imagem
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Salva os resultados
        cv2.imwrite(os.path.join(color_dir, f"{base_name}_original.jpg"), img)
        cv2.imwrite(os.path.join(color_dir, f"{base_name}_mask.jpg"), mask)
        cv2.imwrite(os.path.join(color_dir, f"{base_name}_result.jpg"), result)
        
        return img, mask, result

    def process_batch(self, file_pattern="*.png", lower_hsv=[0, 0, 0], upper_hsv=[180, 255, 60], color_name='dark'):
        """Processa um lote de imagens com a mesma faixa de cores.
        
        Args:
            file_pattern: Padrão para encontrar arquivos (ex: "*.png")
            lower_hsv: Limite inferior da faixa HSV como array [H, S, V]
            upper_hsv: Limite superior da faixa HSV como array [H, S, V]
            color_name: Nome da cor ou faixa para organizar os resultados
            
        Returns:
            Número de imagens processadas com sucesso
        """
        # Lista todas as imagens que correspondem ao padrão
        image_paths = glob.glob(os.path.join(self.image_dir, file_pattern))
        
        if not image_paths:
            print(f"\033[1;31mNenhuma imagem encontrada com o padrão '{file_pattern}' em '{self.image_dir}'\033[0m")
            return 0
        
        success_count = 0
        total_images = len(image_paths)
        
        # Exibe informações iniciais
        print("\n" + "="*80)
        print(f"PROCESSAMENTO DE FILTRO: {color_name.upper()}")
        print(f"Parâmetros HSV: Inferior={lower_hsv}, Superior={upper_hsv}")
        print("="*80)
        
        start_time = time.time()
        
        for idx, image_path in enumerate(image_paths):
            # Status de progresso
            progress = (idx + 1) / total_images
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
            
            # Barra de progresso
            progress_bar_length = 40
            progress_filled = int(progress * progress_bar_length)
            progress_bar = "[" + "=" * progress_filled + " " * (progress_bar_length - progress_filled) + "]"
            
            sys.stdout.write(f"\r{progress_bar} {progress:.1%} | Processando: {os.path.basename(image_path):<20}")
            sys.stdout.flush()
            
            # Processa a imagem
            img, mask, result = self.filter_image(image_path, lower_hsv, upper_hsv, color_name)
            
            if img is not None:
                success_count += 1
        
        # Limpa a linha atual e exibe o resumo
        sys.stdout.write("\r" + " " * 100 + "\r")  # Limpa a linha
        
        print("\n" + "-"*80)
        print(f"RESUMO DO PROCESSAMENTO: {color_name.upper()}")
        print(f"Faixa HSV: Inferior={lower_hsv}, Superior={upper_hsv}")
        print(f"Imagens processadas: {success_count}/{total_images} ({success_count/total_images:.1%})")
        print(f"Tempo de execução: {(time.time() - start_time):.2f} segundos")
        print(f"Resultados salvos em: {self.results_dir}/{color_name}/")
        print("-"*80 + "\n")
        
        return success_count
    
    def show_test_window(self, image_path):
        """Exibe uma janela de teste interativa para ajustar os parâmetros HSV.
        Se o OpenCV não tiver suporte a GUI, oferece uma alternativa baseada em linha de comando.
        
        Args:
            image_path: Caminho para a imagem a ser usada no teste
        """
        if not os.path.exists(image_path):
            print(f"\033[1;31mImagem não encontrada: {image_path}\033[0m")
            return
        
        # Carrega a imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"\033[1;31mErro ao carregar a imagem: {image_path}\033[0m")
            return
        
        # Cabeçalho informativo
        print("\n" + "="*80)
        print(f"TESTE DE FILTROS HSV")
        print(f"Imagem: {os.path.basename(image_path)}")
        print("="*80)
        
        if GUI_SUPPORTED:
            # Versão interativa com GUI
            print("* Use os controles deslizantes para ajustar os valores HSV.")
            print("* Pressione 's' para salvar a configuração atual.")
            print("* Pressione 'q' para sair.")
            print("="*80 + "\n")
            
            # Cria janela e trackbars para ajuste
            cv2.namedWindow('HSV Filter Test')
            cv2.createTrackbar('H Min', 'HSV Filter Test', 0, 180, lambda x: None)
            cv2.createTrackbar('S Min', 'HSV Filter Test', 0, 255, lambda x: None)
            cv2.createTrackbar('V Min', 'HSV Filter Test', 0, 255, lambda x: None)
            cv2.createTrackbar('H Max', 'HSV Filter Test', 180, 180, lambda x: None)
            cv2.createTrackbar('S Max', 'HSV Filter Test', 255, 255, lambda x: None)
            cv2.createTrackbar('V Max', 'HSV Filter Test', 60, 255, lambda x: None)
            
            while True:
                # Lê os valores atuais dos trackbars
                h_min = cv2.getTrackbarPos('H Min', 'HSV Filter Test')
                s_min = cv2.getTrackbarPos('S Min', 'HSV Filter Test')
                v_min = cv2.getTrackbarPos('V Min', 'HSV Filter Test')
                h_max = cv2.getTrackbarPos('H Max', 'HSV Filter Test')
                s_max = cv2.getTrackbarPos('S Max', 'HSV Filter Test')
                v_max = cv2.getTrackbarPos('V Max', 'HSV Filter Test')
                
                # Define os limites HSV
                lower_hsv = np.array([h_min, s_min, v_min])
                upper_hsv = np.array([h_max, s_max, v_max])
                
                # Converte e aplica a máscara
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                result = cv2.bitwise_and(img, img, mask=mask)
                
                # Redimensiona as imagens para exibição
                scale = 0.8
                display_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                display_mask = cv2.resize(mask, (0, 0), fx=scale, fy=scale)
                display_result = cv2.resize(result, (0, 0), fx=scale, fy=scale)
                
                # Exibe as imagens
                cv2.imshow('Original', display_img)
                cv2.imshow('Mask', display_mask)
                cv2.imshow('Result', display_result)
                
                # Mostra os valores atuais no terminal
                sys.stdout.write(f"\rValores HSV atuais: Inferior=[{h_min}, {s_min}, {v_min}], Superior=[{h_max}, {s_max}, {v_max}]")
                sys.stdout.flush()
                
                # Aguarda entrada do usuário
                key = cv2.waitKey(100) & 0xFF
                
                # Se 's' for pressionado, salva os valores atuais
                if key == ord('s'):
                    print("\n\n" + "-"*80)
                    print(f"\033[1;32mValores HSV salvos: Inferior=[{h_min}, {s_min}, {v_min}], Superior=[{h_max}, {s_max}, {v_max}]\033[0m")
                    color_name = input("Digite um nome para esta faixa de cores: ")
                    if not color_name:
                        color_name = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    self.filter_image(image_path, lower_hsv.tolist(), upper_hsv.tolist(), color_name)
                    print(f"\033[1;32mResultados salvos em {self.results_dir}/{color_name}/\033[0m")
                    print("-"*80)
                
                # Se 'q' for pressionado, sai do loop
                if key == ord('q'):
                    break
                    
            # Fecha todas as janelas
            cv2.destroyAllWindows()
            print("\n\033[1;33mTeste interativo encerrado.\033[0m")
        else:
            # Versão alternativa baseada em linha de comando
            print("* GUI não disponível. Usando modo de entrada manual.")
            print("* Digite os valores HSV manualmente para testar filtros de cor.")
            print("="*80 + "\n")
            
            while True:
                print("\nDigite os valores HSV (ou 'q' para sair):")
                try:
                    h_min_input = input("H Min (0-180): ")
                    if h_min_input.lower() == 'q':
                        break
                    
                    h_min = int(h_min_input)
                    s_min = int(input("S Min (0-255): "))
                    v_min = int(input("V Min (0-255): "))
                    h_max = int(input("H Max (0-180): "))
                    s_max = int(input("S Max (0-255): "))
                    v_max = int(input("V Max (0-255): "))
                    
                    # Define os limites HSV
                    lower_hsv = np.array([h_min, s_min, v_min])
                    upper_hsv = np.array([h_max, s_max, v_max])
                    
                    # Converte e aplica a máscara
                    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
                    result = cv2.bitwise_and(img, img, mask=mask)
                    
                    # Salva os resultados temporários
                    temp_dir = os.path.join(self.results_dir, "temp_preview")
                    Path(temp_dir).mkdir(parents=True, exist_ok=True)
                    
                    cv2.imwrite(os.path.join(temp_dir, "original.jpg"), img)
                    cv2.imwrite(os.path.join(temp_dir, "mask.jpg"), mask)
                    cv2.imwrite(os.path.join(temp_dir, "result.jpg"), result)
                    
                    print(f"\n\033[1;32mResultados temporários gerados. Verifique os arquivos em {temp_dir}/\033[0m")
                    print(f"Valores HSV atuais: Inferior=[{h_min}, {s_min}, {v_min}], Superior=[{h_max}, {s_max}, {v_max}]")
                    
                    # Pergunta se deseja salvar os resultados
                    save = input("\nDeseja salvar esta configuração? (s/n): ")
                    if save.lower() == 's':
                        color_name = input("Digite um nome para esta faixa de cores: ")
                        if not color_name:
                            color_name = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        self.filter_image(image_path, lower_hsv.tolist(), upper_hsv.tolist(), color_name)
                        print(f"\033[1;32mResultados salvos em {self.results_dir}/{color_name}/\033[0m")
                except ValueError:
                    print("\033[1;31mEntrada inválida. Use apenas números.\033[0m")
                except Exception as e:
                    print(f"\033[1;31mErro ao processar: {str(e)}\033[0m")
            
            print("\n\033[1;33mTeste manual encerrado.\033[0m")

    def remove_background(self, image_path, bgcolor=[255, 255, 255], output_dir=None):
        """Remove o fundo da imagem baseado em uma cor específica.
        
        Args:
            image_path: Caminho para a imagem
            bgcolor: Cor do fundo a ser removido em BGR (ex: [255, 255, 255] para branco)
            output_dir: Diretório para salvar a imagem resultante (se None, usa o results_dir)
            
        Returns:
            Caminho da imagem resultante
        """
        if not os.path.exists(image_path):
            print(f"\033[1;31mImagem não encontrada: {image_path}\033[0m")
            return None
            
        # Define o diretório de saída
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "background_removed")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extrai o nome base da imagem
        base_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}_nobg.png")
        
        # Carrega a imagem e converte para BGRA
        img = cv2.imread(image_path)
        if img is None:
            print(f"\033[1;31mErro ao carregar a imagem: {image_path}\033[0m")
            return None
            
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # Remove o fundo substituindo a cor especificada por transparência
        mask = np.all(img == bgcolor, axis=2)
        img_bgra[mask] = [0, 0, 0, 0]
        
        # Salva a imagem resultante
        cv2.imwrite(output_path, img_bgra)
        
        print(f"\033[1;32mFundo removido com sucesso. Imagem salva em: {output_path}\033[0m")
        return output_path
        
    def batch_remove_background(self, file_pattern="*.png", bgcolor=[255, 255, 255]):
        """Remove o fundo de várias imagens em lote.
        
        Args:
            file_pattern: Padrão para encontrar arquivos (ex: "*.png")
            bgcolor: Cor do fundo a ser removido em BGR (ex: [255, 255, 255] para branco)
            
        Returns:
            Número de imagens processadas com sucesso
        """
        # Lista todas as imagens que correspondem ao padrão
        image_paths = glob.glob(os.path.join(self.image_dir, file_pattern))
        
        if not image_paths:
            print(f"\033[1;31mNenhuma imagem encontrada com o padrão '{file_pattern}' em '{self.image_dir}'\033[0m")
            return 0
            
        output_dir = os.path.join(self.results_dir, "background_removed")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        total_images = len(image_paths)
        
        # Exibe informações iniciais
        print("\n" + "="*80)
        print(f"PROCESSAMENTO DE REMOÇÃO DE FUNDO")
        print(f"Cor de fundo a ser removida (BGR): {bgcolor}")
        print("="*80)
        
        start_time = time.time()
        
        for idx, image_path in enumerate(image_paths):
            # Status de progresso
            progress = (idx + 1) / total_images
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
            
            # Barra de progresso
            progress_bar_length = 40
            progress_filled = int(progress * progress_bar_length)
            progress_bar = "[" + "=" * progress_filled + " " * (progress_bar_length - progress_filled) + "]"
            
            sys.stdout.write(f"\r{progress_bar} {progress:.1%} | Processando: {os.path.basename(image_path):<20}")
            sys.stdout.flush()
            
            # Remove o fundo da imagem
            result_path = self.remove_background(image_path, bgcolor, output_dir)
            
            if result_path is not None:
                success_count += 1
        
        # Limpa a linha atual e exibe o resumo
        sys.stdout.write("\r" + " " * 100 + "\r")  # Limpa a linha
        
        print("\n" + "-"*80)
        print(f"RESUMO DO PROCESSAMENTO DE REMOÇÃO DE FUNDO")
        print(f"Cor de fundo removida (BGR): {bgcolor}")
        print(f"Imagens processadas: {success_count}/{total_images} ({success_count/total_images:.1%})")
        print(f"Tempo de execução: {(time.time() - start_time):.2f} segundos")
        print(f"Resultados salvos em: {output_dir}/")
        print("-"*80 + "\n")
        
        return success_count
        
    def detect_border_colors(self, image_path, border_width=5, tolerance=20, samples=100):
        """Detecta as cores predominantes nas bordas da imagem.
        
        Args:
            image_path: Caminho para a imagem
            border_width: Largura da borda a ser analisada em pixels
            tolerance: Tolerância para agrupar cores similares
            samples: Número máximo de amostras a serem analisadas por borda
            
        Returns:
            List de tuples (BGR color, frequency) ordenados por frequência
        """
        if not os.path.exists(image_path):
            print(f"\033[1;31mImagem não encontrada: {image_path}\033[0m")
            return []
            
        # Carrega a imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"\033[1;31mErro ao carregar a imagem: {image_path}\033[0m")
            return []
        
        height, width = img.shape[:2]
        border_pixels = []
        
        # Extrai pixels da borda superior
        step_x = max(1, width // samples)
        for x in range(0, width, step_x):
            for y in range(0, min(border_width, height)):
                border_pixels.append(tuple(img[y, x]))
        
        # Extrai pixels da borda inferior
        for x in range(0, width, step_x):
            for y in range(max(0, height - border_width), height):
                border_pixels.append(tuple(img[y, x]))
        
        # Extrai pixels da borda esquerda
        step_y = max(1, height // samples)
        for y in range(0, height, step_y):
            for x in range(0, min(border_width, width)):
                border_pixels.append(tuple(img[y, x]))
        
        # Extrai pixels da borda direita
        for y in range(0, height, step_y):
            for x in range(max(0, width - border_width), width):
                border_pixels.append(tuple(img[y, x]))
        
        # Agrupa cores similares
        grouped_colors = {}
        for color in border_pixels:
            found_similar = False
            for base_color in grouped_colors.keys():
                # Verifica se as cores são similares dentro da tolerância
                if (abs(int(color[0]) - int(base_color[0])) <= tolerance and
                    abs(int(color[1]) - int(base_color[1])) <= tolerance and
                    abs(int(color[2]) - int(base_color[2])) <= tolerance):
                    grouped_colors[base_color] += 1
                    found_similar = True
                    break
            
            if not found_similar:
                grouped_colors[color] = 1
        
        # Ordena as cores por frequência
        sorted_colors = sorted(grouped_colors.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\033[1;32mDetectadas {len(sorted_colors)} cores distintas na borda.\033[0m")
        print(f"Top 5 cores (BGR, frequência): {sorted_colors[:5]}")
        
        return sorted_colors
    
    def display_border_colors(self, image_path, num_colors=5, border_width=5, tolerance=20, samples=100):
        """Detecta e exibe visualmente as cores predominantes nas bordas da imagem.
        Se o OpenCV não tiver suporte a GUI, apenas mostrará informações no console.
        
        Args:
            image_path: Caminho para a imagem
            num_colors: Número de cores principais a exibir
            border_width: Largura da borda a ser analisada em pixels
            tolerance: Tolerância para agrupar cores similares
            samples: Número máximo de amostras a serem analisadas por borda
            
        Returns:
            Lista das cores de borda predominantes como tuples (BGR color, frequency)
        """
        if not os.path.exists(image_path):
            print(f"\033[1;31mImagem não encontrada: {image_path}\033[0m")
            return []
            
        # Carrega a imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"\033[1;31mErro ao carregar a imagem: {image_path}\033[0m")
            return []
            
        print("\n" + "="*80)
        print(f"DETECÇÃO AUTOMÁTICA DE CORES DE BORDA")
        print(f"Imagem: {os.path.basename(image_path)}")
        print("="*80)
        
        # Detecta as cores da borda
        border_colors = self.detect_border_colors(image_path, border_width, tolerance, samples)
        
        if not border_colors:
            print(f"\033[1;31mNenhuma cor de borda detectada.\033[0m")
            return []
            
        # Limita ao número de cores especificado
        top_colors = border_colors[:num_colors]
        
        # Exibe informações textuais sobre as cores detectadas
        print("\nCores de borda predominantes (BGR, frequência, porcentagem):")
        total_freq = sum(freq for _, freq in top_colors)
        
        for i, (color, freq) in enumerate(top_colors):
            percentage = freq / total_freq * 100
            print(f"{i+1}. Color: {color}, Freq: {freq}, {percentage:.1f}%")
        
        # Se o OpenCV tiver suporte a GUI, mostra visualização
        if GUI_SUPPORTED:
            # Cria uma imagem para visualização das cores de borda
            display_height = 100
            display_width = 600
            color_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            
            # Determina a largura de cada bloco de cor
            block_width = display_width // min(len(top_colors), num_colors)
            
            # Preenche a imagem com os blocos de cores detectados
            for i, (color, freq) in enumerate(top_colors):
                if i >= num_colors:
                    break
                    
                # Desenha o bloco de cor
                start_x = i * block_width
                end_x = (i + 1) * block_width if i < len(top_colors) - 1 else display_width
                color_display[:, start_x:end_x] = color
                
                # Adiciona texto com informações da cor
                color_text = f"BGR: {color}"
                pct_text = f"{(freq / total_freq * 100):.1f}%"
                
                # Ajusta a cor do texto conforme a luminosidade da cor de fundo
                brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
                text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)
                
                # Adiciona o texto à imagem
                cv2.putText(color_display, color_text, 
                          (start_x + 5, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
                cv2.putText(color_display, pct_text, 
                          (start_x + 5, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
            # Exibe a imagem com as cores detectadas
            cv2.imshow('Cores de Borda Detectadas', color_display)
            
            # Também destaca as bordas na imagem original
            border_highlight = img.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            # Marca os pixels de borda
            h, w = img.shape[:2]
            mask[:border_width, :] = 255  # Borda superior
            mask[h-border_width:, :] = 255  # Borda inferior
            mask[:, :border_width] = 255  # Borda esquerda
            mask[:, w-border_width:] = 255  # Borda direita
            
            # Aplica uma cor de destaque nas bordas
            highlight_color = (0, 255, 255)  # Amarelo em BGR
            border_highlight[mask == 255] = highlight_color
            
            # Cria uma imagem que destaca onde as cores de borda aparecem em toda a imagem
            color_mask = np.zeros(img.shape, dtype=np.uint8)
            
            for color, _ in top_colors:
                # Cria uma máscara para a cor atual com a tolerância especificada
                lower_bound = np.array([max(0, c - tolerance) for c in color])
                upper_bound = np.array([min(255, c + tolerance) for c in color])
                
                color_mask_current = cv2.inRange(img, lower_bound, upper_bound)
                color_mask[color_mask_current > 0] = highlight_color
            
            # Combina a imagem original com a máscara de cores
            alpha = 0.7
            color_highlighted = cv2.addWeighted(img, alpha, color_mask, 1-alpha, 0)
            
            # Redimensiona para exibição
            scale = min(1.0, 800 / max(img.shape[1], 1))
            display_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            display_border_highlight = cv2.resize(border_highlight, (0, 0), fx=scale, fy=scale)
            display_color_highlighted = cv2.resize(color_highlighted, (0, 0), fx=scale, fy=scale)
            
            # Exibe as imagens
            cv2.imshow('Imagem Original', display_img)
            cv2.imshow('Bordas da Imagem', display_border_highlight)
            cv2.imshow('Ocorrência das Cores de Borda', display_color_highlighted)
            
            print("\n\033[1;32mCores de borda detectadas e exibidas.\033[0m")
            print("Pressione qualquer tecla nas janelas de imagem para continuar...")
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            # Versão alternativa quando não há suporte a GUI
            print("\n\033[1;33mVisualização não disponível (sem suporte a GUI).\033[0m")
            
            # Salva as visualizações em arquivos para referência
            output_dir = os.path.join(self.results_dir, "border_colors")
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            
            base_name = os.path.basename(image_path).split('.')[0]
            
            # Salva uma imagem com as cores detectadas
            display_height = 100
            display_width = 600
            color_display = np.zeros((display_height, display_width, 3), dtype=np.uint8)
            
            block_width = display_width // min(len(top_colors), num_colors)
            
            for i, (color, freq) in enumerate(top_colors):
                if i >= num_colors:
                    break
                
                # Desenha o bloco de cor
                start_x = i * block_width
                end_x = (i + 1) * block_width if i < len(top_colors) - 1 else display_width
                color_display[:, start_x:end_x] = color
            
            # Salva a visualização das cores
            color_display_path = os.path.join(output_dir, f"{base_name}_border_colors.png")
            cv2.imwrite(color_display_path, color_display)
            
            # Destaca as bordas e salva
            border_highlight = img.copy()
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            
            h, w = img.shape[:2]
            mask[:border_width, :] = 255  # Borda superior
            mask[h-border_width:, :] = 255  # Borda inferior
            mask[:, :border_width] = 255  # Borda esquerda
            mask[:, w-border_width:] = 255  # Borda direita
            
            highlight_color = (0, 255, 255)
            border_highlight[mask == 255] = highlight_color
            
            border_highlight_path = os.path.join(output_dir, f"{base_name}_border_highlight.png")
            cv2.imwrite(border_highlight_path, border_highlight)
            
            # Cria e salva a máscara de cores na imagem
            color_mask = np.zeros(img.shape, dtype=np.uint8)
            
            for color, _ in top_colors:
                lower_bound = np.array([max(0, c - tolerance) for c in color])
                upper_bound = np.array([min(255, c + tolerance) for c in color])
                
                color_mask_current = cv2.inRange(img, lower_bound, upper_bound)
                color_mask[color_mask_current > 0] = highlight_color
            
            alpha = 0.7
            color_highlighted = cv2.addWeighted(img, alpha, color_mask, 1-alpha, 0)
            
            color_highlight_path = os.path.join(output_dir, f"{base_name}_color_highlight.png")
            cv2.imwrite(color_highlight_path, color_highlighted)
            
            print(f"\n\033[1;32mVisualizações salvas em {output_dir}/\033[0m")
        
        return top_colors
        
    def auto_remove_border_colors(self, image_path, output_dir=None):
        """Remove automaticamente as cores de borda da imagem sem intervenção do usuário.
        
        Args:
            image_path: Caminho para a imagem
            output_dir: Diretório para salvar a imagem resultante (se None, usa o results_dir)
            
        Returns:
            Caminho da imagem resultante
        """
        # Parâmetros otimizados para detecção automática
        border_width = 5
        tolerance = 15
        samples = 150
        num_colors = 2  # Número de cores de borda principais a remover
        
        print("\n" + "="*80)
        print(f"REMOÇÃO AUTOMÁTICA DE CORES DE BORDA")
        print(f"Imagem: {os.path.basename(image_path)}")
        print("="*80)
        
        # Detecta as cores da borda
        border_colors = self.detect_border_colors(image_path, border_width, tolerance, samples)
        
        if not border_colors:
            print(f"\033[1;31mNenhuma cor de borda detectada. Não foi possível processar a imagem.\033[0m")
            return None
            
        # Usa a função existente para remover as cores detectadas
        result_path = self.remove_border_colors(image_path, num_colors, tolerance, border_width, samples, output_dir)
        
        if result_path:
            print(f"\033[1;32mCores de borda removidas automaticamente com sucesso!\033[0m")
            
        return result_path
        
    def batch_auto_remove_border_colors(self, file_pattern="*.png"):
        """Remove automaticamente as cores de borda de várias imagens em lote.
        
        Args:
            file_pattern: Padrão para encontrar arquivos (ex: "*.png")
            
        Returns:
            Número de imagens processadas com sucesso
        """
        print("\n" + "="*80)
        print(f"REMOÇÃO AUTOMÁTICA DE CORES DE BORDA EM LOTE")
        print(f"Padrão de arquivo: {file_pattern}")
        print("="*80)
        
        # Lista todas as imagens que correspondem ao padrão
        image_paths = glob.glob(os.path.join(self.image_dir, file_pattern))
        
        if not image_paths:
            print(f"\033[1;31mNenhuma imagem encontrada com o padrão '{file_pattern}' em '{self.image_dir}'\033[0m")
            return 0
            
        output_dir = os.path.join(self.results_dir, "auto_border_removed")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        total_images = len(image_paths)
        
        start_time = time.time()
        
        for idx, image_path in enumerate(image_paths):
            # Status de progresso
            progress = (idx + 1) / total_images
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
            
            # Barra de progresso
            progress_bar_length = 40
            progress_filled = int(progress * progress_bar_length)
            progress_bar = "[" + "=" * progress_filled + " " * (progress_bar_length - progress_filled) + "]"
            
            sys.stdout.write(f"\r{progress_bar} {progress:.1%} | Processando: {os.path.basename(image_path):<20}")
            sys.stdout.flush()
            
            # Remove automaticamente as cores de borda da imagem
            result_path = self.auto_remove_border_colors(image_path, output_dir)
            
            if result_path is not None:
                success_count += 1
        
        # Limpa a linha atual e exibe o resumo
        sys.stdout.write("\r" + " " * 100 + "\r")  # Limpa a linha
        
        print("\n" + "-"*80)
        print(f"RESUMO DO PROCESSAMENTO DE REMOÇÃO AUTOMÁTICA DE CORES DE BORDA")
        print(f"Imagens processadas: {success_count}/{total_images} ({success_count/total_images:.1%})")
        print(f"Tempo de execução: {(time.time() - start_time):.2f} segundos")
        print(f"Resultados salvos em: {output_dir}/")
        print("-"*80 + "\n")
        
        return success_count
    
    def remove_border_colors(self, image_path, num_colors=1, tolerance=20, border_width=5, samples=100, output_dir=None):
        """Remove as cores predominantes nas bordas da imagem.
        
        Args:
            image_path: Caminho para a imagem
            num_colors: Número de cores de borda a serem removidas (começando pelas mais frequentes)
            tolerance: Tolerância para agrupar cores similares
            border_width: Largura da borda a ser analisada em pixels
            samples: Número máximo de amostras a serem analisadas por borda
            output_dir: Diretório para salvar a imagem resultante (se None, usa o results_dir)
            
        Returns:
            Caminho da imagem resultante
        """
        if not os.path.exists(image_path):
            print(f"\033[1;31mImagem não encontrada: {image_path}\033[0m")
            return None
            
        # Define o diretório de saída
        if output_dir is None:
            output_dir = os.path.join(self.results_dir, "border_removed")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extrai o nome base da imagem
        base_name = os.path.basename(image_path).split('.')[0]
        output_path = os.path.join(output_dir, f"{base_name}_noborder.png")
        
        # Carrega a imagem e converte para BGRA
        img = cv2.imread(image_path)
        if img is None:
            print(f"\033[1;31mErro ao carregar a imagem: {image_path}\033[0m")
            return None
        
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # Detecta as cores da borda
        border_colors = self.detect_border_colors(image_path, border_width, tolerance, samples)
        
        if not border_colors:
            print(f"\033[1;31mNenhuma cor de borda detectada.\033[0m")
            return None
        
        # Limita ao número de cores especificado
        border_colors = border_colors[:num_colors]
        
        print(f"\033[1;32mRemovendo {len(border_colors)} cor(es) de borda mais frequente(s).\033[0m")
        
        # Para cada cor da borda detectada, cria uma máscara
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        
        for color, _ in border_colors:
            # Cria uma máscara para a cor atual com a tolerância especificada
            lower_bound = np.array([max(0, c - tolerance) for c in color])
            upper_bound = np.array([min(255, c + tolerance) for c in color])
            
            mask = cv2.inRange(img, lower_bound, upper_bound)
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Inverte a máscara para manter apenas o que não é cor de borda
        inverted_mask = cv2.bitwise_not(combined_mask)
        
        # Aplica a máscara invertida na imagem BGRA
        alpha_channel = np.ones(img.shape[:2], dtype=np.uint8) * 255
        alpha_channel[combined_mask > 0] = 0
        img_bgra[:, :, 3] = alpha_channel
        
        # Salva a imagem resultante com transparência
        cv2.imwrite(output_path, img_bgra)
        
        print(f"\033[1;32mCores de borda removidas com sucesso. Imagem salva em: {output_path}\033[0m")
        
        # Também cria uma versão com fundo branco para visualização
        bg_version = img.copy()
        bg_version[combined_mask > 0] = [255, 255, 255]  # Branco onde havia cor de borda
        bg_output_path = os.path.join(output_dir, f"{base_name}_whitebg.png")
        cv2.imwrite(bg_output_path, bg_version)
        
        return output_path
    
    def batch_remove_border_colors(self, file_pattern="*.png", num_colors=1, tolerance=20, border_width=5, samples=100):
        """Remove as cores de borda de várias imagens em lote.
        
        Args:
            file_pattern: Padrão para encontrar arquivos (ex: "*.png")
            num_colors: Número de cores de borda a serem removidas (começando pelas mais frequentes)
            tolerance: Tolerância para agrupar cores similares
            border_width: Largura da borda a ser analisada em pixels
            samples: Número máximo de amostras a serem analisadas por borda
            
        Returns:
            Número de imagens processadas com sucesso
        """
        # Lista todas as imagens que correspondem ao padrão
        image_paths = glob.glob(os.path.join(self.image_dir, file_pattern))
        
        if not image_paths:
            print(f"\033[1;31mNenhuma imagem encontrada com o padrão '{file_pattern}' em '{self.image_dir}'\033[0m")
            return 0
            
        output_dir = os.path.join(self.results_dir, "border_removed")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        success_count = 0
        total_images = len(image_paths)
        
        # Exibe informações iniciais
        print("\n" + "="*80)
        print(f"PROCESSAMENTO DE REMOÇÃO DE CORES DE BORDA")
        print(f"Cores a remover: {num_colors} | Tolerância: {tolerance} | Largura da borda: {border_width}px")
        print("="*80)
        
        start_time = time.time()
        
        for idx, image_path in enumerate(image_paths):
            # Status de progresso
            progress = (idx + 1) / total_images
            elapsed_time = time.time() - start_time
            estimated_total_time = elapsed_time / progress if progress > 0 else 0
            remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
            
            # Barra de progresso
            progress_bar_length = 40
            progress_filled = int(progress * progress_bar_length)
            progress_bar = "[" + "=" * progress_filled + " " * (progress_bar_length - progress_filled) + "]"
            
            sys.stdout.write(f"\r{progress_bar} {progress:.1%} | Processando: {os.path.basename(image_path):<20}")
            sys.stdout.flush()
            
            # Remove as cores de borda da imagem
            result_path = self.remove_border_colors(image_path, num_colors, tolerance, border_width, samples, output_dir)
            
            if result_path is not None:
                success_count += 1
        
        # Limpa a linha atual e exibe o resumo
        sys.stdout.write("\r" + " " * 100 + "\r")  # Limpa a linha
        
        print("\n" + "-"*80)
        print(f"RESUMO DO PROCESSAMENTO DE REMOÇÃO DE CORES DE BORDA")
        print(f"Cores removidas por imagem: {num_colors} | Tolerância: {tolerance}")
        print(f"Imagens processadas: {success_count}/{total_images} ({success_count/total_images:.1%})")
        print(f"Tempo de execução: {(time.time() - start_time):.2f} segundos")
        print(f"Resultados salvos em: {output_dir}/")
        print("-"*80 + "\n")
        
        return success_count

    def apply_multiple_filters(self, image_path, color_ranges=None, output_name=None, remove_mode=False):
        """Aplica múltiplos filtros de cor sequencialmente à mesma imagem e combina os resultados.
        
        Args:
            image_path: Caminho para a imagem
            color_ranges: Dicionário de faixas de cores no formato {nome: ([h_min, s_min, v_min], [h_max, s_max, v_max])}
                         Se None, serão usadas algumas faixas de cores predefinidas
            output_name: Nome para o arquivo de saída. Se None, será gerado automaticamente
            remove_mode: Se True, remove as cores especificadas ao invés de destacá-las
            
        Returns:
            Tuple contendo (caminho do resultado combinado, lista de caminhos para resultados individuais)
        """
        if not os.path.exists(image_path):
            print(f"\033[1;31mImagem não encontrada: {image_path}\033[0m")
            return None, []
            
        # Carrega a imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"\033[1;31mErro ao carregar a imagem: {image_path}\033[0m")
            return None, []
            
        # Se não foi fornecido um conjunto de cores, usa faixas predefinidas
        if color_ranges is None:
            if remove_mode:
                # Cores padrão para remoção conforme solicitado: azul, vermelho, verde, amarelo, rosa, branco e laranja
                color_ranges = {
                    'vermelho1': ([0, 100, 100], [10, 255, 255]),
                    'vermelho2': ([160, 100, 100], [180, 255, 255]),
                    'verde': ([40, 100, 100], [80, 255, 255]),
                    'azul': ([100, 100, 100], [130, 255, 255]),
                    'amarelo': ([25, 100, 100], [35, 255, 255]),
                    'rosa': ([145, 50, 100], [165, 255, 255]),
                    'branco': ([0, 0, 220], [180, 30, 255]),
                    'laranja': ([10, 100, 100], [25, 255, 255]), 
                }
            else:
                # Cores padrão para filtro normal
                color_ranges = {
                    'red': ([0, 100, 100], [10, 255, 255]),
                    'blue': ([100, 100, 100], [130, 255, 255]),
                    'yellow': ([25, 100, 100], [35, 255, 255]),
                }
            
        # Cria um diretório específico para resultados múltiplos
        dir_name = "multi_filter_removed" if remove_mode else "multi_filter"
        multi_filter_dir = os.path.join(self.results_dir, dir_name)
        Path(multi_filter_dir).mkdir(parents=True, exist_ok=True)
        
        # Extrai o nome base da imagem
        base_name = os.path.basename(image_path).split('.')[0]
        if output_name is None:
            color_names = "_".join(color_ranges.keys())
            prefix = "removed_" if remove_mode else "multi_"
            output_name = f"{base_name}_{prefix}{color_names}"
        
        # Exibe informações iniciais
        mode_text = "REMOVENDO" if remove_mode else "APLICANDO"
        print("\n" + "="*80)
        print(f"{mode_text} MÚLTIPLOS FILTROS DE COR")
        print(f"Imagem: {os.path.basename(image_path)}")
        print(f"Cores: {', '.join(color_ranges.keys())}")
        print("="*80)
        
        total_filters = len(color_ranges)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        combined_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        individual_results = []
        
        for idx, (color_name, (lower_hsv, upper_hsv)) in enumerate(color_ranges.items()):
            progress = (idx + 1) / total_filters
            progress_bar_length = 40
            progress_filled = int(progress * progress_bar_length)
            progress_bar = "[" + "=" * progress_filled + " " * (progress_bar_length - progress_filled) + "]"
            
            sys.stdout.write(f"\r{progress_bar} {progress:.1%} | Processando: {color_name:<20}")
            sys.stdout.flush()
            
            mask = cv2.inRange(hsv, np.array(lower_hsv), np.array(upper_hsv))
            
            result = cv2.bitwise_and(img, img, mask=mask)
            
            combined_mask = cv2.bitwise_or(combined_mask, mask)
            
            if remove_mode:
                individual_path = os.path.join(multi_filter_dir, f"{base_name}_removed_{color_name}.jpg")
            else:
                individual_path = os.path.join(multi_filter_dir, f"{base_name}_{color_name}.jpg")
                
            cv2.imwrite(individual_path, result)
            individual_results.append(individual_path)
        
        sys.stdout.write("\r" + " " * 100 + "\r")
        
        if remove_mode:
            inverted_mask = cv2.bitwise_not(combined_mask)
            
            final_result = cv2.bitwise_and(img, img, mask=inverted_mask)
            
            
            white_bg = np.ones_like(img) * 255
            # Aplica o fundo branco apenas onde a máscara indicar pixels removidos
            final_result = np.where(
                np.repeat(inverted_mask[:, :, np.newaxis], 3, axis=2) == 0,
                white_bg,
                final_result
            ).astype(np.uint8)
        else:
            # No modo normal, combina todos os resultados filtrados
            # Usa o máximo para combinar pixels - evita sobreposição escura
            final_result = cv2.bitwise_and(img, img, mask=combined_mask)
        
        # Salva o resultado combinado
        combined_path = os.path.join(multi_filter_dir, f"{output_name}.jpg")
        cv2.imwrite(combined_path, final_result)
        
        operation = "removidas" if remove_mode else "aplicados"
        print("\n" + "-"*80)
        print(f"\033[1;32mProcessamento concluído!\033[0m")
        print(f"Cores {operation}: {', '.join(color_ranges.keys())}")
        print(f"Resultado final salvo em: {combined_path}")
        print(f"Resultados individuais salvos em: {multi_filter_dir}/")
        print("-"*80)
        
        # Se há suporte a GUI, exibe os resultados
        if GUI_SUPPORTED:
            # Redimensiona para exibição
            scale = min(1.0, 800 / max(img.shape[1], 1))
            display_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            display_result = cv2.resize(final_result, (0, 0), fx=scale, fy=scale)
            
            # Exibe a imagem original e o resultado 
            title = 'Resultado: Cores Removidas' if remove_mode else 'Filtros Combinados'
            cv2.imshow('Imagem Original', display_img)
            cv2.imshow(title, display_result)
            
            # Exibe também os resultados individuais se não estiver no modo de remoção
            # No modo de remoção, mostrar apenas o resultado final é mais útil
            if not remove_mode:
                for color_name, individual_path in zip(color_ranges.keys(), individual_results):
                    individual_img = cv2.imread(individual_path)
                    if individual_img is not None:
                        display_individual = cv2.resize(individual_img, (0, 0), fx=scale, fy=scale)
                        cv2.imshow(f'Filtro: {color_name}', display_individual)
                    
            print("\nPressione qualquer tecla nas janelas de imagem para continuar...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
        return combined_path, individual_results

def run_predefined_tests():
    """Executa testes predefinidos de filtragem de cores."""
    print("\n" + "*"*80)
    print("* EXECUÇÃO DE TESTES PREDEFINIDOS DE FILTROS DE COR")
    print("*"*80)
    print(f"* Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("*"*80 + "\n")
    
    tester = ColorRangeTester()
    
    # Cores comuns para teste
    color_ranges = {
        'dark': ([0, 0, 0], [180, 255, 60]),
        'red': ([0, 100, 20], [10, 255, 255]),
        'blue': ([100, 100, 20], [140, 255, 255]),
        'green': ([40, 100, 20], [80, 255, 255]),
        'yellow': ([20, 100, 20], [40, 255, 255])
    }
    
    start_time = time.time()
    total_tests = len(color_ranges)
    current_test = 0
    
    # Processa cada cor
    for color_name, (lower, upper) in color_ranges.items():
        current_test += 1
        print(f"\n\033[1;36mTESTE {current_test}/{total_tests}: {color_name.upper()}\033[0m")
        tester.process_batch(file_pattern="*.png", lower_hsv=lower, upper_hsv=upper, color_name=color_name)
    
    # Resumo final
    print("\n" + "="*80)
    print("RESULTADOS FINAIS DOS TESTES PREDEFINIDOS")
    print("="*80)
    print(f"Testes concluídos: {total_tests}")
    print(f"Tempo total de execução: {(time.time() - start_time):.2f} segundos")
    print("="*80)

def test_specific_image():
    """Test the specific image located at C:/Users/IsraelAntunes/Desktop/grid-search-main/mjlk.png
    First removes the background, then applies all available color ranges automatically."""
    specific_image_path = r"C:/Users/IsraelAntunes/Desktop/grid-search-main/mjlk.png"
    
    if not os.path.exists(specific_image_path):
        print(f"\033[1;31mImagem específica não encontrada: {specific_image_path}\033[0m")
        return
    
    print("\n" + "*"*80)
    print(f"* TESTE AUTOMÁTICO COM IMAGEM ESPECÍFICA: {os.path.basename(specific_image_path)}")
    print("*"*80)
    
    tester = ColorRangeTester()
    
    # Primeiro remove o fundo da imagem
    print("\n" + "="*80)
    print("ETAPA 1: REMOÇÃO DE FUNDO")
    print("="*80)
    
    # Usa branco como cor padrão de fundo para remover
    bgcolor = [255, 255, 255]  # BGR
    print(f"Removendo fundo com cor (BGR): {bgcolor}")
    
    # Remove o fundo da imagem
    background_removed_path = tester.remove_background(specific_image_path, bgcolor)
    
    if background_removed_path is None:
        print("\033[1;31mFalha ao remover fundo. Usando imagem original.\033[0m")
        processed_image_path = specific_image_path
    else:
        processed_image_path = background_removed_path
        print(f"\033[1;32mFundo removido com sucesso! Usando imagem processada: {os.path.basename(processed_image_path)}\033[0m")
    
    # Lista expandida de cores para teste
    color_ranges = {
        'dark': ([0, 0, 0], [180, 255, 60]),
        'light': ([0, 0, 200], [180, 30, 255]),
        'red1': ([0, 100, 100], [10, 255, 255]),
        'red2': ([160, 100, 100], [180, 255, 255]),  # Vermelho também está próximo a 180º
        'orange': ([10, 100, 100], [25, 255, 255]),
        'yellow': ([25, 100, 100], [35, 255, 255]),
        'green': ([35, 100, 100], [85, 255, 255]),
        'teal': ([85, 100, 100], [100, 255, 255]),
        'blue': ([100, 100, 100], [130, 255, 255]),
        'purple': ([130, 100, 100], [155, 255, 255]),
        'pink': ([155, 100, 100], [170, 255, 255]),
        'gray': ([0, 0, 40], [180, 25, 220]),
        'brown': ([10, 50, 20], [30, 255, 200]),
    }
    
    print("\n" + "="*80)
    print(f"ETAPA 2: APLICAÇÃO DE FILTROS DE COR")
    print(f"Imagem: {os.path.basename(processed_image_path)}")
    print("="*80)
    
    start_time = time.time()
    total_filters = len(color_ranges)
    
    for idx, (color_name, (lower, upper)) in enumerate(color_ranges.items()):
        # Status de progresso
        progress = (idx + 1) / total_filters
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / progress if progress > 0 else 0
        remaining_time = estimated_total_time - elapsed_time if estimated_total_time > 0 else 0
        
        # Barra de progresso
        progress_bar_length = 40
        progress_filled = int(progress * progress_bar_length)
        progress_bar = "[" + "=" * progress_filled + " " * (progress_bar_length - progress_filled) + "]"
        
        # Exibe progresso
        sys.stdout.write(f"\r{progress_bar} {progress:.1%} | Aplicando: {color_name:<10}")
        sys.stdout.flush()
        
        # Processa a imagem com o filtro atual
        img, mask, result = tester.filter_image(processed_image_path, lower, upper, color_name)
    
    # Limpa a linha atual e exibe o resumo
    sys.stdout.write("\r" + " " * 100 + "\r")
    
    print("\n" + "-"*80)
    print(f"RESUMO DO PROCESSAMENTO COMPLETO")
    print(f"1. Remoção de fundo: {'Sucesso' if background_removed_path is not None else 'Falha'}")
    print(f"2. Filtros aplicados: {total_filters}")
    print(f"Tempo de execução: {(time.time() - start_time):.2f} segundos")
    print(f"Resultados salvos em: {tester.results_dir}/")
    print("-"*80)
    
    # Pergunte ao usuário se deseja continuar com processamento adicional
    print("\nOpções adicionais:")
    print("1. \033[1;33mAbrir janela interativa para ajuste fino\033[0m")
    print("2. \033[1;34mRemover fundo com configuração personalizada\033[0m")
    print("3. \033[1;31mVoltar ao menu principal\033[0m")
    
    option = input("\nEscolha uma opção: ")
    
    if option == "1":
        tester.show_test_window(processed_image_path)
    elif option == "2":
        print("\nDigite a cor de fundo a ser removida (BGR):")
        b = int(input("B (0-255, padrão 255 para branco): ") or 255)
        g = int(input("G (0-255, padrão 255 para branco): ") or 255)
        r = int(input("R (0-255, padrão 255 para branco): ") or 255)
        
        bgcolor = [b, g, r]
        new_processed_path = tester.remove_background(specific_image_path, bgcolor)
        if new_processed_path:
            print("\nDeseja aplicar os filtros de cor novamente na nova imagem? (s/n)")
            if input().lower() == 's':
                # Chamamos a função recursivamente para processar a nova imagem
                test_specific_image()
    elif option == "3" or option == "":
        return
    else:
        print("\n\033[1;31mOpção inválida.\033[0m")

def main():
    """Função principal para executar o programa."""
    print("\n" + "*"*80)
    print("* PROGRAMA DE TESTE DE FILTRAGEM POR COR")
    print("*"*80)
    print(f"* Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("*"*80)
    
    option_text = [
        "1. \033[1;36mExecutar testes predefinidos (várias cores)\033[0m",
        "2. \033[1;33mAjuste fino de filtros HSV\033[0m" + (" [GUI requerida]" if not GUI_SUPPORTED else ""),
        "3. \033[1;32mProcessar com parâmetros personalizados\033[0m",
        "4. \033[1;34mRemover fundo de imagem\033[0m",
        "5. \033[1;35mTestar imagem específica (remover fundo + filtros de cor)\033[0m",
        "6. \033[1;37mDetectar e remover cores de borda\033[0m",
        "7. \033[1;36mAplicar múltiplos filtros de cor a uma imagem\033[0m",
        "8. \033[1;31mSair\033[0m"
    ]
    
    for line in option_text:
        print("\n" + line)
    
    option = input("\nEscolha uma opção: ")
    
    tester = ColorRangeTester()
    
    if option == "1":
        run_predefined_tests()
    elif option == "2":
        if not GUI_SUPPORTED:
            print("\n\033[1;33mAviso: Esta opção requer suporte a GUI, mas uma alternativa em linha de comando será oferecida.\033[0m")
        
        image_path = input("Digite o caminho da imagem para teste (ex: ./image/1f1f.png): ")
        if not image_path:
            image_path = "./image/3zwv.png"
        tester.show_test_window(image_path)
    elif option == "3":
        print("\n" + "-"*80)
        print("FILTRO PERSONALIZADO - CONFIGURAÇÃO DE PARÂMETROS")
        print("-"*80)
        
        print("\nDigite os valores HSV:")
        try:
            h_min = int(input("H Min (0-180): "))
            s_min = int(input("S Min (0-255): "))
            v_min = int(input("V Min (0-255): "))
            h_max = int(input("H Max (0-180): "))
            s_max = int(input("S Max (0-255): "))
            v_max = int(input("V Max (0-255): "))
            
            lower_hsv = [h_min, s_min, v_min]
            upper_hsv = [h_max, s_max, v_max]
            
            color_name = input("Nome da cor/faixa: ")
            if not color_name:
                color_name = f"custom_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
            pattern = input("Padrão de arquivo (ex: *.png): ")
            if not pattern:
                pattern = "*.png"
            
            print("\n" + "-"*80)
            print(f"INICIANDO PROCESSAMENTO COM CONFIGURAÇÃO PERSONALIZADA")
            print("-"*80)    
            
            tester.process_batch(file_pattern=pattern, lower_hsv=lower_hsv, 
                                upper_hsv=upper_hsv, color_name=color_name)
        except ValueError:
            print("\n\033[1;31mErro: Os valores precisam ser números inteiros.\033[0m")
    elif option == "4":
        print("\n" + "-"*80)
        print("REMOÇÃO DE FUNDO DE IMAGEM")
        print("-"*80)
        
        print("\n1. Processar uma única imagem")
        print("2. Processar em lote")
        bg_option = input("\nEscolha uma opção: ")
        
        if bg_option == "1":
            image_path = input("Digite o caminho da imagem (ex: ./image/1f1f.png): ")
            if not image_path:
                print("\033[1;31mCaminho de imagem necessário.\033[0m")
            else:
                print("\nDigite a cor de fundo a ser removida (BGR):")
                try:
                    b = int(input("B (0-255, padrão 255 para branco): ") or 255)
                    g = int(input("G (0-255, padrão 255 para branco): ") or 255)
                    r = int(input("R (0-255, padrão 255 para branco): ") or 255)
                    
                    bgcolor = [b, g, r]
                    tester.remove_background(image_path, bgcolor)
                except ValueError:
                    print("\n\033[1;31mErro: Os valores precisam ser números inteiros.\033[0m")
                
        elif bg_option == "2":
            pattern = input("Padrão de arquivo (ex: *.png): ")
            if not pattern:
                pattern = "*.png"
                
            print("\nDigite a cor de fundo a ser removida (BGR):")
            try:
                b = int(input("B (0-255, padrão 255 para branco): ") or 255)
                g = int(input("G (0-255, padrão 255 para branco): ") or 255)
                r = int(input("R (0-255, padrão 255 para branco): ") or 255)
                
                bgcolor = [b, g, r]
                tester.batch_remove_background(pattern, bgcolor)
            except ValueError:
                print("\n\033[1;31mErro: Os valores precisam ser números inteiros.\033[0m")
        else:
            print("\n\033[1;31mOpção inválida.\033[0m")
            
    elif option == "5":
        test_specific_image()
    elif option == "6":
        # Opção para detecção e remoção de cores de borda
        print("\n" + "-"*80)
        print("DETECÇÃO E REMOÇÃO DE CORES DE BORDA")
        print("-"*80)
        
        border_options = [
            "1. Detectar cores de borda (análise)" + (" [versão simplificada]" if not GUI_SUPPORTED else ""),
            "2. Remover cores de borda com configuração manual",
            "3. Remover cores de borda automaticamente",
            "4. Processar em lote (remoção automática)"
        ]
        
        for line in border_options:
            print("\n" + line)
            
        border_option = input("\nEscolha uma opção: ")
        
        if border_option == "1":
            image_path = input("Digite o caminho da imagem (ex: ./image/1f1f.png): ")
            if not image_path:
                print("\033[1;31mCaminho de imagem necessário.\033[0m")
            else:
                try:
                    num_colors = int(input("Número de cores principais a exibir (padrão 5): ") or 5)
                    tester.display_border_colors(image_path, num_colors)
                except ValueError:
                    print("\n\033[1;31mErro: Os valores precisam ser números inteiros.\033[0m")
                
        elif border_option == "2":
            image_path = input("Digite o caminho da imagem (ex: ./image/1f1f.png): ")
            if not image_path:
                print("\033[1;31mCaminho de imagem necessário.\033[0m")
            else:
                try:
                    num_colors = int(input("Número de cores de borda a remover (padrão 1): ") or 1)
                    tolerance = int(input("Tolerância da cor (0-255, padrão 20): ") or 20)
                    border_width = int(input("Largura da borda em pixels (padrão 5): ") or 5)
                    
                    tester.remove_border_colors(image_path, num_colors, tolerance, border_width)
                except ValueError:
                    print("\n\033[1;31mErro: Os valores precisam ser números inteiros.\033[0m")
                
        elif border_option == "3":
            image_path = input("Digite o caminho da imagem (ex: ./image/1f1f.png): ")
            if not image_path:
                print("\033[1;31mCaminho de imagem necessário.\033[0m")
            else:
                tester.auto_remove_border_colors(image_path)
                
        elif border_option == "4":
            pattern = input("Padrão de arquivo (ex: *.png): ")
            if not pattern:
                pattern = "*.png"
            
            tester.batch_auto_remove_border_colors(pattern)
            
        else:
            print("\n\033[1;31mOpção inválida.\033[0m")
            
    elif option == "7":
        print("\n" + "-"*80)
        print("APLICAR MÚLTIPLOS FILTROS DE COR")
        print("-"*80)
        
        image_path = input("Digite o caminho da imagem (ex: ./image/1f1f.png): ")
        if not image_path:
            print("\033[1;31mCaminho de imagem necessário.\033[0m")
            return
        
        print("\nEscolha uma opção:")
        print("1. Usar filtros predefinidos (vermelho, azul, amarelo)")
        print("2. Definir filtros personalizados")
        print("3. Remover cores específicas da imagem")
        
        filter_option = input("\nOpção: ")
        
        if filter_option == "1":
            # Usa os filtros predefinidos
            tester.apply_multiple_filters(image_path)
        
        elif filter_option == "2":
            # Configuração manual de filtros
            custom_filters = {}
            
            print("\nDefina os filtros personalizados (deixe o nome em branco para terminar)")
            while True:
                color_name = input("\nNome da cor (ou Enter para terminar): ")
                if not color_name:
                    break
                    
                try:
                    print(f"Digite os valores HSV para {color_name}:")
                    h_min = int(input("H Min (0-180): "))
                    s_min = int(input("S Min (0-255): "))
                    v_min = int(input("V Min (0-255): "))
                    h_max = int(input("H Max (0-180): "))
                    s_max = int(input("S Max (0-255): "))
                    v_max = int(input("V Max (0-255): "))
                    
                    lower_hsv = [h_min, s_min, v_min]
                    upper_hsv = [h_max, s_max, v_max]
                    
                    custom_filters[color_name] = (lower_hsv, upper_hsv)
                    print(f"\033[1;32mFiltro {color_name} adicionado!\033[0m")
                except ValueError:
                    print("\n\033[1;31mErro: Os valores precisam ser números inteiros.\033[0m")
            
            if custom_filters:
                output_name = input("\nNome para o arquivo de saída (opcional): ")
                output_name = output_name if output_name else None
                
                tester.apply_multiple_filters(image_path, custom_filters, output_name)
            else:
                print("\033[1;31mNenhum filtro personalizado definido.\033[0m")
                
        elif filter_option == "3":
            # Configuração para remoção de cores
            colors_to_remove = {}
            
            print("\n\033[1;33mMODO DE REMOÇÃO DE CORES\033[0m")
            print("\nEscolha uma opção:")
            print("1. Usar cores padrão para remoção (azul, vermelho, verde, amarelo, rosa, branco e laranja)")
            print("2. Definir cores personalizadas para remover")
            
            remove_option = input("\nOpção: ")
            
            if remove_option == "1":
                # Usa as cores padrão para remoção
                tester.apply_multiple_filters(image_path, remove_mode=True)
                return
                
            print("\nDefina as cores que deseja REMOVER da imagem (deixe o nome em branco para terminar)")
            while True:
                color_name = input("\nNome da cor a remover (ou Enter para terminar): ")
                if not color_name:
                    break
                    
                try:
                    print(f"Digite os valores HSV para {color_name}:")
                    h_min = int(input("H Min (0-180): "))
                    s_min = int(input("S Min (0-255): "))
                    v_min = int(input("V Min (0-255): "))
                    h_max = int(input("H Max (0-180): "))
                    s_max = int(input("S Max (0-255): "))
                    v_max = int(input("V Max (0-255): "))
                    
                    lower_hsv = [h_min, s_min, v_min]
                    upper_hsv = [h_max, s_max, v_max]
                    
                    colors_to_remove[color_name] = (lower_hsv, upper_hsv)
                    print(f"\033[1;32mCor {color_name} adicionada para remoção!\033[0m")
                except ValueError:
                    print("\n\033[1;31mErro: Os valores precisam ser números inteiros.\033[0m")
            
            if colors_to_remove:
                output_name = input("\nNome para o arquivo de saída (opcional): ")
                output_name = output_name if output_name else None
                
                # Chama a função com o modo de remoção ativado
                tester.apply_multiple_filters(image_path, colors_to_remove, output_name, remove_mode=True)
            else:
                print("\033[1;31mNenhuma cor definida para remoção.\033[0m")
                
        else:
            print("\n\033[1;31mOpção inválida.\033[0m")

    elif option == "8":
        print("\n\033[1;33mPrograma encerrado.\033[0m")
    else:
        print("\n\033[1;31mOpção inválida.\033[0m")

if __name__ == "__main__":
    main()
