import cv2
import numpy as np
import os
import glob
from pathlib import Path
import time
from datetime import datetime
import sys
import argparse

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

class DarkColorExtractor:
    """Classe para extração de cores escuras em imagens, removendo cores claras."""
    
    def __init__(self, image_dir='./image', results_dir='./results/dark_extractor'):
        """Inicializa o extrator de cores escuras.
        
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
        
    def extract_dark_colors(self, image_path, value_threshold=100, output_dir=None):
        """Extrai cores escuras da imagem, removendo as cores claras.
        
        Args:
            image_path: Caminho para a imagem
            value_threshold: Valor máximo para considerar uma cor como escura (0-255)
                Valores menores resultam em imagens mais escuras
            output_dir: Diretório opcional para salvar os resultados
            
        Returns:
            Tuple contendo (imagem original, máscara escura, resultado escuro, resultado claro)
        """
        if not os.path.exists(image_path):
            print(f"\033[1;31mImagem não encontrada: {image_path}\033[0m")
            return None, None, None, None
            
        # Carrega a imagem
        img = cv2.imread(image_path)
        if img is None:
            print(f"\033[1;31mErro ao carregar a imagem: {image_path}\033[0m")
            return None, None, None, None
            
        # Converte para HSV (Hue, Saturation, Value)
        # O canal Value (V) representa o brilho - usaremos isso para determinar as cores escuras
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Extrai apenas o canal de Value (brilho)
        v_channel = hsv[:,:,2]
        
        # Cria a máscara para cores escuras (Value abaixo do threshold)
        dark_mask = v_channel < value_threshold
        dark_mask = dark_mask.astype(np.uint8) * 255
        
        # Cria a máscara para cores claras (inverso da máscara escura)
        light_mask = cv2.bitwise_not(dark_mask)
        
        # Aplica as máscaras para obter as partes escuras e claras
        dark_result = cv2.bitwise_and(img, img, mask=dark_mask)
        light_result = cv2.bitwise_and(img, img, mask=light_mask)
        
        # Caso queira um fundo branco nas áreas removidas
        white_bg = np.ones_like(img) * 255
        dark_with_white_bg = np.where(
            np.repeat(dark_mask[:, :, np.newaxis], 3, axis=2) > 0,
            dark_result,
            white_bg
        ).astype(np.uint8)
        
        # Define o diretório de saída
        if output_dir is None:
            output_dir = self.results_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Extrai o nome base da imagem
        base_name = os.path.basename(image_path).split('.')[0]
        
        # Salva as imagens resultantes
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_original.jpg"), img)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_dark_mask.jpg"), dark_mask)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_dark_only.jpg"), dark_result)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_dark_white_bg.jpg"), dark_with_white_bg)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_light_only.jpg"), light_result)
        
        print(f"\033[1;32mProcessamento concluído para {os.path.basename(image_path)}\033[0m")
        print(f"Threshold de escuridão: {value_threshold}")
        print(f"Resultados salvos em: {output_dir}/")
        
        return img, dark_mask, dark_result, light_result
    
    def process_batch(self, file_pattern="*.png", value_threshold=100):
        """Processa um lote de imagens, extraindo as cores escuras.
        
        Args:
            file_pattern: Padrão para encontrar arquivos (ex: "*.png")
            value_threshold: Valor máximo para considerar uma cor como escura (0-255)
            
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
        print(f"EXTRAÇÃO DE CORES ESCURAS EM LOTE")
        print(f"Threshold de valor (escuridão): {value_threshold}")
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
            img, mask, dark, light = self.extract_dark_colors(image_path, value_threshold)
            
            if img is not None:
                success_count += 1
        
        # Limpa a linha atual e exibe o resumo
        sys.stdout.write("\r" + " " * 100 + "\r")  # Limpa a linha
        
        print("\n" + "-"*80)
        print(f"RESUMO DO PROCESSAMENTO")
        print(f"Threshold de escuridão: {value_threshold}")
        print(f"Imagens processadas: {success_count}/{total_images} ({success_count/total_images:.1%})")
        print(f"Tempo de execução: {(time.time() - start_time):.2f} segundos")
        print(f"Resultados salvos em: {self.results_dir}/")
        print("-"*80 + "\n")
        
        return success_count
    
    def show_interactive_threshold(self, image_path):
        """Exibe uma janela interativa para ajustar o threshold de escuridão.
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
        print(f"AJUSTE INTERATIVO DE THRESHOLD DE ESCURIDÃO")
        print(f"Imagem: {os.path.basename(image_path)}")
        print("="*80)
        
        if GUI_SUPPORTED:
            # Versão interativa com GUI
            print("* Use o controle deslizante para ajustar o threshold de valor (escuridão).")
            print("* Valores menores = apenas cores mais escuras são mantidas.")
            print("* Valores maiores = mais cores são incluídas nos resultados.")
            print("* Pressione 's' para salvar a configuração atual.")
            print("* Pressione 'q' para sair.")
            print("="*80 + "\n")
            
            # Cria janela e trackbar para ajuste
            cv2.namedWindow('Ajuste de Threshold')
            
            # Define um valor inicial para o threshold
            initial_threshold = 100
            cv2.createTrackbar('Threshold', 'Ajuste de Threshold', initial_threshold, 255, lambda x: None)
            
            while True:
                # Lê o valor atual do trackbar
                threshold = cv2.getTrackbarPos('Threshold', 'Ajuste de Threshold')
                
                # Converte para HSV
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                
                # Extrai apenas o canal de Value (brilho)
                v_channel = hsv[:,:,2]
                
                # Cria a máscara para cores escuras (Value abaixo do threshold)
                dark_mask = v_channel < threshold
                dark_mask = dark_mask.astype(np.uint8) * 255
                
                # Aplica a máscara para obter as partes escuras
                dark_result = cv2.bitwise_and(img, img, mask=dark_mask)
                
                # Cria o resultado com fundo branco
                white_bg = np.ones_like(img) * 255
                dark_with_white_bg = np.where(
                    np.repeat(dark_mask[:, :, np.newaxis], 3, axis=2) > 0,
                    dark_result,
                    white_bg
                ).astype(np.uint8)
                
                # Redimensiona as imagens para exibição
                scale = 0.8
                display_img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                display_mask = cv2.cvtColor(cv2.resize(dark_mask, (0, 0), fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)
                display_dark = cv2.resize(dark_result, (0, 0), fx=scale, fy=scale)
                display_white_bg = cv2.resize(dark_with_white_bg, (0, 0), fx=scale, fy=scale)
                
                # Exibe as imagens
                cv2.imshow('Imagem Original', display_img)
                cv2.imshow('Máscara de Cores Escuras', display_mask)
                cv2.imshow('Cores Escuras', display_dark)
                cv2.imshow('Resultado com Fundo Branco', display_white_bg)
                
                # Mostra o threshold atual no terminal
                sys.stdout.write(f"\rThreshold de escuridão atual: {threshold}")
                sys.stdout.flush()
                
                # Aguarda entrada do usuário
                key = cv2.waitKey(100) & 0xFF
                
                # Se 's' for pressionado, salva a configuração atual
                if key == ord('s'):
                    print("\n\n" + "-"*80)
                    print(f"\033[1;32mThreshold de escuridão salvo: {threshold}\033[0m")
                    self.extract_dark_colors(image_path, threshold)
                    print("-"*80)
                
                # Se 'q' for pressionado, sai do loop
                if key == ord('q'):
                    break
                    
            # Fecha todas as janelas
            cv2.destroyAllWindows()
            print("\n\033[1;33mAjuste interativo encerrado.\033[0m")
        else:
            # Versão alternativa baseada em linha de comando
            print("* GUI não disponível. Usando modo de entrada manual.")
            print("* Digite um valor de threshold para testar diferentes níveis de escuridão.")
            print("* Valores menores = apenas cores mais escuras são mantidas.")
            print("* Valores maiores = mais cores são incluídas nos resultados.")
            print("* Intervalo válido: 0-255")
            print("="*80 + "\n")
            
            while True:
                print("\nDigite o valor de threshold para testar (ou 'q' para sair):")
                threshold_input = input("Threshold de escuridão (0-255): ")
                
                if threshold_input.lower() == 'q':
                    break
                    
                try:
                    threshold = int(threshold_input)
                    if threshold < 0 or threshold > 255:
                        print("\033[1;31mErro: O threshold deve estar entre 0 e 255.\033[0m")
                        continue
                        
                    self.extract_dark_colors(image_path, threshold)
                    
                    # Pergunta se deseja continuar testando
                    continue_testing = input("\nDeseja testar outro valor de threshold? (s/n): ")
                    if continue_testing.lower() != 's':
                        break
                        
                except ValueError:
                    print("\033[1;31mEntrada inválida. Use apenas números inteiros entre 0 e 255.\033[0m")
            
            print("\n\033[1;33mAjuste manual encerrado.\033[0m")


def main():
    """Função principal para executar o programa."""
    parser = argparse.ArgumentParser(description='Extrai cores escuras das imagens removendo as cores claras.')
    
    parser.add_argument('-i', '--image', type=str, help='Caminho para uma imagem específica')
    parser.add_argument('-d', '--dir', type=str, default='./image', help='Diretório contendo imagens para processamento em lote')
    parser.add_argument('-p', '--pattern', type=str, default='*.png', help='Padrão para arquivos de imagem (ex: "*.jpg")')
    parser.add_argument('-t', '--threshold', type=int, default=100, help='Threshold de valor para considerar cores escuras (0-255)')
    parser.add_argument('-o', '--output', type=str, default=None, help='Diretório para salvar os resultados')
    parser.add_argument('--interactive', action='store_true', help='Executar o modo interativo de ajuste de threshold')
    parser.add_argument('--batch', action='store_true', help='Processar todas as imagens no diretório')
    
    args = parser.parse_args()
    
    print("\n" + "*"*80)
    print("* EXTRATOR DE CORES ESCURAS")
    print("*"*80)
    print(f"* Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("*"*80)
    
    # Define o diretório de resultados
    results_dir = args.output if args.output else './results/dark_extractor'
    
    # Cria a instância do extrator
    extractor = DarkColorExtractor(image_dir=args.dir, results_dir=results_dir)
    
    # Modo interativo com argumentos da linha de comando
    if args.interactive and args.image:
        extractor.show_interactive_threshold(args.image)
        return
    
    # Processamento em lote
    if args.batch:
        extractor.process_batch(file_pattern=args.pattern, value_threshold=args.threshold)
        return
    
    # Processamento de uma única imagem
    if args.image:
        extractor.extract_dark_colors(args.image, args.threshold)
        return
    
    # Se não houver argumentos específicos, exibe menu interativo
    show_menu(extractor)


def show_menu(extractor):
    """Exibe um menu interativo para o usuário."""
    while True:
        print("\n" + "="*80)
        print("MENU - EXTRATOR DE CORES ESCURAS")
        print("="*80)
        
        options = [
            "1. \033[1;36mProcessar uma imagem\033[0m",
            "2. \033[1;33mProcessar em lote\033[0m",
            "3. \033[1;32mAjuste interativo de threshold\033[0m" + (" [GUI requerida]" if not GUI_SUPPORTED else ""),
            "4. \033[1;31mSair\033[0m"
        ]
        
        for option in options:
            print(option)
            
        choice = input("\nEscolha uma opção: ")
        
        if choice == "1":
            image_path = input("\nCaminho da imagem: ")
            if not image_path:
                print("\033[1;31mCaminho da imagem é necessário.\033[0m")
                continue
                
            try:
                threshold = int(input("Threshold de escuridão (0-255, padrão 100): ") or 100)
                if threshold < 0 or threshold > 255:
                    print("\033[1;31mThreshold deve estar entre 0 e 255. Usando valor padrão 100.\033[0m")
                    threshold = 100
                    
                extractor.extract_dark_colors(image_path, threshold)
            except ValueError:
                print("\033[1;31mValor inválido para threshold. Usando valor padrão 100.\033[0m")
                extractor.extract_dark_colors(image_path, 100)
                
        elif choice == "2":
            pattern = input("\nPadrão de arquivo (ex: *.png, padrão: *.png): ") or "*.png"
            
            try:
                threshold = int(input("Threshold de escuridão (0-255, padrão 100): ") or 100)
                if threshold < 0 or threshold > 255:
                    print("\033[1;31mThreshold deve estar entre 0 e 255. Usando valor padrão 100.\033[0m")
                    threshold = 100
                    
                extractor.process_batch(pattern, threshold)
            except ValueError:
                print("\033[1;31mValor inválido para threshold. Usando valor padrão 100.\033[0m")
                extractor.process_batch(pattern, 100)
                
        elif choice == "3":
            if not GUI_SUPPORTED:
                print("\n\033[1;33mAviso: Esta opção requer suporte a GUI, mas uma alternativa em linha de comando será oferecida.\033[0m")
                
            image_path = input("\nCaminho da imagem: ")
            if not image_path:
                print("\033[1;31mCaminho da imagem é necessário.\033[0m")
                continue
                
            extractor.show_interactive_threshold(image_path)
            
        elif choice == "4":
            print("\n\033[1;33mPrograma encerrado.\033[0m")
            break
            
        else:
            print("\n\033[1;31mOpção inválida. Tente novamente.\033[0m")


if __name__ == "__main__":
    main()
