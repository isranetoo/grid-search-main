import cv2
import numpy as np
import os
import glob
from pathlib import Path
import time
from datetime import datetime
import sys

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
        
        Args:
            image_path: Caminho para a imagem a ser usada no teste
        """
        if not os.path.exists(image_path):
            print(f"\033[1;31mImagem não encontrada: {image_path}\033[0m")
            return
        
        # Cabeçalho informativo
        print("\n" + "="*80)
        print(f"TESTE INTERATIVO DE FILTROS HSV")
        print(f"Imagem: {os.path.basename(image_path)}")
        print("="*80)
        print("* Use os controles deslizantes para ajustar os valores HSV.")
        print("* Pressione 's' para salvar a configuração atual.")
        print("* Pressione 'q' para sair.")
        print("="*80 + "\n")
        
        # Carrega a imagem
        img = cv2.imread(image_path)
        
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

def main():
    """Função principal para executar o programa."""
    print("\n" + "*"*80)
    print("* PROGRAMA DE TESTE DE FILTRAGEM POR COR")
    print("*"*80)
    print(f"* Data/hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print("*"*80)
    
    print("\n1. \033[1;36mExecutar testes predefinidos (várias cores)\033[0m")
    print("2. \033[1;33mAbrir janela interativa para ajuste fino\033[0m")
    print("3. \033[1;32mProcessar com parâmetros personalizados\033[0m")
    print("4. \033[1;31mSair\033[0m")
    
    option = input("\nEscolha uma opção: ")
    
    tester = ColorRangeTester()
    
    if option == "1":
        run_predefined_tests()
    elif option == "2":
        image_path = input("Digite o caminho da imagem para teste (ex: ./image/1f1f.png): ")
        if not image_path:
            image_path = "./image/1f1f.png"
        tester.show_test_window(image_path)
    elif option == "3":
        print("\n" + "-"*80)
        print("FILTRO PERSONALIZADO - CONFIGURAÇÃO DE PARÂMETROS")
        print("-"*80)
        
        print("\nDigite os valores HSV:")
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
    elif option == "4":
        print("\n\033[1;33mPrograma encerrado.\033[0m")
    else:
        print("\n\033[1;31mOpção inválida.\033[0m")

if __name__ == "__main__":
    main()
