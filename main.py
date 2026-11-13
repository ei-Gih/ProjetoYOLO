from pathlib import Path
from tkinter import messagebox
import unicodedata

import customtkinter as ctk
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
PASTA_PROJETO = Path(__file__).resolve().parent
CONFIANCA_MINIMA = 0.40
FUNDO, PAINEL, CARTAO = "#060817", "#16254F", "#060817"
BORDA, TEXTO, SUAVE = "#667D9D", "#ECECEC", "#ACBBC6"
DESTAQUE = "#667D9D"
CHECKLIST_FUNDO = "#ACBBC6"
CHECKLIST_ITEM = "#BDCAD3"
CHECKLIST_TEXTO = "#16254F"
AZUL, VERDE, AMARELO, VERMELHO = TEXTO, TEXTO, SUAVE, SUAVE


def criar_icone_epi(epi):
    """Pictogramas próprios em alta resolução, sem fontes de emoji ou arquivos extras."""
    imagem = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    d = ImageDraw.Draw(imagem)
    cor = CHECKLIST_TEXTO
    def linha(pontos, largura=6):
        d.line(pontos, fill=cor, width=largura, joint="curve")
    if epi == "Jaleco":
        linha([(45, 20), (28, 30), (14, 69), (30, 77), (38, 57), (34, 111), (94, 111), (90, 57), (98, 77), (114, 69), (100, 30), (83, 20)])
        linha([(45, 20), (64, 44), (83, 20)])
        linha([(64, 44), (64, 110)], 4)
        linha([(45, 20), (43, 42), (55, 50)], 4)
        linha([(83, 20), (85, 42), (73, 50)], 4)
        d.rectangle((75, 62, 87, 76), outline=cor, width=3)
    elif epi == "Capacete":
        d.arc((22, 27, 106, 109), 180, 360, fill=cor, width=7)
        linha([(15, 71), (113, 71), (113, 86), (15, 86), (15, 71)])
        linha([(55, 66), (55, 26), (73, 26), (73, 66)], 5)
    elif epi == "Óculos de segurança":
        d.rounded_rectangle((14, 45, 57, 81), radius=10, outline=cor, width=6)
        d.rounded_rectangle((71, 45, 114, 81), radius=10, outline=cor, width=6)
        d.arc((54, 51, 74, 70), 180, 360, fill=cor, width=5)
        linha([(15, 49), (8, 34)], 5)
        linha([(113, 49), (120, 34)], 5)
    elif epi == "Máscara":
        linha([(29, 40), (64, 48), (99, 40), (97, 82), (64, 101), (31, 82), (29, 40)])
        d.arc((8, 43, 43, 84), 90, 270, fill=cor, width=5)
        d.arc((85, 43, 120, 84), 270, 450, fill=cor, width=5)
        linha([(42, 62), (86, 62)], 4)
        linha([(42, 76), (86, 76)], 4)
    elif epi == "Luvas":
        linha([(42, 111), (32, 83), (15, 63), (17, 54), (25, 53), (39, 67), (34, 27), (39, 21), (46, 25), (51, 58), (51, 16), (58, 12), (65, 16), (65, 56), (70, 20), (77, 18), (83, 24), (79, 60), (88, 34), (96, 33), (101, 40), (95, 79), (86, 111), (42, 111)], 5)
    else:
        d.arc((24, 17, 104, 105), 180, 360, fill=cor, width=8)
        d.rounded_rectangle((15, 57, 40, 103), radius=9, outline=cor, width=6)
        d.rounded_rectangle((88, 57, 113, 103), radius=9, outline=cor, width=6)
    return ctk.CTkImage(light_image=imagem, dark_image=imagem, size=(30, 30))


def normalizar(nome):
    texto = unicodedata.normalize("NFKD", nome.lower())
    return "-".join("".join(c for c in texto if not unicodedata.combining(c)).replace("_", " ").replace("-", " ").split())


# Ajuste os aliases para os nomes exatos das classes do seu modelo treinado.
EPIS = {
    "Jaleco": {"jaleco", "lab-coat", "labcoat", "gown", "coat"},
    "Óculos de segurança": {"oculos", "oculos-de-seguranca", "goggles", "safety-glasses", "safety-goggles"},
    "Máscara": {"mascara", "mask", "face-mask", "medical-mask"},
    "Luvas": {"luva", "luvas", "glove", "gloves", "safety-gloves"},
    "Protetor auricular": {"protetor-auricular", "protetor-oricular", "ear-protection", "earmuffs", "ear-muffs", "earplug", "earplugs"},
    "Capacete": {"capacete", "helmet", "hardhat", "hard-hat", "safety-helmet"},
}
PESSOAS = {"person", "pessoa", "worker", "trabalhador"}


def avaliar_pessoas(deteccoes, classes_modelo):
    """Associa cada EPI a uma única pessoa; sobreposições ambíguas não liberam."""
    suportados = {epi: bool(aliases & classes_modelo) for epi, aliases in EPIS.items()}
    pessoas = [d for d in deteccoes if d[0] in PESSOAS]
    presentes = [set() for _ in pessoas]
    for nome, caixa in deteccoes:
        epi = next((e for e, aliases in EPIS.items() if nome in aliases), None)
        if epi is None:
            continue
        x1, y1, x2, y2 = caixa
        area = max(0, x2 - x1) * max(0, y2 - y1)
        candidatos = []
        for i, (_, (px1, py1, px2, py2)) in enumerate(pessoas):
            intersecao = max(0, min(x2, px2) - max(x1, px1)) * max(0, min(y2, py2) - max(y1, py1))
            if area > 0 and intersecao / area >= 0.70:
                candidatos.append(i)
        if len(candidatos) == 1:
            presentes[candidatos[0]].add(epi)
    return pessoas, presentes, suportados


class Aplicacao(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("ZelarIA | Central de segurança e EPIs")
        self.geometry("1440x920")
        self.minsize(1100, 700)
        self.configure(fg_color=FUNDO)
        self.camera = None
        self.after_id = None
        self.camera_ativa = False
        self.modelo = None
        self.protocol("WM_DELETE_WINDOW", self.fechar)
        self.criar_interface()
        self.after(100, self.carregar_modelo)

    def label(self, pai, texto, tamanho=13, cor=TEXTO, negrito=False, **kwargs):
        return ctk.CTkLabel(pai, text=texto, font=("Segoe UI", tamanho, "bold" if negrito else "normal"), text_color=cor, **kwargs)

    def painel(self, pai, **kwargs):
        return ctk.CTkFrame(pai, fg_color=PAINEL, border_color=DESTAQUE, border_width=1, corner_radius=16, **kwargs)

    def criar_interface(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        menu = ctk.CTkFrame(self, width=370, corner_radius=0, fg_color=PAINEL)
        menu.grid(row=0, column=0, sticky="nsew")
        menu.grid_propagate(False)
        menu.grid_columnconfigure(0, weight=1)
        menu.grid_rowconfigure(1, weight=1)
        marca = ctk.CTkFrame(menu, fg_color="transparent")
        marca.grid(row=0, column=0, sticky="ew", padx=20, pady=(10, 8))
        caminho_logo = PASTA_PROJETO / "assets" / "zelaria-logo.png"
        if caminho_logo.is_file():
            with Image.open(caminho_logo) as arquivo_logo:
                imagem_logo = arquivo_logo.convert("RGBA")
            largura_logo = 195
            altura_logo = round(largura_logo * imagem_logo.height / imagem_logo.width)
            self.logo_marca = ctk.CTkImage(light_image=imagem_logo, dark_image=imagem_logo, size=(largura_logo, altura_logo))
            ctk.CTkLabel(marca, text="", image=self.logo_marca, fg_color="transparent", width=largura_logo, height=altura_logo).pack(anchor="w")
        else:
            self.label(marca, "ZelarIA", 25, AZUL, True).pack(anchor="w")
        self.label(marca, "DASHBOARD DE SEGURANÇA", 10, SUAVE, True).pack(anchor="w", pady=(2, 0))
        dashboard = ctk.CTkFrame(menu, fg_color="transparent", corner_radius=0)
        dashboard.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 12))
        dashboard.grid_columnconfigure(0, weight=1)
        dashboard.grid_rowconfigure(0, weight=1)
        corpo = ctk.CTkFrame(self, fg_color="transparent")
        corpo.grid(row=0, column=1, sticky="nsew", padx=18, pady=18)
        corpo.grid_columnconfigure(0, weight=1)
        corpo.grid_rowconfigure(0, weight=1)
        barra = ctk.CTkFrame(corpo, fg_color="transparent")
        barra.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        barra.grid_columnconfigure((0, 1, 2), weight=1, uniform="metricas")
        barra.grid_columnconfigure(3, weight=0, minsize=170)
        self.cards = {}
        for i, (chave, titulo) in enumerate((("pessoas", "PESSOAS NA CENA"), ("pendentes", "COM PENDÊNCIAS"), ("liberadas", "EPIs CONFIRMADOS"))):
            card = self.painel(barra)
            card.configure(fg_color=FUNDO)
            card.grid(row=0, column=i, sticky="nsew", padx=(0, 8))
            self.label(card, titulo, 10, SUAVE, True).pack(padx=8, pady=(12, 0))
            self.cards[chave] = self.label(card, "0", 30, TEXTO, True)
            self.cards[chave].pack(padx=8, pady=(0, 8))
        camera = self.painel(corpo)
        camera.grid(row=0, column=0, sticky="nsew")
        camera.grid_columnconfigure(0, weight=1)
        camera.grid_rowconfigure(1, weight=1)
        camera.grid_propagate(False)
        cab = ctk.CTkFrame(camera, fg_color="transparent")
        cab.grid(row=0, column=0, sticky="ew", padx=18, pady=16)
        self.label(cab, "Câmera ao vivo", 17, negrito=True).pack(side="left")
        self.status_camera = self.label(cab, "●  OFFLINE", 11, SUAVE, True)
        self.status_camera.pack(side="right")
        self.area_video = ctk.CTkLabel(camera, text="ZelarIA\n\nInicie a câmera para verificar os EPIs", font=("Segoe UI", 18), text_color=SUAVE, fg_color=FUNDO, corner_radius=12)
        self.area_video.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))
        self.label(camera, "P1, P2… identificam pessoas apenas no quadro atual.", 10, SUAVE).grid(row=2, column=0, padx=16, pady=(0, 8))
        controles = ctk.CTkFrame(barra, fg_color="transparent")
        controles.grid(row=0, column=3, sticky="nsew")
        controles.grid_columnconfigure(0, weight=1)
        controles.grid_rowconfigure((0, 1), weight=1)
        self.iniciar = ctk.CTkButton(controles, text="▶  Iniciar inspeção", height=40, fg_color="#B8D9BF", hover_color="#A2C7AA", text_color=FUNDO, text_color_disabled=PAINEL, state="disabled", command=self.iniciar_camera)
        self.iniciar.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        self.parar = ctk.CTkButton(controles, text="■  Encerrar", height=40, fg_color="#ECECEC", hover_color="#ACBBC6", text_color=FUNDO, text_color_disabled=PAINEL, state="disabled", command=self.parar_camera)
        self.parar.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

        direito = self.painel(dashboard)
        direito.configure(fg_color=CHECKLIST_FUNDO, border_color=CHECKLIST_FUNDO)
        direito.grid(row=0, column=0, sticky="nsew")
        self.label(direito, "Checklist de proteção", 19, CHECKLIST_TEXTO, negrito=True).pack(anchor="w", padx=20, pady=(12, 2))
        self.label(direito, "TODOS OS 6 EPIs SÃO OBRIGATÓRIOS", 10, CHECKLIST_TEXTO, True).pack(anchor="w", padx=20, pady=(0, 6))
        self.resultado = ctk.CTkLabel(direito, text="AGUARDANDO INSPEÇÃO", height=40, fg_color=CHECKLIST_ITEM, corner_radius=10, font=("Segoe UI", 16, "bold"), text_color=CHECKLIST_TEXTO)
        self.resultado.pack(fill="x", padx=16)
        self.detalhe = self.label(direito, "Inicie o monitoramento para avaliar a cena.", 11, CHECKLIST_TEXTO, wraplength=280, justify="left")
        self.detalhe.pack(anchor="w", padx=20, pady=(4, 6))
        self.checks = {}
        self.icones_epi = {epi: criar_icone_epi(epi) for epi in EPIS}
        for epi in EPIS:
            linha = ctk.CTkFrame(direito, fg_color=CHECKLIST_ITEM, border_color=CHECKLIST_ITEM, border_width=1, corner_radius=8)
            linha.pack(fill="x", padx=12, pady=2)
            linha.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(linha, text="", image=self.icones_epi[epi], width=32).grid(row=0, column=0, rowspan=2, padx=(8, 6), pady=4)
            self.label(linha, epi, 12, CHECKLIST_TEXTO, negrito=True).grid(row=0, column=1, sticky="w", pady=(3, 0))
            status = self.label(linha, "—  Aguardando", 10, CHECKLIST_TEXTO, True, height=18)
            status.grid(row=1, column=1, sticky="w", pady=(0, 3))
            self.checks[epi] = status
        self.label(direito, "RESULTADO POR PESSOA", 10, CHECKLIST_TEXTO, True).pack(anchor="w", padx=20, pady=(6, 3))
        self.lista_pessoas = ctk.CTkTextbox(direito, height=45, fg_color=CHECKLIST_ITEM, font=("Segoe UI", 11), text_color=CHECKLIST_TEXTO, wrap="word")
        self.lista_pessoas.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.escrever(self.lista_pessoas, "Nenhuma avaliação disponível.")
        self.modelo_label = self.label(dashboard, "Carregando modelo...", 10, SUAVE, wraplength=320, justify="left")
        self.modelo_label.grid(row=1, column=0, sticky="ew", padx=8, pady=(5, 0))

    @staticmethod
    def escrever(widget, texto):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", texto)
        widget.configure(state="disabled")

    def carregar_modelo(self):
        caminho = PASTA_PROJETO / "best.pt"
        if not caminho.is_file():
            caminho = PASTA_PROJETO / "yolo11n.pt"
        try:
            if not caminho.is_file():
                raise FileNotFoundError("Adicione best.pt ou yolo11n.pt à pasta do projeto.")
            self.modelo = YOLO(str(caminho))
            self.classes_modelo = {normalizar(n) for n in self.modelo.names.values()}
            faltam = [e for e, aliases in EPIS.items() if not aliases & self.classes_modelo]
            texto = f"MODELO: {caminho.name}  •  "
            texto += "Não valida: " + ", ".join(faltam) if faltam else "Classes dos 6 EPIs disponíveis."
            self.modelo_label.configure(text=texto, text_color=AMARELO if faltam else VERDE)
            self.iniciar.configure(state="normal")
        except Exception as erro:
            self.modelo_label.configure(text=f"Falha ao carregar modelo: {erro}", text_color=VERMELHO)

    def iniciar_camera(self):
        if self.camera_ativa or self.modelo is None:
            return
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.camera.release()
            self.camera = None
            messagebox.showerror("ZelarIA | Câmera", "Não foi possível acessar a webcam.")
            return
        self.camera_ativa = True
        self.iniciar.configure(state="disabled")
        self.parar.configure(state="normal")
        self.status_camera.configure(text="●  AO VIVO", text_color=VERDE)
        self.atualizar_video()

    def mostrar_avaliacao(self, pessoas, presentes, suportados):
        total = len(pessoas)
        completas = all(suportados.values())
        liberadas = sum(len(e) == len(EPIS) for e in presentes) if completas else 0
        pendentes = total - liberadas
        for chave, valor in (("pessoas", total), ("liberadas", liberadas), ("pendentes", pendentes)):
            self.cards[chave].configure(text=str(valor))
        if not total:
            titulo, cor, fundo = "AGUARDANDO PESSOA", SUAVE, FUNDO
            detalhe = "Posicione a pessoa inteira no campo de visão da câmera."
        elif not completas:
            titulo, cor, fundo = "VALIDAÇÃO INDISPONÍVEL", AMARELO, FUNDO
            detalhe = "O modelo não reconhece todos os EPIs. Acesso não liberado."
        elif pendentes:
            titulo, cor, fundo = "ATENÇÃO • NÃO LIBERADO", VERMELHO, FUNDO
            detalhe = f"{pendentes} de {total} pessoa(s) com EPIs não confirmados."
        else:
            titulo, cor, fundo = "✓  OK — LIBERADO", VERDE, FUNDO
            detalhe = "Os seis EPIs foram detectados em cada pessoa neste quadro."
        self.resultado.configure(text=titulo, text_color=CHECKLIST_TEXTO, fg_color=CHECKLIST_ITEM)
        self.detalhe.configure(text=detalhe)
        for epi, label in self.checks.items():
            quantidade = sum(epi in itens for itens in presentes)
            if not total:
                texto, cor_item = "—  Aguardando", SUAVE
            elif not suportados[epi]:
                texto, cor_item = "?  Sem suporte", AMARELO
            elif quantidade == total:
                texto, cor_item = f"✓  OK  {quantidade}/{total}", VERDE
            else:
                texto, cor_item = f"!  Não confirmado  {quantidade}/{total}", VERMELHO
            label.configure(text=texto, text_color=CHECKLIST_TEXTO)
        linhas = []
        for i, itens in enumerate(presentes, 1):
            faltam = [e for e in EPIS if e not in itens and suportados[e]]
            sem_suporte = [e for e in EPIS if not suportados[e]]
            partes = []
            if faltam:
                partes.append("Não confirmado: " + ", ".join(faltam))
            if sem_suporte:
                partes.append("Sem suporte: " + ", ".join(sem_suporte))
            linhas.append(f"P{i}  •  " + (" | ".join(partes) if partes else "✓ OK — Liberado"))
        self.escrever(self.lista_pessoas, "\n\n".join(linhas) or "Nenhuma pessoa detectada.")

    def atualizar_video(self):
        self.after_id = None
        if not self.camera_ativa:
            return
        try:
            sucesso, frame = self.camera.read()
            if not sucesso:
                raise RuntimeError("Falha na captura da câmera.")
            resultado = self.modelo.predict(source=frame, conf=CONFIANCA_MINIMA, verbose=False)[0]
            deteccoes = []
            if resultado.boxes is not None:
                for classe, caixa in zip(resultado.boxes.cls.tolist(), resultado.boxes.xyxy.tolist()):
                    deteccoes.append((normalizar(resultado.names[int(classe)]), caixa))
            pessoas, presentes, suportados = avaliar_pessoas(deteccoes, self.classes_modelo)
            self.mostrar_avaliacao(pessoas, presentes, suportados)
            imagem = resultado.plot()
            for i, (_, caixa) in enumerate(pessoas):
                x1, y1, x2, y2 = map(int, caixa)
                ok = all(suportados.values()) and len(presentes[i]) == len(EPIS)
                cor = (236, 236, 236) if ok else (157, 125, 102)
                cv2.rectangle(imagem, (x1, y1), (x2, y2), cor, 2)
                cv2.putText(imagem, f"P{i + 1}: {'OK' if ok else 'PENDENTE'}", (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, cor, 2)
            pil = Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
            pil.thumbnail((max(100, self.area_video.winfo_width() - 8), max(100, self.area_video.winfo_height() - 8)))
            imagem_tk = ctk.CTkImage(light_image=pil, dark_image=pil, size=pil.size)
            self.area_video.configure(image=imagem_tk, text="")
            self.area_video.image = imagem_tk
        except Exception as erro:
            self.parar_camera()
            self.resultado.configure(text="FALHA NA INSPEÇÃO", text_color=CHECKLIST_TEXTO, fg_color=CHECKLIST_ITEM)
            self.detalhe.configure(text="Verificação interrompida. Nenhuma liberação ativa.")
            return
        self.after_id = self.after(40, self.atualizar_video)

    def parar_camera(self):
        self.camera_ativa = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.area_video.configure(image=None, text="ZelarIA\n\nInspeção encerrada")
        self.area_video.image = None
        self.status_camera.configure(text="●  OFFLINE", text_color=SUAVE)
        self.iniciar.configure(state="normal" if self.modelo else "disabled")
        self.parar.configure(state="disabled")
        self.resultado.configure(text="INSPEÇÃO ENCERRADA", text_color=CHECKLIST_TEXTO, fg_color=CHECKLIST_ITEM)
        self.detalhe.configure(text="Inicie a câmera para uma nova verificação.")
        for label in self.checks.values():
            label.configure(text="—  Aguardando", text_color=CHECKLIST_TEXTO)
        for chave in ("pessoas", "liberadas", "pendentes"):
            self.cards[chave].configure(text="0")
        self.escrever(self.lista_pessoas, "Nenhuma avaliação ativa.")

    def fechar(self):
        self.parar_camera()
        self.destroy()


if __name__ == "__main__":
    Aplicacao().mainloop()
