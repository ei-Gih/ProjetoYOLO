"""Aviso visual não bloqueante, rearmado somente após perder a condição de liberação."""
import tkinter as tk
import customtkinter as ctk


class AvisoAcesso:
    def __init__(self, pai, quadros_minimos=2, validade_ms=3000):
        self.pai = pai
        self.quadros_minimos = quadros_minimos
        self.validade_ms = validade_ms
        self.janela = None
        self.contagem = 0
        self.exibido = False
        self.expiracao = None
        self.detalhe_atual = None

    def atualizar(self, liberado, detalhe):
        if not liberado:
            self.resetar()
            return
        if self.detalhe_atual is not None and self.detalhe_atual != detalhe:
            self.resetar()
        self.detalhe_atual = detalhe
        self.contagem += 1
        if self.expiracao is not None:
            self.pai.after_cancel(self.expiracao)
        self.expiracao = self.pai.after(self.validade_ms, self.resetar)
        if self.contagem >= self.quadros_minimos and not self.exibido:
            self.exibido = True
            self.abrir(detalhe)

    def abrir(self, detalhe):
        janela = ctk.CTkToplevel(self.pai)
        self.janela = janela
        janela.title('ZelarIA — Acesso liberado')
        janela.configure(fg_color='#11823B')
        janela.transient(self.pai)
        largura = min(900, max(600, self.pai.winfo_screenwidth() - 100))
        altura = min(440, max(340, self.pai.winfo_screenheight() - 100))
        x = max(0, (self.pai.winfo_screenwidth() - largura) // 2)
        y = max(0, (self.pai.winfo_screenheight() - altura) // 2)
        janela.geometry(f'{largura}x{altura}+{x}+{y}')
        janela.resizable(False, False)
        janela.protocol('WM_DELETE_WINDOW', self.fechar_aviso)
        ctk.CTkLabel(janela, text='ACESSO LIBERADO', text_color='white',
                     font=('Segoe UI', 44, 'bold')).pack(expand=True, padx=28, pady=(55, 8))
        ctk.CTkLabel(janela, text=detalhe, text_color='white', wraplength=largura-80,
                     font=('Segoe UI', 20)).pack(padx=35, pady=(0, 25))
        ctk.CTkButton(janela, text='Fechar aviso', command=self.fechar_aviso,
                      fg_color='white', text_color='#0B5727', hover_color='#D7F1DE',
                      width=210, height=44, font=('Segoe UI', 16, 'bold')).pack(pady=(0, 35))
        janela.lift()

    def fechar_aviso(self):
        if self.janela is not None:
            try:
                self.janela.destroy()
            except tk.TclError:
                pass
            self.janela = None

    def resetar(self):
        if self.expiracao is not None:
            try:
                self.pai.after_cancel(self.expiracao)
            except tk.TclError:
                pass
            self.expiracao = None
        self.contagem = 0
        self.exibido = False
        self.detalhe_atual = None
        self.fechar_aviso()
