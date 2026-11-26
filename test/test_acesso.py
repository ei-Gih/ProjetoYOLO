import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
from unittest.mock import Mock
from test.aviso_acesso import AvisoAcesso
from main import Aplicacao
from test.avaliacao import EPIS
from test.test_avaliacao import Widget

class Relogio:
    def __init__(self):self.proximo=0;self.callbacks={}
    def after(self,ms,callback):
        self.proximo+=1;self.callbacks[self.proximo]=callback;return self.proximo
    def after_cancel(self,ident):self.callbacks.pop(ident,None)

class AvisoTeste(AvisoAcesso):
    def __init__(self):super().__init__(Relogio());self.aberturas=[]
    def abrir(self,detalhe):self.aberturas.append(detalhe);self.janela=Mock()

class Acesso(unittest.TestCase):
    def test_exige_confirmacoes_consecutivas(self):
        a=AvisoTeste();a.atualizar(True,'ok');self.assertEqual(a.aberturas,[])
        a.atualizar(False,'');a.atualizar(True,'ok');self.assertEqual(a.aberturas,[])
        a.atualizar(True,'ok');self.assertEqual(a.aberturas,['ok'])

    def test_nao_reabre_a_cada_quadro_ou_apos_fechar_manualmente(self):
        a=AvisoTeste();a.atualizar(True,'ok');a.atualizar(True,'ok');a.fechar_aviso()
        for _ in range(5):a.atualizar(True,'ok')
        self.assertEqual(len(a.aberturas),1)
        a.atualizar(False,'');a.atualizar(True,'ok');a.atualizar(True,'ok')
        self.assertEqual(len(a.aberturas),2)

    def test_fecha_se_condicao_desaparece_ou_quadro_expira(self):
        a=AvisoTeste();a.atualizar(True,'ok');a.atualizar(True,'ok');janela=a.janela
        a.pai.callbacks[a.expiracao]()
        janela.destroy.assert_called_once();self.assertIsNone(a.janela);self.assertEqual(a.contagem,0)
        a.atualizar(True,'ok');a.atualizar(True,'ok');janela=a.janela
        a.atualizar(False,'');janela.destroy.assert_called_once();self.assertFalse(a.pai.callbacks)

    def test_mudanca_da_pessoa_fecha_aviso_antigo(self):
        a=AvisoTeste();a.atualizar(True,'P1');a.atualizar(True,'P1');janela=a.janela
        a.atualizar(True,'P2');janela.destroy.assert_called_once()
        self.assertEqual(a.aberturas,['P1'])
        a.atualizar(True,'P2');self.assertEqual(a.aberturas,['P1','P2'])

    def verificar(self,presentes,suportados=None):
        ui=type('UI',(),{})();ui.cards={c:Widget() for c in ('pessoas','liberadas','pendentes')}
        ui.resultado=Widget();ui.detalhe=Widget();ui.lista_pessoas=Widget()
        ui.checks={e:Widget() for e in EPIS};ui.escrever=lambda *args:None;ui.aviso_acesso=Mock()
        pessoas=[('person',(0,0,100,100)) for _ in presentes]
        Aplicacao.mostrar_avaliacao(ui,pessoas,presentes,suportados or {e:True for e in EPIS})
        return ui.aviso_acesso.atualizar.call_args.args[0]

    def test_nao_libera_sem_pessoa_ou_sem_jaleco(self):
        self.assertFalse(self.verificar([]))
        self.assertFalse(self.verificar([set(EPIS)-{'Jaleco'}]))

    def test_nao_soma_epis_de_pessoas_distintas(self):
        self.assertFalse(self.verificar([set(EPIS)-{'Jaleco'},{'Jaleco'}]))
        self.assertTrue(self.verificar([set(EPIS),set()]))

    def test_libera_somente_quando_todos_estao_completos(self):
        self.assertTrue(self.verificar([set(EPIS)]))
        self.assertTrue(self.verificar([set(EPIS),set(EPIS)]))
        support={e:e!='Jaleco' for e in EPIS}
        self.assertFalse(self.verificar([set(EPIS)],support))

if __name__=='__main__':unittest.main()
