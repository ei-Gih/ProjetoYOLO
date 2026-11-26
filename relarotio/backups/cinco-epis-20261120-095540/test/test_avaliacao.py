import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import multiprocessing as mp
import time
from pathlib import Path
from test.avaliacao import EPIS, avaliar_pessoas
from main import Aplicacao

class Widget:
    def __init__(self): self.valor = {}
    def configure(self, **kw): self.valor.update(kw)
    def delete(self, *a): pass
    def insert(self, *a): pass

class Regressao(unittest.TestCase):
    def test_seis_epis_e_duas_pessoas(self):
        classes = {next(iter(a)) for a in EPIS.values()}
        det = [('person', (0,0,100,200)), ('person',(200,0,300,200))]
        det += [(c,(20,20,30,30)) for c in classes]
        pessoas, presentes, suporte = avaliar_pessoas(det,classes)
        self.assertEqual([len(p) for p in presentes], [6,0])
        ui = type('UI',(),{})()
        ui.cards = {c:Widget() for c in ('pessoas','liberadas','pendentes')}
        ui.resultado=Widget(); ui.detalhe=Widget(); ui.lista_pessoas=Widget()
        ui.checks={e:Widget() for e in EPIS}; ui.escrever=lambda *a:None
        Aplicacao.mostrar_avaliacao(ui,pessoas,presentes,suporte)
        self.assertEqual([ui.cards[c].valor['text'] for c in ('pessoas','liberadas','pendentes')], ['2','6','6'])
        Aplicacao.mostrar_avaliacao(ui,[],[],suporte)
        self.assertTrue(all(w.valor['text']=='0' for w in ui.cards.values()))

    def test_colete_nao_confirma_jaleco(self):
        _, presentes, suporte = avaliar_pessoas([('person',(0,0,100,200)),('safety-vest',(10,20,80,120))], {'person','safety-vest','medical-suit'})
        self.assertFalse(suporte['Jaleco']); self.assertEqual(presentes,[set()])

    def test_epi_ambiguo_ou_solto_nao_confirma(self):
        _, presentes, _ = avaliar_pessoas([('person',(0,0,100,200)),('person',(0,0,100,200)),('helmet',(10,10,20,20)),('gloves',(300,300,320,320))], {'helmet','gloves'})
        self.assertEqual(presentes,[set(),set()])

    def test_duas_luvas_contam_um_tipo(self):
        _, presentes, _ = avaliar_pessoas([('person',(0,0,100,200)), ('gloves',(10,80,20,100)), ('gloves',(70,80,80,100))], {'gloves'})
        self.assertEqual(presentes,[{'Luvas'}])

    def test_video_travado_interrompe_inspecao(self):
        from queue import Empty
        from unittest.mock import Mock
        ui=type('UI',(),{})()
        ui.camera_ativa=True; ui.fechando=False; ui.recebeu_quadro=True
        ui.ultimo_quadro=time.monotonic()-31; ui.fila=Mock()
        ui.fila.get_nowait.side_effect=Empty
        ui.falha_inspecao=Mock()
        Aplicacao.atualizar_video(ui)
        ui.falha_inspecao.assert_called_once()

    def test_encerramento_processo_travado(self):
        ctx=mp.get_context('spawn')
        ui=type('UI',(),{})(); ui.sinal_parar=ctx.Event(); ui.fila=ctx.Queue(2)
        ui.processo=ctx.Process(target=time.sleep,args=(60,)); ui.processo.start()
        pid=ui.processo.pid
        Aplicacao.encerrar_processo(ui)
        self.assertIsNone(ui.processo); self.assertIsNone(ui.fila)
        self.assertNotIn(pid,[p.pid for p in mp.active_children()])

if __name__=='__main__': unittest.main()
