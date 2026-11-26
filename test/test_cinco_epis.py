import unittest
from test.avaliacao import EPIS, EPIS_DESATIVADOS, avaliar_pessoas
from test import test_acesso


class CincoEPIs(unittest.TestCase):
    def test_lista_exata_e_acesso_sem_protetor(self):
        self.assertEqual(set(EPIS),{'Capacete','Máscara','Jaleco','Luvas','Óculos de segurança'})
        det=[('person',(0,0,100,200))]+[(next(iter(a)),(20,20,30,30)) for a in EPIS.values()]
        _,presentes,suporte=avaliar_pessoas(det,{d[0] for d in det})
        self.assertEqual(presentes,[set(EPIS)])
        self.assertTrue(test_acesso.Acesso().verificar(presentes,suporte))
        for epi in EPIS:
            self.assertFalse(test_acesso.Acesso().verificar([set(EPIS)-{epi}]))

    def test_protetores_nao_contam_nem_criam_pendencia(self):
        det=[('person',(0,0,100,200))]+[(n,(20,20,30,30)) for n in EPIS_DESATIVADOS]
        _,presentes,suporte=avaliar_pessoas(det,{d[0] for d in det})
        self.assertEqual(presentes,[set()])
        self.assertEqual(len(suporte),5)
        self.assertNotIn('Protetor auricular',suporte)
