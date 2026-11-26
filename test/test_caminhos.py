import unittest
from pathlib import Path
from test.caminhos import RAIZ, DATASETS, TREINOS, BASES
from test import testar_modelo, treinar_cabeca, treinar_plugues, treinar_variacoes
import main

class Caminhos(unittest.TestCase):
    def test_scripts_compartilham_a_raiz_da_aplicacao(self):
        self.assertEqual(RAIZ,Path(main.__file__).resolve().parent)
        self.assertEqual(testar_modelo.RAIZ,RAIZ)
        self.assertEqual(treinar_cabeca.ROOT,RAIZ)
        self.assertEqual(treinar_plugues.ROOT,RAIZ)
        self.assertEqual(treinar_variacoes.RAIZ,RAIZ)
        self.assertEqual(DATASETS,RAIZ/'dataset')
        self.assertEqual(TREINOS,RAIZ/'relarotio/treinos')
        self.assertEqual(BASES,RAIZ/'test/bases')

if __name__=='__main__':unittest.main()
