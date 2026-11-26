import sys
from pathlib import Path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ultralytics.engine.results import Results
from test.detector import DetectorEPIs, limiar_calibrado


class ModeloFalso:
    def __init__(self, caminho):
        self.nome = Path(caminho).name
        self.names = ({0: 'luvas', 1: 'protetor-auricular'} if self.nome.startswith('maos_')
                      else {0: 'person', 1: 'gloves', 2: 'ear-mufs', 3: 'helmet'})

    def predict(self, frame, classes=None, conf=.4, **kw):
        if self.nome.startswith('maos_'):
            boxes = [[10,30,20,40,.69,0], [30,30,40,40,.71,0], [10,5,20,15,.36,1]]
        else:
            boxes = [[0,0,70,100,.9,0], [10,30,20,40,.99,1], [10,5,20,15,.99,2], [10,0,30,20,.8,3]]
        boxes = [b for b in boxes if b[4] >= conf and (classes is None or b[5] in classes)]
        return [Results(frame, path='teste', names=self.names, boxes=np.array(boxes,dtype=np.float32).reshape(-1,6))]


class Calibracao(unittest.TestCase):
    def test_modelo_original_nao_duplica_especialista_e_filtra_por_classe(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz/'calibracao.json').write_text(json.dumps({'especialista_maos_ouvidos':True,
                'maos_ouvidos':{'luvas':.7,'protetor-auricular':.35}}))
            with patch('ultralytics.YOLO', ModeloFalso):
                detector = DetectorEPIs(raiz/'sh17_ppe.pt')
                _, det = detector.prever(np.zeros((100,100,3),dtype=np.uint8))
            self.assertEqual([n for n,b in det], ['person','helmet','luvas','protetor-auricular'])
            self.assertEqual(det[2][1][0], 30)

    def test_best_proprio_nao_recebe_calibracao_de_outros_pesos(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz/'calibracao.json').write_text(json.dumps({'especialista_maos_ouvidos':True}))
            with patch('ultralytics.YOLO', ModeloFalso):
                detector = DetectorEPIs(raiz/'best.pt')
                self.assertEqual(len(detector.modelos), 1)

    def test_ajuste_explicito_preserva_calibracao(self):
        self.assertAlmostEqual(limiar_calibrado('luvas',{'luvas':.7},.5),.8)
        self.assertAlmostEqual(limiar_calibrado('helmet',{},.4),.4)

    def test_ativacao_parcial_preserva_luvas_do_principal(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz/'calibracao.json').write_text(json.dumps({'especialista_maos_ouvidos':True,
                'classes_maos_ouvidos':['protetor-auricular'], 'maos_ouvidos':{'protetor-auricular':.35}}))
            with patch('ultralytics.YOLO', ModeloFalso):
                detector = DetectorEPIs(raiz/'sh17_ppe.pt')
                _, det = detector.prever(np.zeros((100,100,3),dtype=np.uint8))
            self.assertEqual([n for n,b in det], ['person','gloves','helmet','protetor-auricular'])

    def test_especialistas_distintos_sem_duplicacao(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            (raiz/'calibracao.json').write_text(json.dumps({'especialista_maos_ouvidos':True,
                'especialistas':[
                    {'arquivo':'maos_conservador.pt','classes':['luvas'],'limiares':{'luvas':.7}},
                    {'arquivo':'maos_ouvidos.pt','classes':['protetor-auricular'],'limiares':{'protetor-auricular':.35}}]}))
            with patch('ultralytics.YOLO', ModeloFalso):
                detector = DetectorEPIs(raiz/'sh17_ppe.pt')
                _, det = detector.prever(np.zeros((100,100,3),dtype=np.uint8))
            self.assertEqual([n for n,b in det], ['person','helmet','luvas','protetor-auricular'])


if __name__ == '__main__':
    unittest.main()
