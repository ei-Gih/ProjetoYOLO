import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ultralytics.engine.results import Results
from test.detector import DetectorEPIs

class Modelo:
    near=False
    calls=[]
    def __init__(self,path):
        self.name=Path(path).name
        self.names={0:'person'} if self.name=='sh17_ppe.pt' else {0:'mascara',1:'oculos'}
    def predict(self,frame,**kwargs):
        self.calls.append((self.name,frame.shape,kwargs))
        boxes=([[40,0,80,195,.9,0]] if self.near else [[40,40,80,140,.9,0]]) if self.name=='sh17_ppe.pt' else [[10,10,30,30,.9,0],[10,10,30,30,.9,1]]
        return [Results(frame,path='teste',names=self.names,boxes=np.array(boxes,dtype=np.float32))]

class Recorte(unittest.TestCase):
    def test_restaura_coordenadas_e_preserva_mascara_oculos(self):
        Modelo.calls=[];Modelo.near=False
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            path=Path(d)
            (path/'calibracao.json').write_text(json.dumps({'zoom_pessoas':{'ativo':True,'arquivo':'cabeca.pt','classes':['mascara','oculos']}}))
            det=DetectorEPIs(path/'sh17_ppe.pt')
            _,pred=det.prever(np.zeros((200,200,3),dtype=np.uint8))
            self.assertEqual([n for n,b in pred],['person','mascara','oculos'])
            self.assertEqual(pred[1][1],[46.,45.,66.,65.])
            self.assertEqual(Modelo.calls[-1][1][:2],(55,48))
            self.assertIs(Modelo.calls[-1][2]['agnostic_nms'],False)

    def test_nao_amplia_pessoa_proxima(self):
        Modelo.calls=[];Modelo.near=True
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            path=Path(d)
            (path/'calibracao.json').write_text(json.dumps({'zoom_pessoas':{'ativo':True,'arquivo':'cabeca.pt','classes':['mascara','oculos']}}))
            DetectorEPIs(path/'sh17_ppe.pt').prever(np.zeros((200,200,3),dtype=np.uint8))
            self.assertEqual(len(Modelo.calls),1)
        Modelo.near=False

    def test_recorte_desativado_por_padrao(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            self.assertIsNone(DetectorEPIs(Path(d)/'sh17_ppe.pt').zoom)
