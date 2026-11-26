import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ultralytics.engine.results import Results
from test.detector import DetectorEPIs

class Modelo:
    def __init__(self,path):
        self.principal=Path(path).name=='sh17_ppe.pt'
        self.names={0:'person'} if self.principal else {0:'mascara',1:'oculos',2:'capacete'}
    def predict(self,frame,**kw):
        boxes=[] if self.principal else [[200,200,300,300,.9,0],[200,200,300,300,.9,1],[200,200,300,300,.8,1],[180,180,185,185,.9,2],[0,0,100,100,.9,2]]
        return [Results(frame,path='teste',names=self.names,boxes=np.array(boxes,dtype=np.float32).reshape(-1,6))]

class Contexto(unittest.TestCase):
    def test_preserva_classes_restabelece_caixas_e_rejeita_borda_e_pequenos(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            p=Path(d)
            (p/'calibracao.json').write_text(json.dumps({'contexto_proximo':{'ativo':True,'arquivo':'epis.pt','classes':['mascara','oculos','capacete'],'escala':.45}}))
            det=DetectorEPIs(p/'sh17_ppe.pt')
            _,pred=det.prever(np.zeros((640,640,3),dtype=np.uint8))
            self.assertEqual([n for n,b in pred],['mascara','oculos'])
            np.testing.assert_allclose(pred[0][1],[53.333333,53.333333,275.555556,275.555556],atol=.001)
            self.assertTrue(all(r['origem']=='contexto' for r in det.detalhes))
