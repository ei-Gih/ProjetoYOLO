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
from test.detector import DetectorEPIs


class Modelo:
    def __init__(self,path):
        self.name=Path(path).name
        self.names=({0:'protetor-auricular',1:'mascara',2:'capacete'} if self.name=='cabeca.pt'
                    else {0:'person',1:'ear-mufs',2:'face-mask',3:'helmet',4:'gloves',5:'glasses'})
    def predict(self,frame,classes=None,conf=.4,**kw):
        boxes=([[15,15,20,25,.8,0],[20,25,30,35,.8,1],[15,0,35,15,.8,2]] if self.name=='cabeca.pt'
               else [[0,0,60,100,.95,0],[15,15,20,25,.9,1],[20,25,30,35,.9,2],[15,0,35,15,.9,3],[30,50,40,60,.9,4],[15,15,30,20,.9,5]])
        boxes=[b for b in boxes if b[4]>=conf and (classes is None or b[5] in classes)]
        return [Results(frame,path='teste',names=self.names,boxes=np.array(boxes,dtype=np.float32).reshape(-1,6))]


class Cabeca(unittest.TestCase):
    def detector(self,path,classes):
        (path/'calibracao.json').write_text(json.dumps({'especialista_maos_ouvidos':True,'especialistas':[
            {'arquivo':'cabeca.pt','classes':classes,'limiares':{'protetor-auricular':.7,'mascara':.85,'capacete':.7}}]}))
        return DetectorEPIs(path/'sh17_ppe.pt')

    def test_tres_classes_substituidas_sem_duplicar_e_limiares_independentes(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            det=self.detector(Path(d),['protetor-auricular','mascara','capacete'])
            _,boxes=det.prever(np.zeros((100,100,3),dtype=np.uint8))
            self.assertEqual([n for n,_ in boxes],['person','gloves','glasses','capacete'])

    def test_ativacao_somente_capacete_preserva_mascara_e_ouvidos(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            det=self.detector(Path(d),['capacete'])
            _,boxes=det.prever(np.zeros((100,100,3),dtype=np.uint8))
            self.assertEqual([n for n,_ in boxes],['person','face-mask','gloves','glasses','capacete'])

    def test_classe_inexistente_nao_e_silenciosamente_ignorada(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            with self.assertRaises(ValueError):self.detector(Path(d),['luvas'])

    def test_limiar_principal_apenas_nos_pesos_correspondentes(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            path=Path(d)
            (path/'calibracao.json').write_text(json.dumps({'principal':{'helmet':.95}}))
            frame=np.zeros((100,100,3),dtype=np.uint8)
            _,calibrado=DetectorEPIs(path/'sh17_ppe.pt').prever(frame)
            _,proprio=DetectorEPIs(path/'best.pt').prever(frame)
            self.assertNotIn('helmet',[n for n,b in calibrado])
            self.assertIn('helmet',[n for n,b in proprio])

    def test_resolucao_configurada_chega_a_inferencia(self):
        calls=[]
        class Espiao(Modelo):
            def predict(self,frame,**kw):
                calls.append((self.name,kw['imgsz']))
                return super().predict(frame,**kw)
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Espiao):
            path=Path(d)
            (path/'calibracao.json').write_text(json.dumps({'especialista_maos_ouvidos':True,
                'especialistas':[{'arquivo':'cabeca.pt','classes':['capacete'],'imgsz':640}]}))
            DetectorEPIs(path/'sh17_ppe.pt').prever(np.zeros((100,100,3),dtype=np.uint8))
            self.assertIn(('cabeca.pt',640),calls)

    def test_configuracao_antiga_de_protetor_e_ignorada_sem_carregar_pesos(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            path=Path(d)
            (path/'calibracao.json').write_text(json.dumps({'especialista_maos_ouvidos':True,
                'especialistas':[{'arquivo':'inexistente.pt','classes':['earplug']}]}))
            det=DetectorEPIs(path/'sh17_ppe.pt')
            self.assertEqual(len(det.modelos),1)
            _,boxes=det.prever(np.zeros((100,100,3),dtype=np.uint8))
            self.assertNotIn('ear-mufs',[n for n,b in boxes])


if __name__=='__main__':unittest.main()
