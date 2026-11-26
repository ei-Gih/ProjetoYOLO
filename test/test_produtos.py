import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ultralytics.engine.results import Results
from test.detector import DetectorEPIs,mesmo_objeto
from test.test_cabeca import Modelo


class Produto(Modelo):
    def __init__(self,path):
        super().__init__(path)
        if self.name=='produtos.pt':
            self.names={0:'person',1:'capacete',2:'luvas',3:'mascara',4:'oculos',5:'protetor-auricular'}
    def predict(self,frame,classes=None,conf=.4,**kw):
        if self.name!='produtos.pt':return super().predict(frame,classes=classes,conf=conf,**kw)
        boxes=[[15,0,35,15,.95,1],[65,0,85,15,.9,1],[0,0,100,100,.99,0]]
        boxes=[b for b in boxes if b[4]>=conf and (classes is None or b[5] in classes)]
        return [Results(frame,path='teste',names=self.names,boxes=np.array(boxes,dtype=np.float32).reshape(-1,6))]


class Produtos(unittest.TestCase):
    def test_complemento_preserva_original_deduplica_alias_e_nao_injeta_pessoa(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Produto):
            root=Path(d)
            (root/'calibracao.json').write_text(json.dumps({'especialista_maos_ouvidos':True,
                'especialistas':[{'arquivo':'produtos.pt','classes':['capacete'],'complementar':True}]}))
            _,det=DetectorEPIs(root/'sh17_ppe.pt').prever(np.zeros((100,100,3),dtype=np.uint8))
            self.assertEqual(sum(n=='person' for n,b in det),1)
            self.assertEqual(sum(n=='helmet' for n,b in det),1)
            self.assertEqual(sum(n=='capacete' for n,b in det),1)
            self.assertIn(('capacete',[65.,0.,85.,15.]),det)

    def test_duas_luvas_distintas_nao_sao_fundidas(self):
        self.assertFalse(mesmo_objeto('luvas',[50,50,70,70],[('gloves',[10,10,30,30])]))
        self.assertTrue(mesmo_objeto('luvas',[10,10,30,30],[('gloves',[11,11,30,30])]))

    def test_pessoa_nao_pode_ser_ativada_como_epi_complementar(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Produto):
            root=Path(d)
            (root/'calibracao.json').write_text(json.dumps({'especialista_maos_ouvidos':True,
                'especialistas':[{'arquivo':'produtos.pt','classes':['person'],'complementar':True}]}))
            with self.assertRaises(ValueError):DetectorEPIs(root/'sh17_ppe.pt')
