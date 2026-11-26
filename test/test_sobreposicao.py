import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from ultralytics.engine.results import Results
from test.detector import DetectorEPIs,mesmo_objeto
from test.avaliacao import avaliar_pessoas

class Modelo:
    names={0:'person',1:'mascara',2:'oculos'}
    def __init__(self,path):pass
    def predict(self,frame,**kwargs):
        assert kwargs['agnostic_nms'] is False, 'A supressão não pode eliminar classes diferentes.'
        return [Results(frame,path='teste',names=self.names,boxes=np.array([
            [0,0,100,100,.95,0],[20,10,70,40,.9,1],[20,10,70,40,.9,2]],dtype=np.float32))]

class Sobreposicao(unittest.TestCase):
    def test_remove_repeticao_de_epi_sem_remover_oculos_ou_pessoas(self):
        class Repetido(Modelo):
            def predict(self,frame,**kwargs):
                return [Results(frame,path='teste',names=self.names,boxes=np.array([
                    [0,0,100,100,.95,0],[0,0,100,100,.94,0],
                    [20,10,70,40,.9,1],[20,5,70,35,.8,1],
                    [20,10,70,40,.9,2]],dtype=np.float32))]
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Repetido):
            root=Path(d)
            (root/'calibracao.json').write_text(json.dumps({'principal_supressao':{'mascara':.45,'person':.45}}))
            _,pred=DetectorEPIs(root/'sh17_ppe.pt').prever(np.zeros((100,100,3),dtype=np.uint8))
            self.assertEqual([n for n,b in pred],['person','person','mascara','oculos'])

    def test_mascara_e_oculos_sobrepostos_permanecem_separados(self):
        with tempfile.TemporaryDirectory() as d,patch('ultralytics.YOLO',Modelo):
            detector=DetectorEPIs(Path(d)/'sh17_ppe.pt')
            _,pred=detector.prever(np.zeros((100,100,3),dtype=np.uint8))
            _,presentes,_=avaliar_pessoas(pred,detector.classes)
            self.assertEqual(presentes,[{'Máscara','Óculos de segurança'}])
            self.assertFalse(mesmo_objeto('mascara',[20,10,70,40],[('oculos',[20,10,70,40])]))
