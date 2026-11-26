import unittest
from unittest.mock import Mock,call
import cv2
from test.camera_worker import configurar_resolucao

class Resolucao(unittest.TestCase):
    def test_tenta_hd_sem_falhar_quando_backend_nao_suporta_setter(self):
        camera=Mock();camera.set.return_value=False
        configurar_resolucao(camera,(1280,720))
        self.assertEqual(camera.set.call_args_list,[call(cv2.CAP_PROP_FRAME_WIDTH,1280),
            call(cv2.CAP_PROP_FRAME_HEIGHT,720),call(cv2.CAP_PROP_BUFFERSIZE,1)])
