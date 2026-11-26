"""Captura e inferência isoladas; nenhuma chamada Tk neste processo."""
from queue import Full

def configurar_resolucao(camera, resolucao):
    """Solicita mais detalhe; setters não suportados mantêm o modo aceito pela webcam."""
    import cv2
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolucao[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucao[1])
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def executar(modelo_path, fila, parar, confianca):
    import cv2
    from test.detector import DetectorEPIs
    camera = None
    fila.cancel_join_thread()

    def enviar(tipo, dados):
        while not parar.is_set():
            try:
                fila.put((tipo, dados), timeout=0.1)
                return True
            except Full:
                continue
        return False

    try:
        detector = DetectorEPIs(modelo_path, confianca)
        if not enviar('classes', sorted(detector.classes)) or parar.is_set():
            return
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise RuntimeError('Não foi possível acessar a webcam. Verifique conexão e permissões.')
        configurar_resolucao(camera, detector.camera_resolucao)
        while not parar.is_set():
            sucesso, frame = camera.read()
            if not sucesso:
                raise RuntimeError('Falha na captura da câmera.')
            imagem, deteccoes = detector.prever(frame)
            if not enviar('quadro', (imagem, deteccoes)):
                break
    except Exception as erro:
        enviar('erro', str(erro))
        parar.wait(5)
    finally:
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()
