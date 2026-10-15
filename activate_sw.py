#!/usr/bin/env python3
import win32com.client as win32
import time
from pathlib import Path

def _prop(obj, name, *args):
    """Read a SolidWorks API member."""
    attr = getattr(obj, name)
    if args:
        return attr(*args)
    return attr() if callable(attr) else attr

try:
    app = win32.GetActiveObject("SldWorks.Application")
except Exception as e:
    print(f"Erro ao conectar: {e}")
    exit(1)

app.Visible = True

model = app.ActiveDoc
if model:
    title = _prop(model, "GetTitle")
    print(f"Modelo ativo: {title}")

    # View isométrica
    model.ShowNamedView2("*Isometric", 7)
    time.sleep(0.3)
    _prop(model, "ViewZoomtofit2")
    time.sleep(0.3)

    print("SolidWorks ativado e visível em view isométrica")
else:
    print("Nenhum modelo aberto")
