#!/usr/bin/env python3
import win32com.client as win32
import win32gui
import win32con
import time

app = win32.GetActiveObject("SldWorks.Application")
app.Visible = True

# Procurar por janelas do SolidWorks
def enum_windows(hwnd, lParam):
    if "SldWorks" in win32gui.GetWindowText(hwnd):
        print(f"Encontrada janela: {win32gui.GetWindowText(hwnd)}")
        win32gui.SetForegroundWindow(hwnd)
        win32gui.ShowWindow(hwnd, win32con.SW_NORMAL)
        return False
    return True

win32gui.EnumWindows(enum_windows, None)
time.sleep(0.3)
print("SolidWorks trazido para frente")
