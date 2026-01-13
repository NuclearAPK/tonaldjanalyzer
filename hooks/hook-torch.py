# Runtime hook to fix torch DLL loading on Windows
import os
import sys
import ctypes

if sys.platform == 'win32':
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))

    # Set environment variables
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # PyInstaller 6+ puts files in _internal subfolder
    # sys._MEIPASS already points to _internal for folder-based builds
    torch_lib = os.path.join(base_path, 'torch', 'lib')
    torch_bin = os.path.join(base_path, 'torch', 'bin')

    dll_dirs = [base_path]
    if os.path.exists(torch_lib):
        dll_dirs.append(torch_lib)
    if os.path.exists(torch_bin):
        dll_dirs.append(torch_bin)

    # Add to PATH
    os.environ['PATH'] = os.pathsep.join(dll_dirs) + os.pathsep + os.environ.get('PATH', '')

    # Add DLL directories (Windows 10+)
    if hasattr(os, 'add_dll_directory'):
        for d in dll_dirs:
            try:
                os.add_dll_directory(d)
            except OSError:
                pass

    # Pre-load critical DLLs in correct order using LOAD_WITH_ALTERED_SEARCH_PATH
    if os.path.exists(torch_lib):
        dll_load_order = [
            'fbgemm.dll',
            'asmjit.dll',
            'uv.dll',
            'libiomp5md.dll',
            'c10.dll',
            'torch_cpu.dll',
            'torch.dll',
        ]

        for dll_name in dll_load_order:
            dll_path = os.path.join(torch_lib, dll_name)
            if os.path.exists(dll_path):
                try:
                    ctypes.WinDLL(dll_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    try:
                        ctypes.CDLL(dll_path)
                    except OSError:
                        pass
