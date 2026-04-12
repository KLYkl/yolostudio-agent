from __future__ import annotations

import importlib
import sys
import tempfile
import types
from pathlib import Path

if __package__ in {None, ""}:
    repo_root = Path(__file__).resolve().parents[2]
    parent_root = repo_root.parent
    for candidate in (repo_root, parent_root):
        path = str(candidate)
        if path not in sys.path:
            sys.path.insert(0, path)


MODULE_NAME = 'yolostudio_agent.agent.server.services.predict_service'
UTIL_MODULES = [
    'utils',
    'utils.constants',
    'utils.file_utils',
    'utils.label_writer',
]
DEPENDENCY_MODULES = [
    'cv2',
    'numpy',
    'PIL',
    'PIL.Image',
    'PIL.ImageDraw',
]


def _install_dependency_stubs() -> dict[str, object | None]:
    originals = {name: sys.modules.get(name) for name in DEPENDENCY_MODULES}

    cv2_stub = types.ModuleType('cv2')
    numpy_stub = types.ModuleType('numpy')
    pil_stub = types.ModuleType('PIL')
    image_stub = types.ModuleType('PIL.Image')
    draw_stub = types.ModuleType('PIL.ImageDraw')

    pil_stub.UnidentifiedImageError = RuntimeError
    pil_stub.Image = image_stub
    pil_stub.ImageDraw = draw_stub
    image_stub.Image = object
    draw_stub.Draw = lambda image: None

    sys.modules['cv2'] = cv2_stub
    sys.modules['numpy'] = numpy_stub
    sys.modules['PIL'] = pil_stub
    sys.modules['PIL.Image'] = image_stub
    sys.modules['PIL.ImageDraw'] = draw_stub
    return originals


def main() -> None:
    original_modules = {name: sys.modules.pop(name, None) for name in [MODULE_NAME, *UTIL_MODULES]}
    original_dependencies = _install_dependency_stubs()

    try:
        with tempfile.TemporaryDirectory(prefix='predict_service_import_') as tmp:
            tmp_root = Path(tmp)
            utils_dir = tmp_root / 'utils'
            utils_dir.mkdir(parents=True, exist_ok=True)
            (utils_dir / '__init__.py').write_text('', encoding='utf-8')
            (utils_dir / 'constants.py').write_text(
                "IMAGE_EXTENSIONS = {'.jpg'}\nVIDEO_EXTENSIONS = {'.mp4'}\n",
                encoding='utf-8',
            )
            (utils_dir / 'file_utils.py').write_text(
                "from pathlib import Path\n"
                "def discover_files(source, suffixes):\n"
                "    return []\n"
                "def get_unique_dir(path):\n"
                "    return Path(path)\n",
                encoding='utf-8',
            )

            sys.path.insert(0, str(tmp_root))
            importlib.invalidate_caches()

            module = importlib.import_module(MODULE_NAME)
            out_path = tmp_root / 'labels' / 'sample.txt'
            module._write_yolo_txt_from_xyxy(
                out_path,
                [{'class_id': 2, 'xyxy': [0, 10, 20, 30]}],
                40,
                40,
            )

            assert out_path.exists(), out_path
            content = out_path.read_text(encoding='utf-8').strip()
            assert content == '2 0.250000 0.500000 0.500000 0.500000', content
            print('predict service label writer fallback ok')
    finally:
        sys.path = [item for item in sys.path if not item.startswith('/tmp/predict_service_import_')]
        sys.modules.pop(MODULE_NAME, None)
        for name, module in original_modules.items():
            if module is not None:
                sys.modules[name] = module
        for name in DEPENDENCY_MODULES:
            sys.modules.pop(name, None)
        for name, module in original_dependencies.items():
            if module is not None:
                sys.modules[name] = module


if __name__ == '__main__':
    main()
