#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WoT Mod build script.
Compiles the Python 2.7 mod file and packages it into a .wotmod.
"""
import os
import sys
import shutil
import zipfile
import subprocess
import py_compile

MOD_NAME = 'dc.position'
MOD_VERSION = '1.0.0'
SOURCE_FILE = 'mod_dc_position.py'
TARGET_DIR = os.path.join('res', 'scripts', 'client', 'gui', 'mods')
OUTPUT_FILE = '{}_{}.wotmod'.format(MOD_NAME, MOD_VERSION)


def find_python27():
    """Try to find a Python 2.7 interpreter."""
    possible_paths = [
        'python2.7',
        'python2',
        'py -2.7',
        'py -2',
    ]

    if sys.platform == 'win32':
        possible_paths.extend([
            r'C:\Python27\python.exe',
            r'C:\Python27\pythonw.exe',
        ])

    for cmd in possible_paths:
        try:
            result = subprocess.run(
                [cmd, '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if '2.7' in result.stdout or '2.7' in result.stderr:
                return cmd
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            continue

    return None


def compile_with_python27(python_cmd, source_file):
    """Compile the mod using Python 2.7."""
    try:
        subprocess.run(
            [python_cmd, '-m', 'py_compile', source_file],
            check=True,
            cwd=os.path.dirname(os.path.abspath(source_file)) or '.'
        )
        print('OK: compiled with {}'.format(python_cmd))
        return True
    except subprocess.CalledProcessError as exc:
        print('ERROR: compile failed: {}'.format(exc))
        return False
    except Exception as exc:
        print('ERROR: compile error: {}'.format(exc))
        return False


def compile_with_current_python(source_file):
    """Compile using current Python (may be incompatible with WoT Python 2.7)."""
    try:
        py_compile.compile(source_file, doraise=True)
        print('OK: compiled with current Python (compatibility not guaranteed)')
        return True
    except py_compile.PyCompileError as exc:
        print('ERROR: compile failed: {}'.format(exc))
        return False
    except Exception as exc:
        print('ERROR: compile error: {}'.format(exc))
        return False


def build_mod():
    print('=' * 60)
    print('WoT Mod build')
    print('=' * 60)

    if not os.path.exists(SOURCE_FILE):
        print('ERROR: source file not found: {}'.format(SOURCE_FILE))
        return False

    os.makedirs(TARGET_DIR, exist_ok=True)
    print('OK: ensured directory: {}'.format(TARGET_DIR))

    python27 = find_python27()
    compiled_file = SOURCE_FILE + 'c'
    if python27:
        print('Found Python 2.7: {}'.format(python27))
        if not compile_with_python27(python27, SOURCE_FILE):
            print('WARN: Python 2.7 compile failed, trying current Python')
            if not compile_with_current_python(SOURCE_FILE):
                return False
    else:
        print('WARN: Python 2.7 not found, using current Python')
        print('      The .pyc may be incompatible with WoT (Python 2.7).')
        if not compile_with_current_python(SOURCE_FILE):
            return False

    if not os.path.exists(compiled_file):
        print('ERROR: compiled file not found: {}'.format(compiled_file))
        return False

    target_file = os.path.join(TARGET_DIR, os.path.basename(compiled_file))
    shutil.copy2(compiled_file, target_file)
    print('OK: copied to {}'.format(target_file))

    try:
        os.remove(compiled_file)
        print('OK: cleaned temporary file: {}'.format(compiled_file))
    except Exception:
        pass

    print('\nPackaging: {}'.format(OUTPUT_FILE))
    try:
        with zipfile.ZipFile(OUTPUT_FILE, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk('res'):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = file_path.replace(os.sep, '/')
                    zipf.write(file_path, arc_name)
                    print('  + {}'.format(arc_name))
        print('OK: packaged {}'.format(OUTPUT_FILE))
        return True
    except Exception as exc:
        print('ERROR: package failed: {}'.format(exc))
        return False


if __name__ == '__main__':
    success = build_mod()
    sys.exit(0 if success else 1)
