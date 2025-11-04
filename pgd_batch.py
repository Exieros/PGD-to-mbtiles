import argparse
import os
import subprocess
import sys
from typing import List


def _option_present(args_list: List[str], option: str) -> bool:
    """Returns True if option ("--flag" or "--flag=value") is present in args_list."""
    option_eq = option + '='
    for token in args_list:
        if token == option:
            return True
        if token.startswith(option_eq):
            return True
    return False


def find_pgd_files(root: str) -> List[str]:
    """Рекурсивно ищет все файлы с расширением .pgd внутри каталога."""
    collected: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith('.pgd'):
                collected.append(os.path.join(dirpath, name))
    return collected


def run_pgd_assemble(pgd_path: str, extra_args: List[str], defaults) -> int:
    """Запускает pgd_assemble.py для указанного PGD-файла."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    assemble_script = os.path.join(script_dir, 'pgd_assemble.py')
    cmd = [sys.executable, assemble_script, '--pgd', pgd_path]
    
    if defaults.format and not _option_present(extra_args, '--format'):
        cmd.extend(['--format', defaults.format])
    if defaults.zoom_pad is not None and not _option_present(extra_args, '--zoom-pad'):
        cmd.extend(['--zoom-pad', str(defaults.zoom_pad)])

    has_out_name = _option_present(extra_args, '--out-name')
    if not has_out_name:
        default_out = pgd_path
        cmd.extend(['--out-name', default_out])

    cmd.extend(extra_args)
    print(f"\n=== Обработка {pgd_path} ===")
    result = subprocess.run(cmd, check=False)
    if result.returncode == 0:
        print(f"=== Успех: {pgd_path} ===\n")
    else:
        print(f"=== Ошибка ({result.returncode}): {pgd_path} ===\n")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Рекурсивно находит все PGD-файлы в каталоге и прогоняет их через pgd_assemble2.py.'
    )
    parser.add_argument(
        '--pgd-path',
        required=True,
        help='Каталог для поиска PGD-файлов (обрабатывается рекурсивно).'
    )
    parser.add_argument(
        '--format',
        choices=['webp', 'png'],
        default='webp',
        help='Формат, который будет передан pgd_assemble.py по умолчанию.'
    )
    parser.add_argument(
        '--zoom-pad',
        type=int,
        default=None,
        help='Значение --zoom-pad для pgd_assemble.py, если не переопределено через extra args.'
    )
    parser.add_argument(
        '--stop-on-error',
        action='store_true',
        help='Остановить обработку при первой ошибке.'
    )
    args, extra = parser.parse_known_args()

    target_dir = os.path.abspath(args.pgd_path)
    if not os.path.isdir(target_dir):
        raise SystemExit(f'Указанный путь не является каталогом: {target_dir}')

    pgd_files = find_pgd_files(target_dir)
    if not pgd_files:
        print('PGD-файлы не найдены.')
        return

    total = len(pgd_files)
    print(f'Найдено PGD-файлов: {total}')
    failures = 0

    for idx, pgd_file in enumerate(pgd_files, start=1):
        print(f'[{idx}/{total}]')
        rc = run_pgd_assemble(pgd_file, extra, args)
        if rc != 0:
            failures += 1
            if args.stop_on_error:
                print('Остановлено из-за ошибки.')
                break

    print('--- Итог ---')
    print(f'Всего файлов: {total}')
    print(f'Успешно: {total - failures}')
    print(f'С ошибками: {failures}')


if __name__ == '__main__':
    main()
