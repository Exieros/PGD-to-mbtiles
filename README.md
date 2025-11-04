# PGD-to-mbtiles: Работа с проприетарным форматом PGD из AlpineQuest (Offline Maps)
Скрипт не связан с разработчиками AlpineQuest и предоставляется "как есть".

## Назначение
Конвертирует карты проприетарного формата PGD Alpine Quest (Offline Maps) в mbtiles [пакетно]
Если у вас на руках только файлы слоев PGD, вы можете конвертировать их в mbtiles, которое поддерживается преобладающим количеством картографических приложений, например Guru Maps на ios(так же опционально можно сгенерировать *.ms для него чтобы слой оторажался с наложением на карту)

## Установка зависимостей
```sh
python -m pip install -r requirements.txt
```

## Использование
```sh
python pgd_assemble.py --pgd path/to/file.pgd \
	[--out-name output_name] \
	[--format {webp|png}] \
	[--save-mosaic] \
	[--save-bounds] \
	[--save-ms] \
	[--zoom-pad N] \
	[--no-dedup]
```
- `--pgd` — входной PGD-файл (обязательный параметр).
- `--out-name` — базовое имя результатов (мозаика, mbtiles, `.ms`).
- `--format` — формат изображений (`webp` или `png`).
- `--save-mosaic` — сохранить промежуточную мозаику.
- `--save-bounds` — записать `*_bounds.json` с охватом карты.
- `--save-ms` — создать профиль Guru Maps (`*.ms`). GuruMaps игнорирует type=ovarlay в метаданных.
- `--zoom-pad` — отступ вниз от автоматически вычисленного максимального зума (нижние уровни строятся из тайлов максимального масштаба — без геометрического дрейфа).
- `--no-dedup` — отключить дедупликацию одинаковых тайлов.

```sh
python pgd_batch.py --pgd-path path/to/folder \
	[--format {webp|png}] \
	[--zoom-pad N] \
	[--stop-on-error] \
	[additional pgd_assemble options ...]
```
- `--pgd-path` — каталог для рекурсивного поиска `.pgd` (обязательный параметр).
- `--format` — формат, передаваемый каждому запуску `pgd_assemble.py`.
- `--zoom-pad` — значение по умолчанию для всех файлов.
- `--stop-on-error` — остановить пакет при первой ошибке.
- Дополнительные аргументы после известных опций (например, `--save-ms`) будут переданы непосредственно `pgd_assemble.py`.

**@soreixe**
**CC0**