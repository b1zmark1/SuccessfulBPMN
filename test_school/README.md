preprocess.py подготавливает входное изображение диаграммы так, чтобы дальше можно было двумя разными путями решать две разные задачи:
OCR-путь — получить изображение, где текст читается максимально стабильно.
Geometry/CV-путь — получить изображение, где формы (рамки, окружности, границы контейнеров, стрелки) выделяются максимально устойчиво, а мелкий шум/текст меньше мешает.
Он не извлекает узлы и рёбра, он только генерирует «сырьё» для следующих этапов.

вход python diagram_cv/preprocess.py --input <path_to_jpg_png> --outdir <output_dir>
выход возвращает 3 строки-пути:
ocr_path → outdir/04_ocr_ready.png
cv_path → outdir/05_cv_binary.png
geom_path→ outdir/06_geom.png

00_original.png - исходник (после ресайза)
01_gray.png - grayscale версия
02_denoised.png - после контраста + шумоподавления
04_ocr_ready.png - бинаризация adaptive threshold
05_cv_binary.png - бинаризация Otsu
06_geom.png - морфология для «цельной геометрии»


Конфиг PreprocessConfig
max_side=1800 — ограничение размера изображения по максимальной стороне (ускорение и стабилизация)
clahe_clip_limit, clahe_tile_grid_size — параметры CLAHE (локальный контраст)
denoise_h — сила fastNlMeansDenoising
adaptive_block, adaptive_c — параметры adaptive threshold для OCR ветки
close_kernel — размер ядра для 06_geom.png (морфологическое “залечивание”)

1) Чтение изображения
cv2.imread(..., IMREAD_COLOR)
если не прочитал — ошибка
2) Масштабирование (resize)
если max(h,w) > max_side, уменьшает пропорционально.
сохраняет как 00_original.png
3) Перевод в grayscale
01_gray.png
4) Нормализация контраста (CLAHE)
увеличивает локальный контраст, особенно полезно если фон неравномерный
5) Подавление шума
fastNlMeansDenoising
сохраняет 02_denoised.png
6) OCR-ready ветка: 04_ocr_ready.png
adaptiveThreshold (Gaussian)
рассчитано на то, чтобы буквы не пропадали при неравномерном освещении
7) CV binary ветка: 05_cv_binary.png
Otsu threshold
бинарка “для геометрии”
8) Geometry ветка: 06_geom.png
инвертирует 05 (линии становятся белыми)
делает MORPH_CLOSE с ядром close_kernel×close_kernel
инвертирует обратно


diagram_cv/detect.py
Назначение
По входам из preprocess:
06_geom.png → найти узлы (элементы) и контейнеры
05_cv_binary.png → найти рёбра (линии/стрелки) и привязать их к узлам
Сохраняет:
10_overlay_nodes.png
11_overlay_edges.png
result.json
debug 20..23

вход
python diagram_cv/detect.py \
  --geom out/.../06_geom.png \
  --cvbin out/.../05_cv_binary.png \
  --original out/.../00_original.png \
  --outdir out/...
 выход
 10_overlay_nodes.png — bboxes узлов + контейнеров на оригинале
11_overlay_edges.png — найденные рёбра на оригинале
result.json — данные (узлы, контейнеры, рёбра)
20_linework.png — “чернила” (все чёрные пиксели из cvbin как белые на чёрном)
21_remove_mask.png — что вырезаем как области узлов/контейнеров
22_connectors.png — оставшиеся после вырезания пиксели, которые считаются “коннекторами”
23_connectors_skeleton.png — скелетизация коннекторов


Алгоритм detect_nodes(geom)
Вход
geom_gray = 06_geom.png (grayscale/binary)
Логика
приводит к бинарному виду “чёрное по белому”
инвертирует → контуры становятся белыми
ищет контуры с иерархией RETR_CCOMP
берёт только те контуры:
parent == -1 (внешние)
child != -1 (есть внутренняя “дырка”)
Это сделано потому что большинство узлов BPMN — это рамка/контур (а значит “дырка” внутри).
фильтры по площади:
area >= min_node_area
area/img_area <= max_node_area_ratio
сортирует по (y,x)
классифицирует фигуры (_classify_shape):
circle — высокая circularity + почти квадратный bbox
diamond — 4 вершины + угол minAreaRect похож на поворот
rectangle — 4 вершины
rounded_rect — >=5 вершин
unknown — иначе
определяет контейнер:
bbox_area_ratio в заданном диапазоне
bbox достаточно большой по ширине/высоте
kind подходит
Алгоритм detect_edges(cvbin, nodes)
Вход
cvbin_gray = 05_cv_binary.png
elements/containers = результат detect_nodes
Логика
приводит к бинарному виду
строит “чернила”:
linework_01 = (cvbin < 128) → 1 там где линии/текст/рамки
строит remove_mask:
прямоугольники bbox каждого узла + pad
прямоугольники bbox каждого контейнера + pad
вычитает mask:
connectors_01 = linework_01; connectors_01[mask>0]=0
Задача: убрать внутренности узлов/контейнеров, чтобы в коннекторах осталось больше “стрелок”.
делает MORPH_OPEN(3x3) чтобы убрать крошки
скелетизация:
пытается cv2.ximgproc.thinning (opencv-contrib)
иначе Zhang-Suen
строит граф пикселей скелета, ищет терминалы по степени deg != 2
трассирует пути между терминалами
endpoints привязывает к узлам по расширенному bbox (touch_dilate_px)
формирует edges:
если endpoint попал строго в один узел с каждой стороны → ok
иначе → ambiguous