import os
import cv2
import yaml

# Paleta de colores BGR por clase (se cicla si hay más clases)
CLASS_COLORS = [
    (0, 0, 255),    # rojo
    (255, 0, 0),    # azul
    (0, 255, 0),    # verde
    (0, 255, 255),  # amarillo
    (255, 0, 255),  # magenta
    (255, 255, 0),  # cian
    (128, 0, 128),  # morado
    (0, 128, 255),  # naranja
    (128, 128, 0),  # oliva
    (0, 128, 128),  # teal
]

# === CONFIGURACIÓN ===
# Cargar configuración desde config.yaml
def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

# === FUNCIÓN PARA EXTRAER CLASES DESDE data.yaml ===
def load_classes_from_data_yaml(data_yaml_path):
    """Load class names from data.yaml file."""
    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"❌ data.yaml not found: {data_yaml_path}")
    
    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    if 'names' not in data:
        raise ValueError(f"❌ 'names' field not found in {data_yaml_path}")
    
    names = data['names']
    # Handle both list and dict formats
    if isinstance(names, dict):
        # If names is a dict like {0: 'class0', 1: 'class1'}, convert to list
        max_idx = max(names.keys()) if names else -1
        names_list = [names.get(i, f'class_{i}') for i in range(max_idx + 1)]
        return names_list
    elif isinstance(names, list):
        return names
    else:
        raise ValueError(f"❌ 'names' field must be a list or dict, got {type(names)}")

# === FUNCIÓN PARA DIBUJAR LOS BOUNDING BOXES ===
def draw_boxes(image_path, label_path, output_path, classes):
    # Leer imagen
    image = cv2.imread(image_path)
    if image is None:
        print(f"⚠️ No se pudo leer la imagen: {image_path}")
        return False
    h, w, _ = image.shape

    # Si no hay archivo de etiquetas, saltar
    if not os.path.exists(label_path):
        print(f"❌ No se encontró el label para: {os.path.basename(image_path)}")
        return False

    with open(label_path, "r") as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id, x_center, y_center, box_w, box_h = map(float, parts)

            # Convertir coordenadas normalizadas a píxeles
            x_center *= w
            y_center *= h
            box_w *= w
            box_h *= h

            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)

            # Dibujar rectángulo y etiqueta
            color = CLASS_COLORS[int(class_id) % len(CLASS_COLORS)] if CLASS_COLORS else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            class_name = classes[int(class_id)] if int(class_id) < len(classes) else f"cls_{int(class_id)}"
            cv2.putText(image, class_name, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Guardar la imagen con bounding boxes
    cv2.imwrite(output_path, image)
    return True

# === FUNCIÓN PARA PROCESAR UN SPLIT (train/valid/test) ===
def process_split(origin_dir, target_dir, split_name, classes):
    """Process a single split (train, valid, or test)."""
    images_dir = os.path.join(origin_dir, split_name, "images")
    labels_dir = os.path.join(origin_dir, split_name, "labels")
    output_dir = os.path.join(target_dir, split_name)
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Verificar que el directorio de imágenes existe
    if not os.path.exists(images_dir):
        print(f"  ⚠️ El directorio de imágenes no existe: {images_dir}")
        return 0
    
    # Procesar todas las imágenes
    image_count = 0
    
    for filename in os.listdir(images_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(images_dir, filename)
            label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + ".txt")
            output_path = os.path.join(output_dir, filename)
            
            if draw_boxes(image_path, label_path, output_path, classes):
                image_count += 1
    
    return image_count

# === FUNCIÓN PARA PROCESAR UN DATASET COMPLETO ===
def process_dataset(dataset_config):
    origin = dataset_config["origin"]
    target = dataset_config["target"]
    
    # Obtener nombre del dataset desde el path
    dataset_name = os.path.basename(os.path.normpath(origin))
    
    print(f"\n{'='*60}")
    print(f"📦 Procesando dataset: {dataset_name}")
    print(f"   Origen: {origin}")
    print(f"   Destino: {target}")
    print(f"{'='*60}")
    
    # Verificar que el directorio origen existe
    if not os.path.exists(origin):
        print(f"❌ El directorio origen no existe: {origin}")
        return
    
    # Cargar clases desde data.yaml
    data_yaml_path = os.path.join(origin, "data.yaml")
    try:
        classes = load_classes_from_data_yaml(data_yaml_path)
        print(f"📋 Clases encontradas ({len(classes)}): {classes}")
    except Exception as e:
        print(f"❌ Error al cargar clases: {e}")
        return
    
    # Procesar cada split (train, valid, test)
    splits = ["train", "valid", "test"]
    total_images = 0
    
    for split in splits:
        print(f"\n  🔄 Procesando split: {split}")
        count = process_split(origin, target, split, classes)
        total_images += count
        print(f"  ✅ {split}: {count} imágenes procesadas")
    
    print(f"\n✅ Dataset '{dataset_name}' completado: {total_images} imágenes procesadas en total")
    print(f"📁 Imágenes guardadas en: {target}")


# === CARGA Y PROCESAMIENTO ===
config = load_config()

# Verificar si hay una lista de datasets
if "datasets" in config:
    datasets = config["datasets"]
    print(f"🚀 Iniciando procesamiento de {len(datasets)} dataset(s)\n")
    
    for i, dataset in enumerate(datasets, 1):
        print(f"\n[{i}/{len(datasets)}]")
        process_dataset(dataset)
    
    print(f"\n{'='*60}")
    print("🎯 Proceso completo. Todos los datasets han sido procesados.")
    print(f"{'='*60}")
else:
    # Backward compatibility: verificar formato antiguo
    if "images_dir" in config and "labels_dir" in config and "output_dir" in config:
        print("⚠️ Formato de configuración antiguo detectado. Procesando como dataset único...")
        # Convertir formato antiguo al nuevo formato
        origin_dir = os.path.dirname(os.path.dirname(config["images_dir"]))
        dataset_config = {
            "origin": origin_dir,
            "target": config["output_dir"]
        }
        process_dataset(dataset_config)
    else:
        print("❌ Formato de configuración no reconocido. Por favor, use el formato con 'datasets' y 'origin'/'target'.")

