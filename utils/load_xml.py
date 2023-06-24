import xml.etree.ElementTree as ET


def load_coco_box(xml_path):
    boxes = []
    tree = ET.ElementTree(file=xml_path)
    obj_list = tree.findall('object')
    for obj in obj_list:
        name = obj.find('name').text        
        box = obj.find('bndbox')
        x = int(box.find('x').text)
        y = int(box.find('y').text)
        w = int(box.find('w').text)
        h = int(box.find('h').text)
        boxes.append([x, y, w, h, name])
    return boxes
