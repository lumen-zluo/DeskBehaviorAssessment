import json
import os

name2id = {'ruler': 0, 'worksheet': 1, 'eraser': 2, 'pen': 3}


def convert(img_size, box):
    dw = 1. / (img_size[0])
    dh = 1. / (img_size[1])
    x = (box[0] + box[2]) / 2.0 - 1
    y = (box[1] + box[3]) / 2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def decode_json(json_floder_path, json_name):

    folder = r'C:\Users\Owner\Downloads\dataset\yolo\\'
    os.makedirs(folder,exist_ok=True)
    txt_name = os.path.join(folder,json_name[0:-5] + '.txt')
    txt_file = open(txt_name, 'w')

    json_path = os.path.join(json_floder_path, json_name)
    data = json.load(open(json_path, 'r', encoding='gb2312'))

    img_w = data['imageWidth']
    img_h = data['imageHeight']

    for i in data['shapes']:
        label_name = i['label']

        if label_name == 'Ruler':
            label_name = 'ruler'

        if label_name =='Worksheet':
            label_name = 'worksheet'

        if label_name =='Eraser':
            label_name ='eraser'

        if label_name =='Pen':
            label_name = 'pen'

        if (i['shape_type'] == 'polygon'):
            x1 = min([p[0] for p in i['points']])
            x2 = max([p[0] for p in i['points']])

            y1 = min([p[1] for p in i['points']])
            y2 = max([p[1] for p in i['points']])

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')

        if (i['shape_type'] == 'rectangle'):
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])

            bb = (x1, y1, x2, y2)
            bbox = convert((img_w, img_h), bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')

    txt_file.close()


if __name__ == "__main__":

    json_floder_path = r"C:\Users\Owner\Downloads\dataset\labels"
    json_names = os.listdir(json_floder_path)
    for json_name in json_names:
        decode_json(json_floder_path, json_name)
