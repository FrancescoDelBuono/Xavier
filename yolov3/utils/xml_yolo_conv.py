import argparse
import xml.etree.ElementTree as ET
import os


def xml_to_txt_yolo(annotation_path, out_path):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    tree = ET.parse(annotation_path)
    objs = tree.findall('object')
    coords = {}
    for obj in objs:
        id_ = obj.find('id').text
        for pol in obj.findall('polygon'):
            frame = pol.find('t').text
            points = []
            for pt in pol.findall('pt'):
                x = pt.find('x').text
                y = pt.find('y').text
                points.append([x, y])

            if frame in coords:
                coords[frame].update({id_: {'tl': points[0], 'br': points[2]}})
            else:
                coords[frame] = {id_: {'tl': points[0], 'br': points[2]}}
    vid = annotation_path[-8:-4]

    for frame, ids in coords.items():
        f = open('{}/{}_{}.txt'.format(out_path, vid, frame), 'w+')
        f.write("{}\n".format(len(coords[frame])))
        for pts in ids.values():
            f.write("{} {} {} {}\n".format(pts['tl'][0], pts['tl'][1], pts['br'][0], pts['br'][1]))
        f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert xml vatic file into txt yolo")
    parser.add_argument("xml_path")
    parser.add_argument("out_directory_path")
    args = parser.parse_args()
    xml_to_txt_yolo(args.xml_path, args.out_directory_path)
