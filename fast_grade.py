from typing import Union,Annotated
from fastapi.encoders import jsonable_encoder

from fastapi import FastAPI, Request, Response, Query
from fastapi.responses import PlainTextResponse
import io
import os
import cv2
import sys
import matplotlib.pyplot as plt
import pandas as pd
from grade_util import get_data, loadAde20K, seg_to_lines
import json
sys.path.append('/home/kelly/Documents/Projects/EntS')
from utils_ade20k import loadAde20K
import numpy as np
from datetime import date

from PIL import Image

app = FastAPI()


DATAPATHS = {}
DATAPATHS['ade_root'] = '/home/kelly/Documents/Projects/EntS/ADE/pregraph'
DATAPATHS['ade_index'] = 'index_ade20k.pkl'
ade_index, id_df = get_data(DATAPATHS)
bpath = '.'
if 'annotated_df.pckl'  in os.listdir(f'{bpath}'):
    Annotated_DF = pd.read_pickle(f'{bpath}/annotated_df.pckl')
else:
    Annotated_DF = pd.DataFrame.from_dict({key:[''] for key in 'image_id elements date_added'.split(' ')})


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get('item/{item_id}',response_class=PlainTextResponse)
def get_existing_values(item_id):
    if item_id not in Annotated_DF.image_id.values:
        return '' 
    else:
        slim_df = Annotated_DF[Annotated_DF.image_id==item_id]
        graphs = [item for item in slim_df['elements']]
        graph_str = '|'.join(graphs)
        return graph_str 

@app.put('/images/add_graph/{pic_id}/{group_string}')
async def add_group(pic_id:int, group_string:str):
    group_string
    temp_df = pd.DataFrame.from_dict(
        {'image_id':str(pic_id),
         'elements': group_string,
         'date_added': str(date.today())
         }
    )
    Annotated_DF = Annotated_DF.append(temp_df,ignore_index=True)
    return {"status": "SUCCESS"}

@app.get('/images/graphs/{pic_id}')
def get_existing_subgraphs(pic_id:int):
    print(Annotated_DF.keys())
    if pic_id in Annotated_DF.image_id:
        slim_df = Annotated_DF[Annotated_DF.image_id==pic_id]
        annotated_json = json.dumps(slim_df['elements'].values)
    else:
        annotated_json = json.dumps([])
    return Response(jsonable_encoder(annotated_json), media_type="application/json")


@app.get("/images/masks/{pic_id}/{x_loc}_{y_loc}")
def get_inst_mask_from_point(pic_id:int,x_loc:int,y_loc:int):
    root_path = DATAPATHS['ade_root']
    img_name = '{}/{}'.format(ade_index['folder'][pic_id],ade_index['filename'][pic_id])
    info = loadAde20K('{}/{}'.format(root_path, img_name))
    segments =info['instance_mask']
    ent_val = segments[x_loc,y_loc]
    mask_img = np.where(segments==ent_val,ent_val,0)
    success, im = cv2.imencode('.png', mask_img)
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im.tobytes(), headers=headers, media_type='image/png')


@app.get("/images/{pic_id}")
def get_image_rgb(pic_id:int):
    root_path = DATAPATHS['ade_root']
    img_name = '{}/{}'.format(ade_index['folder'][pic_id],ade_index['filename'][pic_id])
    info = loadAde20K('{}/{}'.format(root_path, img_name))
    img = cv2.imread(info['img_name'])[:,:,::-1]
    success, im = cv2.imencode('.png', img)
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im.tobytes(), headers=headers, media_type='image/png')

@app.get("/images/instances/{pic_id}")
def get_image_instance_mask(pic_id:int):
    root_path = DATAPATHS['ade_root']
    img_name = '{}/{}'.format(ade_index['folder'][pic_id],ade_index['filename'][pic_id])
    info = loadAde20K('{}/{}'.format(root_path, img_name))
    img =info['instance_mask']
    success, im = cv2.imencode('.png', img)
    img_shape = img.shape
    print(img_shape)
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im.tobytes(), headers=headers, media_type='image/png')


@app.get("/images/outlines/{pic_id}")
def get_image(pic_id:int):
    root_path = DATAPATHS['ade_root']
    img_name = '{}/{}'.format(ade_index['folder'][pic_id],ade_index['filename'][pic_id])
    info = loadAde20K('{}/{}'.format(root_path, img_name))
    img = cv2.imread(info['img_name'])[:,:,::-1]
    segments =info['instance_mask']
    segment_lines = seg_to_lines(segments)
    img = np.where(np.repeat(segment_lines[:,:,np.newaxis],3,2).astype(bool),
                   np.stack([segment_lines*255,
                          segment_lines*255,
                          segment_lines*0],2),
                   img.astype('float64')
                   )
    success, im = cv2.imencode('.png', img)
    img_shape = img.shape
    headers = {'Content-Disposition': 'inline; filename="test.png"'}
    return Response(im.tobytes(), headers=headers, media_type='image/png')
#    return Response(img.tobytes())



@app.post("/graph/add")
async def add_image_subgraph(info: Request):
    new_graph_data = await info.json()
    pic_id = new_graph_data['pic_id']
    new_graph = [val for key,vale in new_graph_data['graph'].items]
    return {"status": "SUCCESS",
            "sizeOfGraph":len(new_graph)}



