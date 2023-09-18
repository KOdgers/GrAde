import flet as ft
import pandas as pd
import json
import os
from datetime import datetime,date
from flet.matplotlib_chart import MatplotlibChart
from grade_util import *
import matplotlib
import matplotlib.pyplot as plt
import requests as req
from dataclasses import dataclass
from icecream import ic
  
    
@dataclass
class Image_Maps:
    """Class for tracking binary map, and map tracking object instances"""
    image:np.ndarray
    binary_mask:np.ndarray
    entity_mask:np.ndarray
    binary_preview_mask:np.ndarray
    preview_mask:np.ndarray
    instance_mask:np.ndarray
    entities:list
    preview_indices:list
    image_graph:list
    image_idx:int

    def add_entity_check_from_idx(self,idx):
        if idx in self.entities:
            self.remove_entity_from_idx(idx)
            return False
        else:
            return True 

    def remove_entity_from_idx(self,idx):
        mask = np.where(self.instance_mask==idx, 0,1)
        self.binary_mask = mask*self.binary_mask
        self.entity_mask *=self.binary_mask
        self.entities.remove(idx)
        self.remove_preview()

    def add_from_loc(self,loc):
        ent_val = self.instance_mask[loc[0],loc[1]]
        self.add_from_idx(ent_val)

    def add_from_idx(self,idx):
        if self.add_entity_check_from_idx(idx):
            mask_img = np.where(self.instance_mask==idx,idx,0)
            ic(mask_img.shape, self.entity_mask.shape)
            self.entity_mask += mask_img
            self.binary_mask = np.clip(self.entity_mask,0,1)
            self.entities.append(idx)
            self.recalc_preview()


    def add_preview(self,idx):
        new_temp_indices = [j for i in self.image_graph if idx in i for j in i if j not in self.preview_indices]
        if idx in new_temp_indices:
            new_temp_indices.remove(idx)
        
        new_temp_indices = ([new_temp_indices] if type(new_temp_indices)!=list else new_temp_indices)
        if len(new_temp_indices)>0:
            for idx in new_temp_indices:
                mask_img = np.where(self.instance_mask==idx,idx,0)
                self.preview_mask += mask_img
                self.binary_preview_mask = np.clip(self.preview_mask,0,1)


    def remove_preview(self):
        self.recalc_preview()

    def recalc_preview(self):
        adjacent_idx = [j for i in self.image_graph if all([ii in i for ii in self.entities]) for j in i]
        adjacent_idx = list(set(adjacent_idx))
        if len(adjacent_idx)>0:
            self.preview_mask = sum(list(self.instance_mask==i for i in adjacent_idx))
        else:
            self.preview_mask*=0
        self.binary_preview_mask = np.clip(self.preview_mask,0,1)
        
    def save_group_state(self):
        self.image_graph.append(self.entities)
        self.entities=[]
        self.preview_mask*=0
        self.binary_preview_mask*=0
        self.entity_mask*=0
        self.binary_mask*=0


matplotlib.use("svg")

def load_id_df():
    id_df = pd.read_json('id_df.json',orient='records', lines=True)
    return id_df

def load_data():
    datapaths = {}
    datapaths['ade_root'] = '/home/kelly/Documents/Projects/EntS/ADE/pregraph'
    datapaths['ade_index'] = 'index_ade20k.pkl'
    data, id_df = get_data(datapaths)

    return data,id_df,datapaths

def byte_to_plt(data,img_shape):
    gfig = np.frombuffer(data['image'],dtype=np.unit8).reshape(figure_data['img_shape'])
    seg_fig, seg_ax = plt.subplots(figsize=(10,10))
    seg_fig.subplots_adjust(0,0,1,1)
    seg_ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),cmap='gray')
    seg_ax.imshow(segment_lines,alpha=segment_lines)
    seg_ax.axis('off')
    seg_ax.margins(0,0)
    return seg_fig

def parse_rest_graphs(sgraph_json):
    vals = sgraph_json.json()
    unique_mask_keys = list(set([j for i in sgraph_json for j in i]))

    return vals 






def main(page: ft.Page):
    page.title = "graph-ade entity builder"
    page.theme = ft.Theme(
        color_scheme_seed=ft.colors.ORANGE,
    )
    new_entry = ft.Ref[ft.TextField]()
    new_entry_btn = ft.Ref[ft.ElevatedButton]()
    image_mpl = ft.Ref[ft.matplotlib_chart.MatplotlibChart]()
    image_name_ref = ft.Ref[ft.Text]()
    num_graphs_ref = ft.Ref[ft.Text]()
    collection_type_ref = ft.Ref[ft.Text]()


    data,id_df,datapaths =  load_data()
    ic()
    current_syn = []
    def add_item(e: ft.TapEvent):
        loc_x = int((e.local_x-4)*2)
        loc_y = int((e.local_y-4)*2)
        update_masks(current_map_data,[loc_y,loc_x])
        print(loc_x,loc_y)


    def save_new_group_state(e):
        current_map_data.save_group_state()
        base_fig = current_map_data.image
        mask_fig = current_map_data.binary_mask
        update_fig, update_ax = plt.subplots(figsize=(10,10))
        update_fig.subplots_adjust(0,0,1,1)
        update_ax.imshow(base_fig)
        update_ax.imshow(mask_fig,alpha=mask_fig)
        update_ax.axis('off')
        update_ax.margins(0,0)
        display_figure.figure = update_fig
        display_figure.update()
        page.update()

        

    def update_masks(current_map_data,location):
#        figure_data = req.get(f'http://127.0.0.1:8000/images/masks/19/{location[0]}_{location[1]}')
        current_map_data.add_from_loc(location)
#        arr = np.frombuffer(figure_data.content, np.uint8)
#        mask_fig = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
#        current_map_data.add_entity(mask_fig,current_pic_graphs)
        base_fig = current_map_data.image
        mask_fig = current_map_data.binary_mask
        preview_mask = current_map_data.binary_preview_mask
        figsize_x = (base_fig.shape[0]/base_fig.shape[1])*10
        update_fig, update_ax = plt.subplots(figsize=(10,figsize_x))
        update_fig.subplots_adjust(0,0,1,1)
        update_ax.imshow(base_fig)
        ic(np.max(preview_mask))
        if np.max(preview_mask)>0:
            ic(np.max(preview_mask))
            update_ax.imshow(preview_mask, alpha = preview_mask*0.5,cmap='Oranges')
        update_ax.imshow(mask_fig,alpha=mask_fig*0.7,cmap='Greens')
        update_ax.axis('off')
        update_ax.margins(0,0)
        display_figure.figure = update_fig
        display_figure.update()
        page.update()
        
    if 'current_map_data' not in locals():
        ic()
        figure_data = req.get('http://127.0.0.1:8000/images/outlines/19')
        figure_graph_data = req.get('http://127.0.0.1:8000/images/graphs/19')
        figure_instance_data = req.get('http://127.0.0.1:8000/images/instances/19')
        ic()
        current_image_graphs = parse_rest_graphs(figure_graph_data)
        arr = np.frombuffer(figure_data.content, np.uint8)
        gfig = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        inst_arr = np.frombuffer(figure_instance_data.content, np.uint8)
        figsize_x = (gfig.shape[0]/gfig.shape[1])*10
        seg_fig, seg_ax = plt.subplots(figsize=(10,figsize_x))
        seg_fig.subplots_adjust(0,0,1,1)
        current_map_data= Image_Maps(image=gfig,
                                      binary_mask = np.zeros(gfig.shape[:2]),
                                      entity_mask = np.zeros(gfig.shape[:2]),
                                      preview_mask = np.zeros(gfig.shape[:2]),
                                      binary_preview_mask = np.zeros(gfig.shape[:2]),
                                      instance_mask = cv2.imdecode(inst_arr, cv2.IMREAD_UNCHANGED),
                                      entities= [],
                                      preview_indices=[],
                                      image_graph = [current_image_graphs],
                                      image_idx= 0)

        ic(current_map_data.instance_mask.shape)
        seg_ax.imshow(current_map_data.image)
        seg_ax.axis('off')
        seg_ax.margins(0,0)

        img_size = gfig.shape
        display_figure = MatplotlibChart(seg_fig,
                            ref=image_mpl, 
                            expand=True)

    def save_and_switch(e):
        current_map_data.image_idx+=1
        figure_data = req.get(f'http://127.0.0.1:8000/images/outlines/{current_map_data.image_idx}')
        figure_graph_data = req.get(f'http://127.0.0.1:8000/images/graphs/{current_map_data.image_idx}')
        figure_instance_data = req.get(f'http://127.0.0.1:8000/images/instances/{current_map_data.image_idx}')
        ic()
        current_image_graphs = parse_rest_graphs(figure_graph_data)
        arr = np.frombuffer(figure_data.content, np.uint8)
        gfig = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        inst_arr = np.frombuffer(figure_instance_data.content, np.uint8)
        current_map_data.image =gfig
        current_map_data.binary_mask = np.zeros(gfig.shape[:2])
        current_map_data.entity_mask = np.zeros(gfig.shape[:2])
        current_map_data.preview_mask = np.zeros(gfig.shape[:2])
        current_map_data.binary_preview_mask = np.zeros(gfig.shape[:2])
        current_map_data.instance_mask = cv2.imdecode(inst_arr, cv2.IMREAD_UNCHANGED)
        current_map_data.entities= []
        current_map_data.preview_indices=[]
        current_map_data.image_graph = [current_image_graphs]
        w0=gfig.shape[1]//2
        h0=gfig.shape[0]//2
        card.content.width=w0
        card.content.height=h0
        image_stack.width=w0
        image_stack.height=h0
        card.width=w0
        card.height=h0
        display_figure.width = w0
        display_figure.height = h0

        ic(gfig.shape)
        figsize_x = (gfig.shape[0]/gfig.shape[1])*10
        seg_fig, seg_ax = plt.subplots(figsize=(10,figsize_x))
        seg_fig.subplots_adjust(0,0,1,1)
        seg_ax.imshow(current_map_data.image)
        seg_ax.axis('off')
        seg_ax.margins(0,0)
        display_figure.figure = seg_fig 
        display_figure.update()
        card.content.update()
        card.update()
        image_stack.update()
        page.update()


    card = ft.GestureDetector(
           left=-4,
           top=-4,
           width=img_size[0]//2,
           height=img_size[1]//2,
           on_double_tap_down=add_item,
           content=ft.Container(bgcolor=ft.colors.BLUE,
                                width=img_size[0]//2, 
                                height=img_size[1]//2,
                                content= display_figure ),
            )   
    image_stack = ft.Stack([card],width=img_size[0]//2, height=img_size[1]//2)
    page.add(ft.Row([ft.Column([
                            ft.Text('image_name',ref=image_name_ref),
                            ft.Text('num_graphs',ref=num_graphs_ref),
                            ft.Text('collection_type',ref=collection_type_ref)
                            ]),
                    image_stack,
                     ft.Column([ft.ElevatedButton(text="Save Group",on_click=save_new_group_state),
                     ft.ElevatedButton(text="Delete Selections"),
                                ft.ElevatedButton(text="Next Image", on_click=save_and_switch,data=current_map_data),
                     ])
                     ]))
 

ft.app(port=8550,view=ft.AppView.WEB_BROWSER, target=main)

