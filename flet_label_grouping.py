import flet as ft
import pandas as pd
import pickle as pckl
import os
from datetime import datetime,date





def load_id_df():
    id_df = pd.read_json('id_df.json',orient='records', lines=True)
    return id_df

def load_pre_sorted():
    now = datetime.now()
    bpath = './data'
    if any(['labelled_words' in item for item in os.listdir(bpath)]):
        most_recent_str = get_most_recent_filename(bpath)
        words_labelled = pd.read_json(f"{bpath}/labelled_words_{most_recent_str}.json",orient='records', lines=True)
        file_name = f'{bpath}/labelled_words_{most_recent_str+1}.json' 
    else:
        words_labelled = pd.DataFrame(data=None, columns = ['word','synonym'])
    return words_labelled, file_name

def get_most_recent_filename(bpath):
    valid_items = [item.rstrip('.json').split('_')[-1] for item in os.listdir(bpath) if 'labelled_words' in item]
    most_recent = max([int(item) for item in valid_items])
    return most_recent


def load_data():
    id_df = load_id_df()
    psort, new_name = load_pre_sorted()
    return id_df, psort, new_name
def prep_data(e):
    existing_labels = e.control.data
    if len(existing_labels)>=1:
        for item in existing_labels:
            old_entries.current.controls.append(
                ft.ElevatedButton(item, on_click = btn_click)
            )


def main(page: ft.Page):
    page.title = "graph-ade synonym builder"
    new_entry = ft.Ref[ft.TextField]()
    new_entry_btn = ft.Ref[ft.ElevatedButton]()
    old_entries = ft.Ref[ft.Column]()
    current_selection = ft.Ref[ft.Column]()
    next_btn = ft.Ref[ft.ElevatedButton]()
    word_text = ft.Ref[ft.Text]()
    start_btn = ft.Ref[ft.ElevatedButton]()

    id_df, lwords, file_name = load_data()

    word_candidates = id_df.class_name.values
    word_candidates = [item for item in word_candidates if item not in lwords.word.values]
    if len(lwords)>0:

        existing_labels = sorted(list(set([item for items in lwords.synonym for item in items])))
    else:
        existing_labels = []
    print(existing_labels)
    # txt_number = ft.TextField(value="0", text_align=ft.TextAlign.RIGHT, width=100)
    item_name = 'box thingy'
    item_name = word_candidates[0]
    current_syn = []
    def add_text(e):
        if new_entry.current.value not in existing_labels:
            old_entries.current.controls.append(
                ft.ElevatedButton(new_entry.current.value, on_click = btn_click,data = len(old_entries.current.controls)),
                )
            current_selection.current.controls.append(
                ft.ElevatedButton(f"{new_entry.current.value}",on_click = remove_click, data = len(current_selection.current.controls)))
            existing_labels.append(new_entry.current.value)        
            current_syn.append(new_entry.current.value)
            new_entry.current.value = ''
            page.update()
    
    def  btn_click(e):
        current_selection.current.controls.append(ft.ElevatedButton(f"{existing_labels[e.control.data]}",on_click = remove_click,data = len(current_selection.current.controls)))
        current_syn.append(existing_labels[e.control.data])
        page.update()

    def remove_click(e): 
        del current_selection.current.controls[e.control.data]
        del current_syn[e.control.data]
        for i in range(0,len(current_selection.current.controls)):
            current_selection.current.controls[i].data = i
            current_selection.current.controls[i].label = f"{current_syn[i]}"
        page.update()

    def entry_button(e):
        current_syn = e.control.data
        lwords.loc[len(lwords)] = [word_candidates[0], current_syn]
        lwords.to_json(file_name,orient='records', lines=True)

        word_candidates.pop(0)
        word_text.current.value = word_candidates[0]
        current_syn = []
        current_selection.current.controls = []
        page.update()

    def prep_data(e):
        existing_labels = e.control.data
        if len(existing_labels)>=1:
            for ii, item in enumerate(existing_labels):
                print(item)
                old_entries.current.controls.append(
                    ft.ElevatedButton(item, on_click = btn_click,data=ii )
                )
        page.update()
    #start_btn.disabled = True



    input_text = ft.Column([
        ft.Text(f"{item_name}", ref = word_text),
        ft.TextField(ref = new_entry, label = 'custom name'),
        ft.ElevatedButton('Accept', ref = new_entry_btn, on_click = add_text)
    ])
#    input_text.current.controls.append(ft.Text(f"{item_name}"))
#    input_text.current.controls.append(ft.TextField(ref = new_entry, label = 'custom name'))
#    input_text.current.controls.append(ft.ElevatedButton('Accept',ref = new_entry_btn ))

    page.add(
        ft.Row(
            [   ft.ElevatedButton('Load Data', on_click=prep_data, data = existing_labels),
                input_text,
                ft.ElevatedButton('Save',on_click=entry_button, data = current_syn),
                ft.Container(
                ft.Column(ref = old_entries,
                        scroll=ft.ScrollMode.AUTO,
                         ),
                expand=False,
                margin=5,
                padding=5,
                bgcolor=ft.colors.DEEP_PURPLE_300,
                border_radius=10,
                alignment=ft.alignment.top_left,
                ),
                ft.Column(ref = current_selection)
            ], expand=True
        )
    )

ft.app(port=8550,view=ft.AppView.WEB_BROWSER, target=main)
