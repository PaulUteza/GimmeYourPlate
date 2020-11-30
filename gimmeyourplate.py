#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import wpod
import supervisely
import anpr_ocr_prediction
import handwritten
from PIL import Image
import numpy as np

PAGE_CONFIG = {"page_title":"Gimme Your Plate","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

MODEL_PATH = 'anpr_models/'


def create_read_plate():
	img_file_buffer_plate = st.file_uploader("Upload an image with a plate in the box below", type=["png", "jpg", "jpeg"])
	if img_file_buffer_plate is not None:
		image_plate = np.array(Image.open(img_file_buffer_plate))
		st.image(image_plate, use_column_width=True)
		model = ['WPOD-NET', 'SUPERVISELY']
		model_choice = st.selectbox('Choose the model :', model)
		plates = None
		if model_choice == 'WPOD-NET':
			wpod_model = MODEL_PATH + 'wpod/wpod-net'
			dmin_value = st.slider("Adjust this value for a better detection", 100, 1000, 256)
			assertion_raised = False
			while plates is None and not assertion_raised:
				try:
					box_image, plates = wpod.make_prediction(image_plate, wpod_model, dmin_value)
					st.pyplot(box_image)
					for plate in plates:
						plate_to_show = plate[..., ::-1]
						st.image(plate_to_show)
						ocr_plate = anpr_ocr_prediction.make_predictions(plate)
						st.write(ocr_plate[0])
				except AssertionError:
					st.write('No plate detected ! Try to adjust the value.')
					assertion_raised = True

		else:
			supervisely_model = MODEL_PATH + 'supervisely/model'
			box_image, plates = supervisely.make_prediction(image_plate, supervisely_model)
			st.pyplot(box_image)
			plates = np.array(plates)
			st.image(plates, use_column_width=True)
			ocr_plate = anpr_ocr_prediction.make_predictions(plates)
			st.write(ocr_plate[0])


def create_handwritten():
	img_file_buffer_handwritten = st.file_uploader("Upload an image with handwritten text", type=["png", "jpg", "jpeg"])
	if img_file_buffer_handwritten is not None:
		image_handwritten = np.array(Image.open(img_file_buffer_handwritten))
		st.image(image_handwritten, use_column_width=True)
		fig, no_spell, with_spell = handwritten.make_predict(image_handwritten)
		st.write('Word segmentation :')
		st.pyplot(fig)
		st.write('Without Spell : '+no_spell)
		st.write('With Spell : '+with_spell)

def main():
	st.title("Artificial Intelligence Project")
	st.subheader("Plates Detection using object detection and Handwritting recognition")
	menu = ['Read a plate', 'Handwritting recognition',"About"]
	choice = st.sidebar.selectbox('Menu',menu)
	if choice == 'Read a plate':
		create_read_plate()
	if choice =='Handwritting recognition':
		create_handwritten()

if __name__ == '__main__':
	main()
