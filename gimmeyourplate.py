#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import wpod
import supervisely
import anpr_ocr_prediction
from PIL import Image
import numpy as np

PAGE_CONFIG = {"page_title":"Gimme Your Plate","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

MODEL_PATH = 'anpr_models/'


def create_read_plate():
	img_file_buffer = st.file_uploader("Upload an image with a plate in the box below", type=["png", "jpg", "jpeg"])
	if img_file_buffer is not None:
		image = np.array(Image.open(img_file_buffer))
		st.image(image, use_column_width=True)
		model = ['WPOD-NET', 'SUPERVISELY']
		model_choice = st.selectbox('Choose the model :', model)
		plates = None
		if model_choice == 'WPOD-NET':
			wpod_model = MODEL_PATH + 'wpod/wpod-net'
			dmin_value = st.slider("Adjust this value for a better detection", 100, 1000, 256)
			assertion_raised = False
			while plates is None and not assertion_raised:
				try:
					box_image, plates = wpod.make_prediction(image, wpod_model, dmin_value)
					st.pyplot(box_image)
					for plate in plates:
						plate = plate[..., ::-1]
						st.image(plate)
						ocr_plate = anpr_ocr_prediction.make_predictions(plate)
						st.write(ocr_plate[0])
				except AssertionError:
					st.write('No plate detected ! Try to adjust the value.')
					assertion_raised = True

		else:
			supervisely_model = MODEL_PATH + 'supervisely/model'
			box_image, plates = supervisely.make_prediction(image, supervisely_model)
			st.pyplot(box_image)
			plates = np.array(plates)
			st.image(plates, use_column_width=True)
			ocr_plate = anpr_ocr_prediction.make_predictions(plates)
			st.write(ocr_plate[0])

def main():
	st.title("Artificial Intelligence Project")
	st.subheader("Plates Detection using object detection and Optical Character Recognition")
	menu = ['Read a plate', 'Handwritten recognition',"About"]
	choice = st.sidebar.selectbox('Menu',menu)
	if choice == 'Read a plate':
		create_read_plate()
if __name__ == '__main__':
	main()
