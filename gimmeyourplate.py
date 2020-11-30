#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import wpod
import supervisely
import anpr_ocr_prediction
import handwritten
import plate_ocr
from PIL import Image
import numpy as np

PAGE_CONFIG = {"page_title":"Gimme Your Plate","page_icon":":smiley:","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)

MODEL_PATH = 'anpr_models/'


def create_read_plate():
	st.subheader("Detect license plate in the wild")
	img_file_buffer_plate = st.file_uploader("Upload an image with a plate in the box below", type=["png", "jpg", "jpeg"])
	if img_file_buffer_plate is not None:
		image_plate = np.array(Image.open(img_file_buffer_plate))
		st.image(image_plate, use_column_width=True)
		model = ['WPOD-NET', 'SUPERVISELY']
		model_choice = st.selectbox('Choose the model :', model)
		model_ocr = ['OpenCV&MobileNet', 'SUPERVISELY']
		model_ocr_choice = st.selectbox('Choose the OCR model :', model_ocr)
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
						if model_ocr_choice == 'OpenCV&MobileNet':
							ocr_plate, segmented_plot = plate_ocr.make_predictions(plate)
							st.pyplot(segmented_plot)
						if model_ocr_choice == 'SUPERVISELY':
							ocr_plate = anpr_ocr_prediction.make_predictions(plate)
							ocr_plate = ocr_plate[0]
						print(ocr_plate)
						st.write('Predicted plate : ' + str(ocr_plate))
				except AssertionError:
					st.write('No plate detected ! Try to adjust the value.')
					assertion_raised = True

		else:
			supervisely_model = MODEL_PATH + 'supervisely/model'
			box_image, plates = supervisely.make_prediction(image_plate, supervisely_model)
			st.pyplot(box_image)
			plates = np.array(plates)
			st.image(plates, use_column_width=True)
			if model_ocr_choice == 'OpenCV&MobileNet':
				ocr_plate, segmented_plot = plate_ocr.make_predictions(plates)
				st.pyplot(segmented_plot)
			if model_ocr_choice == 'SUPERVISELY':
				ocr_plate = anpr_ocr_prediction.make_predictions(plates)
				ocr_plate = ocr_plate[0]
			st.write(ocr_plate)


def create_handwritten():
	st.subheader("Word recognition of handwritten text")
	img_file_buffer_handwritten = st.file_uploader("Upload an image with handwritten text", type=["png", "jpg", "jpeg"])
	if img_file_buffer_handwritten is not None:
		image_handwritten = np.array(Image.open(img_file_buffer_handwritten))
		st.image(image_handwritten, use_column_width=True)
		fig, no_spell, with_spell = handwritten.make_predict(image_handwritten)
		st.write('Word segmentation :')
		st.pyplot(fig)
		st.write('Without Spell : '+no_spell)
		st.write('With Spell : '+with_spell)


def create_about():

	st.markdown('## Artificial Intelligence Project')

	st.markdown('### This project was created by Laurent Feroldi, Marcel Mounsi, Rayhane Talbi, Paul Uteza.')

	st.markdown('Link to the GitHub repository can be found [here](https://github.com/PaulUteza/GimmeYourPlate)')




def main():
	st.title("Gimme Your Plate")
	menu = ['Read a plate', 'Handwriting recognition',"About"]
	choice = st.sidebar.selectbox('Menu',menu)
	image_plate = None
	image_handwritten = None
	if choice == 'Read a plate':
		create_read_plate()
	if choice =='Handwriting recognition':
		create_handwritten()
	if choice == 'About':
		create_about()

if __name__ == '__main__':
	main()
